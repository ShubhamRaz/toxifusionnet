"""Comprehensive integration and unit test suite for ToxiFusionNet multimodal multitask pipeline."""
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from toxifusionnet.data.dataset import HateSarcasmDataset, make_collate
from toxifusionnet.labels import IGNORE_INDEX, TASKS, LABEL_SPACES, apply_label_config
from toxifusionnet.models.clip import GatedVisionFusion, build_clip_vision
from toxifusionnet.models.encoder import BertConfig, BertModel, train_wordpiece_tokenizer
from toxifusionnet.models.losses import MultiTaskLoss
from toxifusionnet.models.toxifusionnet import ToxiFusionNet
from toxifusionnet.preprocessing.normalize import TextNormalizer
from toxifusionnet.training.metrics import evaluate_predictions, flatten_metrics
from toxifusionnet.training.trainer import predict_loader, predictions_frame


@pytest.fixture
def dummy_encoder_and_tok():
    words = [["hello", "world"], ["this", "is", "a", "test"], ["madarchod", "nahi", "bolna"]]
    tok = train_wordpiece_tokenizer(words, vocab_size=100)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=128, max_position_embeddings=128, pad_token_id=tok.pad_token_id)
    enc = BertModel(cfg)
    return enc, tok


def test_modality_presence_mask_invariance():
    """Verify that when image_mask is 0.0, GatedVisionFusion returns EXACTLY the text representation."""
    d = 64
    fusion = GatedVisionFusion(d)
    text_vec = torch.randn(4, d)
    vis_vec = torch.randn(4, d)

    # 1. Image mask all zeros
    image_mask_zeros = torch.zeros(4, 1)
    out_zeros = fusion(text_vec, vis_vec, image_mask_zeros)
    # Must be exactly text_vec
    assert torch.allclose(out_zeros, text_vec, atol=1e-7), "Zero image mask must leave text representation unaltered!"

    # 2. Image mask all ones
    image_mask_ones = torch.ones(4, 1)
    out_ones = fusion(text_vec, vis_vec, image_mask_ones)
    assert not torch.allclose(out_ones, text_vec), "Active image mask should fuse visual representation!"

    # 3. Mixed mask
    image_mask_mixed = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    out_mixed = fusion(text_vec, vis_vec, image_mask_mixed)
    assert torch.allclose(out_mixed[1], text_vec[1], atol=1e-7)
    assert torch.allclose(out_mixed[3], text_vec[3], atol=1e-7)
    assert not torch.allclose(out_mixed[0], text_vec[0])
    print("\n[PASS] Modality presence mask strict invariance verified.")


def test_missing_image_and_alt_extension_handling():
    """Test image resolution with alternate extension fallback and missing image safety."""
    repo_root = Path(__file__).resolve().parent.parent
    proc_csv = repo_root / "data" / "processed" / "multibully.csv"
    img_dir = repo_root / "data" / "raw" / "bully_data" / "bully_data"
    if not img_dir.is_dir():
        alt_img_dir = repo_root / "data" / "raw" / "bully_data"
        if alt_img_dir.is_dir():
            img_dir = alt_img_dir

    assert proc_csv.exists(), f"Processed multibully.csv must exist at {proc_csv}"
    assert img_dir.exists(), f"Bully data image directory must exist at {img_dir}"
    df = pd.read_csv(proc_csv).head(20)

    # Add a synthetic missing row
    df.loc[0, "id"] = "non_existent_image_12345.jpg"
    df.loc[0, "source_dataset"] = "multibully"

    normalizer = TextNormalizer()
    words = [normalizer(t).words for t in df["text"]]
    tok = train_wordpiece_tokenizer(words, vocab_size=200)

    ds = HateSarcasmDataset(df, tok, normalizer, max_length=32, image_dir=img_dir, use_clip=True)
    assert len(ds) == 20

    # Row 0 had missing image -> has_image == 0.0
    assert ds[0]["has_image"] == 0.0
    assert ds[0]["pixel_values"].shape == (3, 224, 224)

    # Check collate
    collate = make_collate(tok.pad_token_id)
    batch = collate([ds[i] for i in range(4)])
    assert batch["pixel_values"].shape == (4, 3, 224, 224)
    assert batch["image_mask"].shape == (4, 1)
    assert batch["image_mask"][0].item() == 0.0
    print("[PASS] Missing image resolution and collate verified.")


def test_loss_masking_with_ignore_index():
    """Verify that MultiTaskLoss ignores IGNORE_INDEX (-100) labels completely."""
    tasks = ["sarcasm", "hate", "target", "severity"]
    loss_fn = MultiTaskLoss(tasks=tasks, weighting="fixed", weights={t: 1.0 for t in tasks})

    logits = {
        "sarcasm": torch.randn(2, 2, requires_grad=True),
        "hate": torch.randn(2, 2, requires_grad=True),
        "target": torch.randn(2, 4, requires_grad=True),
        "severity": torch.randn(2, 3, requires_grad=True),
    }
    out = {"logits": logits, "attn": {"hate": torch.zeros(2, 5)}}

    # Row 0: all valid; Row 1: target and severity are masked (-100)
    batch = {
        "labels": {
            "sarcasm": torch.tensor([1, 0]),
            "hate": torch.tensor([1, 0]),
            "target": torch.tensor([0, IGNORE_INDEX]),
            "severity": torch.tensor([1, IGNORE_INDEX]),
        },
        "hate_attn_target": None, "hate_attn_valid": None
    }

    loss, parts = loss_fn(out, batch)
    assert not torch.isnan(loss), "Loss must not be NaN"
    loss.backward()

    # Target and severity logits for Row 1 must have 0 gradient (not supervised)
    assert (logits["target"].grad[1] == 0).all(), "Masked target sample must have 0 gradient"
    assert (logits["severity"].grad[1] == 0).all(), "Masked severity sample must have 0 gradient"
    # Row 0 must have non-zero gradient
    assert not (logits["target"].grad[0] == 0).all()
    print("[PASS] Masked multi-task loss with IGNORE_INDEX verified.")


def test_model_forward_and_backward_text_and_clip(dummy_encoder_and_tok):
    """Verify model forward pass in both text-only mode and CLIP multimodal mode."""
    apply_label_config({"target": ["individual", "organization", "community", "society"]})
    enc, tok = dummy_encoder_and_tok
    d = enc.config.hidden_size

    # 1. Text-only model
    cfg_text = {
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "use_sarcasm_branch": True,
        "use_fusion": True,
        "use_cue_features": True,
        "use_clip": False,
        "dropout": 0.1,
        "head_hidden": 64
    }
    model_text = ToxiFusionNet(enc, cfg_text)

    input_ids = torch.randint(0, len(tok), (3, 16))
    attn_mask = torch.ones(3, 16)
    cue = torch.randn(3, 16)

    out_text = model_text(input_ids, attn_mask, cue=cue)
    assert "hate" in out_text["logits"]
    assert "sarcasm" in out_text["logits"]
    assert "target" in out_text["logits"]
    assert "severity" in out_text["logits"]
    assert out_text["logits"]["hate"].shape == (3, 2)
    assert out_text["logits"]["target"].shape == (3, 4)

    # 2. Multimodal model with offline CLIP vision
    cfg_clip = {
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "use_sarcasm_branch": True,
        "use_fusion": True,
        "use_cue_features": True,
        "use_clip": True,
        "clip_offline": True,
        "clip_hidden_size": 128,
        "freeze_clip": True,
        "dropout": 0.1,
        "head_hidden": 64
    }
    model_clip = ToxiFusionNet(enc, cfg_clip)
    assert model_clip.use_clip is True

    pixel_values = torch.randn(3, 3, 224, 224)
    image_mask = torch.tensor([[1.0], [0.0], [1.0]])  # mixed batch

    out_clip = model_clip(input_ids, attn_mask, cue=cue, pixel_values=pixel_values, image_mask=image_mask)
    for t in ["hate", "sarcasm", "target", "severity"]:
        assert t in out_clip["logits"]
        assert out_clip["logits"][t].shape[0] == 3

    # Backward pass test
    loss = sum(lg.sum() for lg in out_clip["logits"].values())
    loss.backward()

    # Frozen CLIP parameters must have no gradients
    for p in model_clip.clip_vision.parameters():
        assert p.grad is None

    # Trainable fusion projection must have gradients
    assert model_clip.vis_proj[0].weight.grad is not None
    print("[PASS] Model forward and backward pass (text and CLIP) verified.")


def test_source_specific_evaluation():
    """Verify that predictions_frame and evaluate_predictions compute separate metrics for multibully and hot."""
    pred_data = {
        "id": ["mb_1", "mb_2", "hot_1", "hot_2"],
        "text": ["post 1", "post 2", "post 3", "post 4"],
        "source_dataset": ["multibully", "multibully", "hot", "hot"],
        "gold_hate": [1, 0, 1, 0],
        "pred_hate": [1, 0, 0, 0],
        "gold_sarcasm": [0, 1, 0, 0],
        "pred_sarcasm": [0, 1, 0, 1],
        "gold_target": [0, IGNORE_INDEX, 1, IGNORE_INDEX],
        "pred_target": [0, 0, 1, 0],
        "gold_severity": [1, IGNORE_INDEX, 2, IGNORE_INDEX],
        "pred_severity": [1, 0, 2, 0],
    }
    df = pd.DataFrame(pred_data)
    tasks = ["hate", "sarcasm", "target", "severity"]

    overall_res = evaluate_predictions(df, tasks)
    mb_res = evaluate_predictions(df[df.source_dataset == "multibully"], tasks)
    hot_res = evaluate_predictions(df[df.source_dataset == "hot"], tasks)

    # MultiBully had 100% hate accuracy (1, 0 vs 1, 0)
    assert mb_res["hate"]["macro_f1"] == 1.0
    # HOT had 50% hate accuracy (1, 0 vs 0, 0)
    assert hot_res["hate"]["macro_f1"] < 1.0
    print("[PASS] Source-specific evaluation decoupling verified.")


if __name__ == "__main__":
    pytest.main(["-s", __file__])
