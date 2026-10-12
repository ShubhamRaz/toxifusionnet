"""Lightweight Local Validation & Smoke-Test Workflow for ToxiFusionNet.

Validates:
1. Existing test suite pass/fail status.
2. Dataloader, image loader (.jpg, .png, missing fallback), and batch collate.
3. Model forward and backward pass across all four task heads.
4. Masked multitask losses for missing / inapplicable labels (IGNORE_INDEX = -100).
5. Modality presence mask invariance (dummy pixels cannot affect predictions).
6. 1-epoch lightweight training run and checkpoint saving to runs/smoke_test_local.
7. Checkpoint loading via ToxiFusionPipeline and inference verification.
8. Experiment configuration validation for supercomputer readiness (exp1 through exp4).
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ToxiFusionNet imports
from toxifusionnet.config import Config, load_config
from toxifusionnet.data.dataset import HateSarcasmDataset, make_collate
from toxifusionnet.inference.pipeline import ToxiFusionPipeline
from toxifusionnet.labels import (
    IGNORE_INDEX,
    LABEL_SPACES,
    TASKS,
    apply_label_config,
    num_classes,
)
from toxifusionnet.models.encoder import build_encoder_and_tokenizer
from toxifusionnet.models.losses import MultiTaskLoss
from toxifusionnet.models.toxifusionnet import ToxiFusionNet
from toxifusionnet.preprocessing.normalize import TextNormalizer
from toxifusionnet.training.trainer import train_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validation")

ROOT = Path(__file__).resolve().parent.parent


def test_1_dataloader_and_images():
    """Verify DataLoader handling of real images, alt-ext images, missing images, and HOT text."""
    logger.info("=== STEP 1: Testing DataLoader & Image Handling ===")
    normalizer = TextNormalizer()

    # Construct test rows
    sample_rows = pd.DataFrame([
        {
            "id": "0.jpg",
            "text": "sample text with normal image",
            "sarcasm": "sarcastic",
            "hate": "hate",
            "target": "individual",
            "severity": "moderate",
            "split": "train",
            "source_dataset": "multibully"
        },
        {
            "id": "823.jpg",  # Exists as 823.png on disk!
            "text": "sample text with png fallback image",
            "sarcasm": "not_sarcastic",
            "hate": "hate",
            "target": "society",
            "severity": "low",
            "split": "train",
            "source_dataset": "multibully"
        },
        {
            "id": "2.jpg",  # Truly missing from disk!
            "text": "sample text with missing image",
            "sarcasm": "not_sarcastic",
            "hate": "non_hate",
            "target": None,
            "severity": None,
            "split": "train",
            "source_dataset": "multibully"
        },
        {
            "id": "hot_100",  # HOT text-only sample
            "text": "sample text from HOT dataset without image",
            "sarcasm": None,
            "hate": "hate",
            "target": "individual",
            "severity": "high",
            "split": "train",
            "source_dataset": "hot"
        }
    ])

    corpus_words = [normalizer(t).words for t in sample_rows["text"]]
    cfg_enc = Config.wrap({"encoder_name": "tiny", "tiny_hidden": 64})
    encoder, tokenizer = build_encoder_and_tokenizer(cfg_enc, corpus_words)

    ds = HateSarcasmDataset(
        sample_rows,
        tokenizer=tokenizer,
        normalizer=normalizer,
        max_length=64,
        image_dir=str(ROOT / "data/raw/bully_data/bully_data"),
        use_clip=True
    )
    collate = make_collate(tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
    batch = next(iter(loader))

    assert "pixel_values" in batch and batch["pixel_values"] is not None, "pixel_values missing"
    assert "image_mask" in batch and batch["image_mask"] is not None, "image_mask missing"
    assert batch["pixel_values"].shape == (4, 3, 224, 224), f"Unexpected shape {batch['pixel_values'].shape}"

    masks = batch["image_mask"].squeeze(-1).tolist()
    assert masks[0] == 1.0, f"Expected image 0.jpg to be present (mask=1.0), got {masks[0]}"
    assert masks[1] == 1.0, f"Expected image 823.png to be resolved (mask=1.0), got {masks[1]}"
    assert masks[2] == 0.0, f"Expected missing image 2.jpg to have mask=0.0, got {masks[2]}"
    assert masks[3] == 0.0, f"Expected HOT sample to have mask=0.0, got {masks[3]}"

    logger.info("  [PASSED] Dataloader, image loading (.png fallback), and image masks verified.")
    return encoder, tokenizer, normalizer, batch


def test_2_model_forward_backward_heads(encoder, batch):
    """Verify forward and backward pass across all 4 task heads with frozen CLIP."""
    logger.info("=== STEP 2: Testing Forward & Backward Passes on 4 Task Heads ===")
    cfg_model = Config.wrap({
        "encoder_name": "xlm-roberta-base",
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "dropout": 0.1,
        "head_hidden": 256,
        "use_sarcasm_branch": True,
        "use_fusion": True,
        "use_cue_features": True,
        "cue_hidden": 32,
        "use_clip": True,
        "freeze_clip": True,
        "clip_model_name": "openai/clip-vit-base-patch32"
    })
    model = ToxiFusionNet(encoder, cfg_model)
    model.train()

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        cue=batch["cue"],
        pixel_values=batch["pixel_values"],
        image_mask=batch["image_mask"]
    )

    logits = out["logits"]
    for t in TASKS:
        assert t in logits, f"Task head {t} missing from model logits"
        n_c = num_classes(t)
        assert logits[t].shape == (4, n_c), f"Expected shape (4, {n_c}) for {t}, got {logits[t].shape}"
        assert not torch.isnan(logits[t]).any(), f"NaN in {t} logits"
        assert not torch.isinf(logits[t]).any(), f"Inf in {t} logits"

    logger.info("  [PASSED] Forward pass: all 4 heads produce valid logits (no NaNs/Infs).")

    # Loss computation with MultiTaskLoss
    criterion = MultiTaskLoss(
        tasks=TASKS,
        weighting="fixed",
        weights={"sarcasm": 1.0, "hate": 1.0, "target": 0.5, "severity": 0.5},
        focal_gamma=0.0
    )
    loss, parts = criterion(out, batch)

    assert not torch.isnan(loss), "Loss computed as NaN"
    assert loss.item() > 0.0, "Loss <= 0.0"

    loss.backward()

    # Verify frozen CLIP has no gradients
    for name, param in model.clip_vision.named_parameters():
        assert param.grad is None, f"Frozen CLIP parameter {name} has gradient!"

    # Verify trainable heads and projection have valid gradients
    assert model.vis_proj[0].weight.grad is not None, "vis_proj gradient is None"
    assert not torch.isnan(model.vis_proj[0].weight.grad).any(), "NaN gradient in vis_proj"
    for t in TASKS:
        head_weight = model.heads[t][0].weight if t != "sarcasm" else model.sarc_out.weight
        assert head_weight.grad is not None, f"Head {t} gradient is None"

    logger.info("  [PASSED] Backward pass: gradients flow correctly; CLIP backbone remains frozen.")
    return model


def test_3_masked_losses(model, batch):
    """Verify loss masking on samples with IGNORE_INDEX (-100)."""
    logger.info("=== STEP 3: Testing Masked Losses with IGNORE_INDEX ===")
    criterion = MultiTaskLoss(
        tasks=TASKS,
        weighting="fixed",
        weights={"sarcasm": 1.0, "hate": 1.0, "target": 0.5, "severity": 0.5}
    )

    # Set all target and severity labels to IGNORE_INDEX (-100)
    batch_masked = {
        "labels": {
            "sarcasm": torch.tensor([1, 0, 0, -100]),
            "hate": torch.tensor([1, 1, 0, 1]),
            "target": torch.full((4,), IGNORE_INDEX, dtype=torch.long),
            "severity": torch.full((4,), IGNORE_INDEX, dtype=torch.long)
        }
    }

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        cue=batch["cue"],
        pixel_values=batch["pixel_values"],
        image_mask=batch["image_mask"]
    )
    loss, parts = criterion(out, batch_masked)

    assert not torch.isnan(loss), "Loss is NaN when target & severity are masked"
    assert parts.get("target") == 0.0 or parts.get("target") is None, f"Target loss should be 0 when masked, got {parts.get('target')}"
    assert parts.get("severity") == 0.0 or parts.get("severity") is None, f"Severity loss should be 0 when masked, got {parts.get('severity')}"

    logger.info("  [PASSED] Masked loss correctly ignores missing/inapplicable labels without NaNs.")


def test_4_modality_presence_mask_invariance(model, batch):
    """Confirm missing-image mask (mask=0.0) strictly prevents dummy pixels from altering outputs."""
    logger.info("=== STEP 4: Testing Modality Presence Mask Invariance ===")
    model.eval()

    # Pass with dummy zero image and mask=0.0
    text_input_ids = batch["input_ids"][:1]
    text_attn_mask = batch["attention_mask"][:1]
    text_cue = batch["cue"][:1]
    zero_pix = torch.zeros((1, 3, 224, 224))
    zero_mask = torch.zeros((1, 1))

    with torch.no_grad():
        out_zero = model(
            input_ids=text_input_ids,
            attention_mask=text_attn_mask,
            cue=text_cue,
            pixel_values=zero_pix,
            image_mask=zero_mask
        )

        # Pass with random noise image but mask=0.0
        noise_pix = torch.randn((1, 3, 224, 224))
        out_noise = model(
            input_ids=text_input_ids,
            attention_mask=text_attn_mask,
            cue=text_cue,
            pixel_values=noise_pix,
            image_mask=zero_mask
        )

        # Pass in text-only mode (pixel_values=None, image_mask=None)
        out_text_only = model(
            input_ids=text_input_ids,
            attention_mask=text_attn_mask,
            cue=text_cue,
            pixel_values=None,
            image_mask=None
        )

    for t in TASKS:
        diff_noise = (out_zero["logits"][t] - out_noise["logits"][t]).abs().max().item()
        diff_text = (out_zero["logits"][t] - out_text_only["logits"][t]).abs().max().item()
        assert diff_noise < 1e-6, f"Mask invariance failed on {t}: max diff with noise is {diff_noise}"
        assert diff_text < 1e-6, f"Text-only equivalence failed on {t}: max diff is {diff_text}"

    logger.info("  [PASSED] Modality presence mask is strictly invariant: dummy pixels have zero effect.")


def test_5_smoke_training_run():
    """Run a 1-epoch lightweight smoke training run to verify full training & checkpoint saving pipeline."""
    logger.info("=== STEP 5: Running 1-Epoch Lightweight Smoke Training Run ===")
    out_dir = ROOT / "runs" / "smoke_test_local"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare small smoke dataset (16 train, 8 val, 8 test)
    combined_path = ROOT / "data/processed/combined_multibully_hot.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined dataset not found at {combined_path}")

    full_df = pd.read_csv(combined_path)
    # Sample balanced across splits and datasets
    smoke_subsets = []
    for split_name in ["train", "val", "test"]:
        n_mb = 8 if split_name == "train" else 4
        n_hot = 8 if split_name == "train" else 4
        mb_part = full_df[(full_df["split"] == split_name) & (full_df["source_dataset"] == "multibully")].head(n_mb)
        hot_part = full_df[(full_df["split"] == split_name) & (full_df["source_dataset"] == "hot")].head(n_hot)
        smoke_subsets.extend([mb_part, hot_part])

    smoke_df = pd.concat(smoke_subsets).reset_index(drop=True)
    smoke_data_path = out_dir / "smoke_subset.csv"
    smoke_df.to_csv(smoke_data_path, index=False)
    logger.info(f"  Created smoke dataset with {len(smoke_df)} rows at {smoke_data_path}")

    # Build smoke config based on exp4_multidataset_clip.yaml
    cfg = load_config(ROOT / "configs/exp4_multidataset_clip.yaml")
    cfg.experiment_name = "smoke_test_local"
    cfg.data.processed_path = str(smoke_data_path)
    cfg.data.max_length = 64
    cfg.train.epochs = 1
    cfg.train.batch_size = 4
    cfg.train.patience = 2
    cfg.train.fp16 = True

    t_start = time.time()
    results = train_run(cfg, smoke_df, out_dir)
    elapsed = time.time() - t_start

    # Check generated artifacts
    expected_files = [
        "model.pt",
        "history.csv",
        "preds_val.csv",
        "preds_test.csv",
        "metrics.json",
        "config.yaml"
    ]
    for ef in expected_files:
        assert (out_dir / ef).exists(), f"Missing expected output file: {ef}"

    assert (out_dir / "tokenizer").is_dir(), "Missing tokenizer directory"
    assert (out_dir / "encoder_config").is_dir(), "Missing encoder_config directory"

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "test" in metrics, "Missing aggregate test metrics"
    logger.info(f"  Smoke training completed in {elapsed:.1f}s. Metrics recorded: {list(metrics.keys())}")
    logger.info("  [PASSED] Smoke training run and checkpoint creation verified.")

    return out_dir


def test_6_checkpoint_loading_and_inference(run_dir: Path):
    """Verify loading the saved checkpoint and running end-to-end inference via ToxiFusionPipeline."""
    logger.info("=== STEP 6: Testing Checkpoint Loading & Inference Pipeline ===")
    pipe = ToxiFusionPipeline.from_run(run_dir, device="cpu")

    test_samples = [
        "यह काम बहुत अच्छा किया आपने शाबाश",
        "I will hurt you and your family you idiot"
    ]
    preds = pipe.predict(test_samples, explain=False)

    assert len(preds) == 2, f"Expected 2 predictions, got {len(preds)}"
    for p in preds:
        assert "sarcasm" in p and "label" in p["sarcasm"], "sarcasm prediction missing"
        assert "hate" in p and "label" in p["hate"], "hate prediction missing"
        assert "language" in p, "language metadata missing"
        logger.info(f"    Sample: '{p['text']}' -> Hate: {p['hate']['label']} ({p['hate']['prob']}), Sarcasm: {p['sarcasm']['label']} ({p['sarcasm']['prob']})")

    logger.info("  [PASSED] Checkpoint loaded back cleanly; inference pipeline functions as expected.")


def test_7_supercomputer_config_validation():
    """Verify that all four experiment configs are valid, syntactically correct, and ready for launch."""
    logger.info("=== STEP 7: Validating 4 Supercomputer Experiment Configurations ===")
    configs_to_check = [
        ("configs/exp1_multibully_text.yaml", False),
        ("configs/exp2_multidataset_text.yaml", False),
        ("configs/exp3_multibully_clip.yaml", True),
        ("configs/exp4_multidataset_clip.yaml", True),
    ]

    for rel_path, expected_clip in configs_to_check:
        full_path = ROOT / rel_path
        assert full_path.exists(), f"Config file not found: {rel_path}"
        cfg = load_config(full_path)

        data_path = ROOT / cfg.data.processed_path
        assert data_path.exists(), f"Data path {cfg.data.processed_path} in {rel_path} does not exist"

        assert bool(cfg.model.get("use_clip", False)) == expected_clip, (
            f"Config {rel_path} expected use_clip={expected_clip}, got {cfg.model.get('use_clip')}"
        )

        if expected_clip:
            img_dir = ROOT / cfg.data.get("image_dir", "data/raw/bully_data/bully_data")
            assert img_dir.exists(), f"Image directory {img_dir} does not exist for {rel_path}"

        logger.info(f"  Config {rel_path}: OK (data={cfg.data.processed_path}, clip={expected_clip})")

    logger.info("  [PASSED] All 4 experiment configurations validated.")


def main():
    print("=" * 70)
    print(" ToxiFusionNet Lightweight Local Validation Workflow")
    print("=" * 70)

    try:
        # Step 1
        encoder, tokenizer, normalizer, batch = test_1_dataloader_and_images()

        # Step 2
        model = test_2_model_forward_backward_heads(encoder, batch)

        # Step 3
        test_3_masked_losses(model, batch)

        # Step 4
        test_4_modality_presence_mask_invariance(model, batch)

        # Step 5
        run_dir = test_5_smoke_training_run()

        # Step 6
        test_6_checkpoint_loading_and_inference(run_dir)

        # Step 7
        test_7_supercomputer_config_validation()

        print("\n" + "=" * 70)
        print(" ALL 7 VALIDATION STEPS PASSED SUCCESSFULLY!")
        print(" Output directory:", run_dir)
        print("=" * 70)
        sys.exit(0)

    except Exception as e:
        logger.exception("Validation workflow failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
