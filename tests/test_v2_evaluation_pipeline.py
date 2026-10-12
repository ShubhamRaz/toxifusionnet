"""Automated tests for MultiBully v2 Evaluation Pipeline.

Covers:
1. Group leakage prevention across train/val/test in all 5 folds.
2. Rare-class support verification (high severity, society, organization).
3. CLIP loader fail-fast verification on research configurations.
4. Multimodal image tensor routing and zero-perturbation modality masking.
5. Loss masking with IGNORE_INDEX for non-hate conditional labels.
6. Bootstrap confidence interval computation for uncertainty estimation.
"""
import os
import pytest
import numpy as np
import pandas as pd
import torch

from toxifusionnet.data.grouped_splits import (
    assign_group_ids,
    audit_fold_integrity,
    make_grouped_cv_splits,
    normalize_ocr_text,
)
from toxifusionnet.labels import IGNORE_INDEX, TASKS
from toxifusionnet.models.clip import GatedVisionFusion, build_clip_vision
from toxifusionnet.models.encoder import BertConfig, BertModel, train_wordpiece_tokenizer
from toxifusionnet.models.losses import MultiTaskLoss
from toxifusionnet.models.toxifusionnet import ToxiFusionNet
from toxifusionnet.training.stats import bootstrap_ci


@pytest.fixture
def workspace_dir():
    return r"c:\Users\SHUBHAM RAJ\Desktop\Others\Toxic\toxifusionnet\toxifusionnet"


@pytest.fixture
def multibully_df(workspace_dir):
    csv_path = os.path.join(workspace_dir, "data", "processed", "multibully.csv")
    assert os.path.exists(csv_path), f"Processed CSV not found at {csv_path}"
    return pd.read_csv(csv_path)


def test_group_leakage_and_fold_integrity(multibully_df):
    """Verify zero group leakage across train, val, and test partitions in all 5 folds."""
    group_df, fold_dfs = make_grouped_cv_splits(multibully_df, n_splits=5, seed=42, val_frac=0.15)
    audit = audit_fold_integrity(fold_dfs, len(multibully_df))

    assert audit["passed"] is True, "Fold integrity audit must pass"
    assert audit["total_test_evaluated"] == len(multibully_df)
    assert audit["unique_test_evaluated"] == len(multibully_df)

    for f, d in enumerate(fold_dfs):
        tr_g = set(d[d["split"] == "train"]["group_id"])
        va_g = set(d[d["split"] == "val"]["group_id"])
        te_g = set(d[d["split"] == "test"]["group_id"])

        assert len(tr_g & te_g) == 0, f"Fold {f}: Train-Test group leakage detected!"
        assert len(va_g & te_g) == 0, f"Fold {f}: Val-Test group leakage detected!"
        assert len(tr_g & va_g) == 0, f"Fold {f}: Train-Val group leakage detected!"


def test_rare_class_support_across_all_folds(multibully_df):
    """Verify high severity, society, and organization have non-zero representation in all folds."""
    _, fold_dfs = make_grouped_cv_splits(multibully_df, n_splits=5, seed=42, val_frac=0.15)

    for f, d in enumerate(fold_dfs):
        for part in ["train", "val", "test"]:
            sub = d[d["split"] == part]
            high_sev = (sub["severity"] == "high").sum()
            soc = (sub["target"] == "society").sum()
            org = (sub["target"] == "organization").sum()

            assert high_sev > 0, f"Fold {f} {part} must have high severity samples (got 0)!"
            assert soc > 0, f"Fold {f} {part} must have society samples (got 0)!"
            assert org > 0, f"Fold {f} {part} must have organization samples (got 0)!"


def test_clip_loader_fails_fast_in_research_mode():
    """Verify build_clip_vision raises RuntimeError when pretrained loading fails and fallback is disabled."""
    # 1. Invalid Hub model name in research mode must fail clearly
    cfg_invalid = {
        "clip_model_name": "non_existent/non_existent_clip_checkpoint_xyz",
        "clip_offline": False,
        "clip_allow_random_fallback": False,
    }
    with pytest.raises(RuntimeError) as exc_info:
        build_clip_vision(cfg_invalid)
    assert "Failed to load pretrained CLIP model" in str(exc_info.value)
    assert "Research runs require genuine pretrained weights" in str(exc_info.value)

    # 2. Explicit offline mode must succeed without Hub access
    cfg_offline = {
        "clip_offline": True,
        "clip_hidden_size": 128,
        "clip_layers": 2,
        "clip_heads": 4,
    }
    model, processor, dim = build_clip_vision(cfg_offline)
    assert dim == 128
    assert model is not None


def test_multimodal_image_tensor_routing_and_masking():
    """Verify image tensor routing and zero visual perturbation when image_mask == 0.0."""
    d = 64
    fusion = GatedVisionFusion(d)
    text_vec = torch.randn(4, d)
    vis_vec = torch.randn(4, d)

    # All zeros image mask
    zeros_mask = torch.zeros(4, 1)
    fused_zeros = fusion(text_vec, vis_vec, zeros_mask)
    assert torch.allclose(fused_zeros, text_vec, atol=1e-7), "Zero image mask must leave text vector completely unchanged"

    # Mixed mask
    mixed_mask = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    fused_mixed = fusion(text_vec, vis_vec, mixed_mask)
    assert torch.allclose(fused_mixed[1], text_vec[1], atol=1e-7)
    assert torch.allclose(fused_mixed[3], text_vec[3], atol=1e-7)
    assert not torch.allclose(fused_mixed[0], text_vec[0])


def test_loss_masking_for_conditional_labels():
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

    # Row 1 is non-hate: target and severity are masked (-100)
    batch = {
        "labels": {
            "sarcasm": torch.tensor([1, 0]),
            "hate": torch.tensor([1, 0]),
            "target": torch.tensor([0, IGNORE_INDEX]),
            "severity": torch.tensor([1, IGNORE_INDEX]),
        },
        "hate_attn_target": None,
        "hate_attn_valid": None,
    }

    loss, _ = loss_fn(out, batch)
    loss.backward()

    # Masked row must have strictly zero gradient
    assert (logits["target"].grad[1] == 0).all(), "Masked target sample must receive 0 gradient"
    assert (logits["severity"].grad[1] == 0).all(), "Masked severity sample must receive 0 gradient"
    # Unmasked row must have non-zero gradient
    assert not (logits["target"].grad[0] == 0).all()
    assert not (logits["severity"].grad[0] == 0).all()


def test_bootstrap_confidence_intervals():
    """Verify bootstrap 95% CI computation for uncertainty estimation."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0] * 5)
    y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1, 1] * 5)

    res = bootstrap_ci(y_true, y_pred, n_boot=200, seed=42)
    assert "estimate" in res
    assert "ci95" in res
    assert 0.0 <= res["ci95"][0] <= res["estimate"] <= res["ci95"][1] <= 1.0
