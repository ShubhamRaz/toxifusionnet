"""Comprehensive tests for the simplified ToxiFusionNet schema (main categories only).

Verifies:
1. Label space registration, dynamic mutation, and safe default restoration.
2. Label to ID mapping and missing-label (IGNORE_INDEX) handling.
3. Dataset transformation class counts and strict preservation of non-hate masks.
4. Model head architectures (all 4 heads have exactly 2 output logits).
5. Masked loss gradient behavior (non-hate samples contribute zero gradient to target/severity).
6. Inference pipeline output schema (only main categories exposed, no subtype fields).
7. Cross-validation fold cache 1-to-1 alignment with canonical multibully_v2_cv_folds.csv.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from toxifusionnet.config import Config, load_config
from toxifusionnet.data.dataset import HateSarcasmDataset, make_collate
from toxifusionnet.experiments.cv import (
    CANONICAL_FOLD_CACHE_SHA256,
    CURATED_FOLD_CACHE_SHA256,
    validate_fold_cache,
    validate_fold_cache_checksum,
)
from toxifusionnet.inference.pipeline import ToxiFusionPipeline
from toxifusionnet.labels import (
    BINARY_SEVERITY_LABELS,
    BINARY_TARGET_LABELS,
    DEFAULT_SEVERITY_LABELS,
    DEFAULT_TARGET_LABELS,
    HATE_LABELS,
    IGNORE_INDEX,
    LABEL_SPACES,
    SARCASM_LABELS,
    SEVERITY_LABELS,
    TARGET_LABELS,
    TASKS,
    apply_label_config,
    id_to_label,
    label_to_id,
    num_classes,
    set_severity_labels,
    set_target_labels,
)
from toxifusionnet.models.encoder import build_encoder_and_tokenizer
from toxifusionnet.models.losses import MultiTaskLoss
from toxifusionnet.models.toxifusionnet import ToxiFusionNet
from toxifusionnet.preprocessing.normalize import TextNormalizer
from toxifusionnet.training.balancing import compute_fold_class_weights
from toxifusionnet.training.metrics import evaluate_predictions, flatten_metrics, task_metrics_from_arrays


@pytest.fixture(autouse=True)
def restore_label_spaces():
    """Ensure every test runs with clean label state and restores defaults on teardown."""
    yield
    apply_label_config(None)


def test_label_space_registration_and_restoration():
    """Verify setting binary target and severity updates LABEL_SPACES in place and apply_label_config(None) restores defaults."""
    set_target_labels(BINARY_TARGET_LABELS)
    set_severity_labels(BINARY_SEVERITY_LABELS)
    assert list(TARGET_LABELS) == list(BINARY_TARGET_LABELS)
    assert list(SEVERITY_LABELS) == list(BINARY_SEVERITY_LABELS)
    assert LABEL_SPACES["target"] == list(BINARY_TARGET_LABELS)
    assert LABEL_SPACES["severity"] == list(BINARY_SEVERITY_LABELS)
    assert num_classes("target") == 2
    assert num_classes("severity") == 2

    # Reset to default
    apply_label_config(None)
    assert list(TARGET_LABELS) == list(DEFAULT_TARGET_LABELS)
    assert list(SEVERITY_LABELS) == list(DEFAULT_SEVERITY_LABELS)
    assert num_classes("target") == len(DEFAULT_TARGET_LABELS)
    assert num_classes("severity") == 3


def test_label_to_id_and_id_to_label_binary():
    """Verify canonical labels, synonyms, and missing labels in binary spaces."""
    apply_label_config({
        "target": list(BINARY_TARGET_LABELS),
        "severity": list(BINARY_SEVERITY_LABELS),
    })

    # Target
    assert label_to_id("target", "target_absent") == 0
    assert label_to_id("target", "target_present") == 1
    assert label_to_id("target", "none") == 0
    assert label_to_id("target", "individual") == 1
    assert label_to_id("target", None) == IGNORE_INDEX
    assert label_to_id("target", np.nan) == IGNORE_INDEX
    assert id_to_label("target", 0) == "target_absent"
    assert id_to_label("target", 1) == "target_present"

    # Severity
    assert label_to_id("severity", "low") == 0
    assert label_to_id("severity", "moderate_or_high") == 1
    assert label_to_id("severity", "mild") == 0
    assert label_to_id("severity", "severe") == 1
    assert label_to_id("severity", "moderate") == 1
    assert label_to_id("severity", "high") == 1
    assert label_to_id("severity", None) == IGNORE_INDEX
    assert label_to_id("severity", np.nan) == IGNORE_INDEX
    assert id_to_label("severity", 0) == "low"
    assert id_to_label("severity", 1) == "moderate_or_high"


def test_multibully_simplified_class_counts_and_masks():
    """Verify data/cleaned/multibully_simplified.csv matches exact expected counts and mask guarantees."""
    csv_path = Path("data/cleaned/multibully_simplified.csv")
    assert csv_path.exists(), "multibully_simplified.csv must be generated"

    df = pd.read_csv(csv_path)
    assert len(df) == 5718, f"Expected 5,718 rows, got {len(df)}"

    # Hate
    hate_counts = df["hate"].value_counts().to_dict()
    assert hate_counts == {"non_hate": 4138, "hate": 1580}
    assert df["hate"].isna().sum() == 0

    # Sarcasm
    sarc_counts = df["sarcasm"].value_counts().to_dict()
    assert sarc_counts == {"not_sarcastic": 3589, "sarcastic": 2129}
    assert df["sarcasm"].isna().sum() == 0

    # Target
    target_counts = df["target"].value_counts().to_dict()
    assert target_counts == {"target_present": 1437, "target_absent": 143}
    assert df["target"].isna().sum() == 4138  # Strictly all non-hate rows!

    # Severity
    sev_counts = df["severity"].value_counts().to_dict()
    assert sev_counts == {"moderate_or_high": 1439, "low": 141}
    assert df["severity"].isna().sum() == 4138  # Strictly all non-hate rows!

    # Check non-hate conditioning
    non_hate_mask = df["hate"] == "non_hate"
    assert df.loc[non_hate_mask, "target"].isna().all(), "All non-hate posts must have NaN target"
    assert df.loc[non_hate_mask, "severity"].isna().all(), "All non-hate posts must have NaN severity"

    # Check hate posts have 100% supervision on simplified target and severity
    hate_mask = df["hate"] == "hate"
    assert df.loc[hate_mask, "target"].notna().all(), "All hate posts must have target supervision"
    assert df.loc[hate_mask, "severity"].notna().all(), "All hate posts must have severity supervision"


def test_canonical_fold_cache_matches_simplified_dataset():
    """Verify simplified dataset is completely aligned with canonical multibully_v2_cv_folds.csv without fold regeneration."""
    csv_path = Path("data/cleaned/multibully_simplified.csv")
    folds_path = Path("data/cleaned/multibully_v2_cv_folds.csv")

    df = pd.read_csv(csv_path)
    folds = pd.read_csv(folds_path)

    assert set(df["id"]) == set(folds["id"]), "IDs must match 1-to-1 between simplified dataset and fold cache"

    # Validate fold cache integrity: each fold has a disjoint test partition, together covering all 5718 samples
    fold_dict = validate_fold_cache(folds, expected_ids=set(df["id"]), folds=5)
    assert len(fold_dict) == 5
    total_test_ids = sum(len(fold_dict[f]) for f in range(5))
    assert total_test_ids == 5718
    pooled_test_ids = set().union(*(fold_dict[f] for f in range(5)))
    assert pooled_test_ids == set(df["id"])


def test_model_heads_have_two_output_logits():
    """Verify ToxiFusionNet instantiates heads with exactly 2 logits when target and severity are binary."""
    apply_label_config({
        "target": list(BINARY_TARGET_LABELS),
        "severity": list(BINARY_SEVERITY_LABELS),
    })

    cfg = Config.wrap({
        "encoder_name": "tiny", "tiny_hidden": 32, "tiny_layers": 1, "head_hidden": 32,
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "use_sarcasm_branch": True, "use_fusion": True, "use_cue_features": True, "cue_hidden": 16,
    })
    enc, tok = build_encoder_and_tokenizer(cfg, [["sample", "text"], ["another", "sentence"]])
    model = ToxiFusionNet(enc, cfg)

    # Inspect head output dimensions
    assert model.sarc_out.out_features == 2
    assert model.heads["hate"][-1].out_features == 2
    assert model.heads["target"][-1].out_features == 2
    assert model.heads["severity"][-1].out_features == 2

    # Forward pass
    batch_size = 4
    input_ids = torch.randint(0, 10, (batch_size, 8))
    mask = torch.ones((batch_size, 8), dtype=torch.long)
    cue = torch.zeros((batch_size, 16))

    out = model(input_ids, mask, cue)
    for t in ["sarcasm", "hate", "target", "severity"]:
        assert out["logits"][t].shape == (batch_size, 2), f"Task {t} should output [B, 2] logits"


def test_masked_loss_excludes_non_hate_samples_from_target_and_severity():
    """Verify samples with IGNORE_INDEX contribute strictly zero gradient to target and severity heads."""
    apply_label_config({
        "target": list(BINARY_TARGET_LABELS),
        "severity": list(BINARY_SEVERITY_LABELS),
    })

    cfg = Config.wrap({
        "encoder_name": "tiny", "tiny_hidden": 32, "tiny_layers": 1, "head_hidden": 32,
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "use_sarcasm_branch": True, "use_fusion": True, "use_cue_features": True, "cue_hidden": 16,
    })
    enc, _ = build_encoder_and_tokenizer(cfg, [["sample", "text"]])
    model = ToxiFusionNet(enc, cfg)

    criterion = MultiTaskLoss(
        tasks=["sarcasm", "hate", "target", "severity"],
        weighting="fixed",
        weights={"sarcasm": 1.0, "hate": 1.0, "target": 1.0, "severity": 1.0},
    )

    batch_size = 2
    input_ids = torch.randint(0, 10, (batch_size, 8))
    mask = torch.ones((batch_size, 8), dtype=torch.long)
    cue = torch.zeros((batch_size, 16))

    out = model(input_ids, mask, cue)

    # Batch where both samples are non-hate: target and severity are IGNORE_INDEX (-100)
    batch = {
        "labels": {
            "sarcasm": torch.tensor([0, 1], dtype=torch.long),
            "hate": torch.tensor([0, 0], dtype=torch.long),
            "target": torch.tensor([IGNORE_INDEX, IGNORE_INDEX], dtype=torch.long),
            "severity": torch.tensor([IGNORE_INDEX, IGNORE_INDEX], dtype=torch.long),
        },
        "hate_attn_target": torch.zeros((batch_size, 8)),
        "hate_attn_valid": torch.zeros(batch_size, dtype=torch.bool),
        "sarc_attn_target": torch.zeros((batch_size, 8)),
        "sarc_attn_valid": torch.zeros(batch_size, dtype=torch.bool),
    }

    loss, parts = criterion(out, batch)
    assert "target" not in parts, "Masked target task should not be in loss parts"
    assert "severity" not in parts, "Masked severity task should not be in loss parts"

    loss.backward()

    # Target and severity heads should have strictly zero or None gradients
    for param in model.heads["target"].parameters():
        assert param.grad is None or torch.all(param.grad == 0.0)
    for param in model.heads["severity"].parameters():
        assert param.grad is None or torch.all(param.grad == 0.0)

    # Hate and sarcasm heads must have non-zero gradients
    hate_has_grad = any(p.grad is not None and torch.any(p.grad != 0.0) for p in model.heads["hate"].parameters())
    sarc_has_grad = any(p.grad is not None and torch.any(p.grad != 0.0) for p in model.sarc_out.parameters())
    assert hate_has_grad, "Hate head must receive gradient updates"
    assert sarc_has_grad, "Sarcasm head must receive gradient updates"


def test_inference_pipeline_outputs_only_main_categories():
    """Verify ToxiFusionPipeline produces output with only main categories and no subtype fields."""
    apply_label_config({
        "target": list(BINARY_TARGET_LABELS),
        "severity": list(BINARY_SEVERITY_LABELS),
    })

    cfg = Config.wrap({
        "encoder_name": "tiny", "tiny_hidden": 32, "tiny_layers": 1, "head_hidden": 32,
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "use_sarcasm_branch": True, "use_fusion": True, "use_cue_features": True, "cue_hidden": 16,
    })
    enc, tok = build_encoder_and_tokenizer(cfg, [["hello", "world"], ["hate", "target"]])
    model = ToxiFusionNet(enc, cfg)
    normalizer = TextNormalizer()

    pipe = ToxiFusionPipeline(
        model=model, tokenizer=tok, normalizer=normalizer, max_length=16, device="cpu",
        hate_threshold=0.5, sarcasm_threshold=0.5
    )

    preds = pipe.predict(["hello world clean post", "abusive angry post"], explain=False)
    assert len(preds) == 2

    for res in preds:
        assert "hate" in res
        assert "sarcasm" in res
        assert res["hate"]["label"] in ("non_hate", "hate")
        assert res["sarcasm"]["label"] in ("not_sarcastic", "sarcastic")

        # Subtypes must NOT exist in the top-level output dictionary
        assert "individual" not in res
        assert "community" not in res
        assert "organization" not in res
        assert "society" not in res

        if res["hate"]["label"] == "hate":
            assert res["target"] is not None
            assert res["target"]["label"] in ("target_absent", "target_present")
            assert set(res["target"]["probs"].keys()) == {"target_absent", "target_present"}
            assert res["severity"] is not None
            assert res["severity"]["label"] in ("low", "moderate_or_high")
            assert set(res["severity"]["probs"].keys()) == {"low", "moderate_or_high"}
        else:
            assert res["target"] is None
            assert res["severity"] is None


def test_smoke_training_step_simplified(tmp_path):
    """Verify that train_run completes an end-to-end training cycle on simplified schema."""
    apply_label_config({
        "target": list(BINARY_TARGET_LABELS),
        "severity": list(BINARY_SEVERITY_LABELS),
    })

    cfg = load_config("configs/smoke_simplified.yaml")
    df = pd.read_csv("data/cleaned/multibully_simplified.csv").head(64)
    # Ensure splits exist in subset
    df = df.copy()
    df["split"] = ["train"] * 48 + ["val"] * 8 + ["test"] * 8

    from toxifusionnet.training.trainer import train_run
    res = train_run(cfg, df, tmp_path / "smoke_run")

    assert "test_flat" in res
    flat = res["test_flat"]
    assert "mean_macro_f1" in flat
    assert "hate_macro_f1" in flat
    assert "sarcasm_macro_f1" in flat
    assert "target_macro_f1" in flat
    assert "severity_macro_f1" in flat
    assert Path(tmp_path / "smoke_run" / "model.pt").exists()
    assert Path(tmp_path / "smoke_run" / "metrics.json").exists()
