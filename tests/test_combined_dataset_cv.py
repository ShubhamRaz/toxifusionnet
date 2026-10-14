"""Tests for E6/E7 combined MultiBully + HOT dataset cross-validation pipeline."""
import json
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from toxifusionnet.data.schema import load_processed_csv
from toxifusionnet.data.grouped_splits import normalize_ocr_text
from toxifusionnet.models.clip import GatedVisionFusion
from toxifusionnet.data.dataset import HateSarcasmDataset, make_collate
from toxifusionnet.preprocessing.normalize import TextNormalizer
from toxifusionnet.labels import label_to_id, IGNORE_INDEX


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"


def test_multibully_fold_cache_preservation():
    """Verify that canonical MultiBully fold assignments are strictly preserved."""
    fold_cache_path = DATA_DIR / "multibully_v2_cv_folds.csv"
    assert fold_cache_path.exists(), f"multibully_v2_cv_folds.csv must exist at {fold_cache_path}"
    folds = pd.read_csv(fold_cache_path)
    assert len(folds) == 28590, f"Expected 28,590 rows (5,718 * 5), got {len(folds)}"
    assert set(folds["fold"].unique()) == {0, 1, 2, 3, 4}

    test_counts = folds[folds["split"] == "test"].groupby("fold")["id"].count().tolist()
    assert sum(test_counts) == 5718, f"Expected exactly 5,718 test evaluations, got {sum(test_counts)}"
    # Every ID must appear exactly once in test across all 5 folds
    assert folds[folds["split"] == "test"]["id"].nunique() == 5718


def test_auxiliary_hot_data_isolation():
    """Verify that auxiliary HOT samples are injected exclusively into train split."""
    mb = pd.read_csv(DATA_DIR / "multibully.csv")
    if "source_dataset" not in mb:
        mb["source_dataset"] = "multibully"
    hot = pd.read_csv(DATA_DIR / "hot_v2.csv")
    folds_cache = pd.read_csv(DATA_DIR / "multibully_v2_cv_folds.csv")

    for f in range(5):
        mb_f = mb.drop(columns=["split"], errors="ignore").merge(
            folds_cache[folds_cache["fold"] == f][["id", "split"]], on="id"
        )
        hot_f = hot.copy()
        hot_f["split"] = "train"
        hot_f["fold"] = f
        combined = pd.concat([mb_f, hot_f], ignore_index=True)

        # Val must be 100% MultiBully
        val_samples = combined[combined["split"] == "val"]
        assert all(val_samples["source_dataset"] == "multibully")
        assert len(val_samples) == len(mb_f[mb_f["split"] == "val"])

        # Test must be 100% MultiBully
        test_samples = combined[combined["split"] == "test"]
        assert all(test_samples["source_dataset"] == "multibully")
        assert len(test_samples) == len(mb_f[mb_f["split"] == "test"])

        # HOT samples are exclusively in train
        hot_in_comb = combined[combined["source_dataset"] == "hot"]
        assert all(hot_in_comb["split"] == "train")
        assert len(hot_in_comb) == len(hot)


def test_cross_dataset_duplicate_leakage():
    """Verify zero duplicate OCR text overlap between MultiBully and HOT."""
    mb = pd.read_csv(DATA_DIR / "multibully.csv")
    hot = pd.read_csv(DATA_DIR / "hot_v2.csv")

    mb_norm = set(normalize_ocr_text(mb["text"].fillna("")).unique())
    hot_norm = set(normalize_ocr_text(hot["text"].fillna("")).unique())

    overlap = mb_norm.intersection(hot_norm)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts between MultiBully and HOT"


def test_missing_image_modality_masking():
    """Verify that GatedVisionFusion produces zero visual perturbation when image_mask == 0.0."""
    torch.manual_seed(42)
    fusion = GatedVisionFusion(d=64)
    text_vec = torch.randn(8, 64)
    vis_vec = torch.randn(8, 64)  # random/dummy visual features

    # Mask: 4 present, 4 missing
    image_mask = torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])

    out = fusion(text_vec, vis_vec, image_mask=image_mask)

    # For masked samples (indices 4..7), output MUST be identically equal to text_vec
    diff_masked = (out[4:] - text_vec[4:]).abs().max().item()
    assert diff_masked == 0.0, f"Expected 0.0 difference for image_mask=0, got {diff_masked}"

    # For present samples (indices 0..3), output should be fused (different from text_vec)
    diff_present = (out[:4] - text_vec[:4]).abs().max().item()
    assert diff_present > 0.0, "Present images should be fused into representation"


def test_hot_dataset_label_masking():
    """Verify that offensive-excluded samples and missing tasks map to IGNORE_INDEX."""
    hot = pd.read_csv(DATA_DIR / "hot_v2.csv")

    # In hot_v2, orig_hot_label 2.0 (offensive) has hate = NaN
    off_samples = hot[hot["orig_hot_label"] == 2.0]
    for _, row in off_samples.head(20).iterrows():
        assert label_to_id("hate", row["hate"]) == IGNORE_INDEX

    # orig_hot_label 0.0 (normal) has hate = 0 (non_hate)
    norm_samples = hot[hot["orig_hot_label"] == 0.0]
    for _, row in norm_samples.head(20).iterrows():
        assert label_to_id("hate", row["hate"]) == 0

    # orig_hot_label 1.0 (hate) has hate = 1 (hate)
    hate_samples = hot[hot["orig_hot_label"] == 1.0]
    for _, row in hate_samples.head(20).iterrows():
        assert label_to_id("hate", row["hate"]) == 1
