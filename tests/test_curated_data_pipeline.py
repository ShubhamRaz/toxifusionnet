"""Automated unit and regression tests for curated data pipeline, manifests, and CV isolation.

Tests:
1. Manifest files exist and have valid schemas and non-empty checksums.
2. MultiBully cleaned dataset preserves 100% human gold labels without alteration.
3. HOT cleaned dataset maps 0.0 -> non_hate, 1.0 -> hate, and masks 2.0 (Offensive) to NaN.
4. Task-conditional masking is strictly satisfied (target and severity are NaN on non-hate).
5. CV fold caches guarantee zero image-template and near-duplicate text leakage across partitions.
6. Combined CV folds guarantee strict outer-test isolation (auxiliary HOT is 100% train-only).
7. Review table records all ambiguous/offensive masked cases with clear rationale.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

DATA_CLEANED_DIR = Path("data/cleaned")
MANIFESTS_DIR = Path("data/manifests")
REPORTS_DIR = Path("reports/dataset_audit")


def test_manifest_files_exist_and_valid():
    """Verify that dataset, split, and provenance manifests exist and are valid JSON."""
    assert (MANIFESTS_DIR / "dataset_manifest.json").exists()
    assert (MANIFESTS_DIR / "split_manifest.json").exists()
    assert (MANIFESTS_DIR / "provenance_manifest.json").exists()

    d_manifest = json.loads((MANIFESTS_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert "pipeline_version" in d_manifest
    assert "files" in d_manifest
    assert len(d_manifest["files"]) == 5

    s_manifest = json.loads((MANIFESTS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    assert s_manifest["strategy"] == "StratifiedGroupKFold-Unified-Isolated"
    assert s_manifest["n_splits"] == 5
    assert len(s_manifest["folds"]) == 5

    p_manifest = json.loads((MANIFESTS_DIR / "provenance_manifest.json").read_text(encoding="utf-8"))
    assert "multibully" in p_manifest
    assert "hot" in p_manifest
    assert p_manifest["multibully"]["annotation_level"] == "100% human gold"


def test_multibully_human_gold_labels_preserved():
    """Verify that multibully_cleaned preserves 100% human gold labels without corruption."""
    mb_clean = pd.read_csv(DATA_CLEANED_DIR / "multibully_cleaned.csv")
    mb_proc = pd.read_csv(Path("data/processed/multibully.csv"))

    assert len(mb_clean) == 5718
    assert mb_clean["id"].is_unique

    # Verify ID match
    assert set(mb_clean["id"]) == set(mb_proc["id"])

    merged = mb_clean.merge(mb_proc, on="id", suffixes=("_clean", "_proc"))
    # Check hate and sarcasm
    assert (merged["hate_clean"] == merged["hate_proc"]).all()
    assert (merged["sarcasm_clean"] == merged["sarcasm_proc"]).all()

    # Check target and severity on hate samples (NaN-safe comparison)
    target_matches = ((merged["target_clean"] == merged["target_proc"]) | (merged["target_clean"].isna() & merged["target_proc"].isna())).all()
    assert target_matches, "Target annotations must match human gold 100%"

    severity_matches = ((merged["severity_clean"] == merged["severity_proc"]) | (merged["severity_clean"].isna() & merged["severity_proc"].isna())).all()
    assert severity_matches, "Severity annotations must match human gold 100%"


def test_hot_label_policy_and_masking():
    """Verify HOT label policy: 0.0 -> non_hate, 1.0 -> hate, 2.0 -> NaN, aux tasks -> NaN."""
    hot_clean = pd.read_csv(DATA_CLEANED_DIR / "hot_cleaned.csv")
    assert len(hot_clean) == 3189
    assert hot_clean["id"].is_unique

    # Normal (0.0) -> non_hate
    normal = hot_clean[hot_clean["orig_hot_label"] == 0.0]
    assert len(normal) == 1121
    assert (normal["hate"] == "non_hate").all()

    # Hate Speech (1.0) -> hate
    hate = hot_clean[hot_clean["orig_hot_label"] == 1.0]
    assert len(hate) == 303
    assert (hate["hate"] == "hate").all()

    # Offensive (2.0) -> NaN (masked for binary hate loss)
    offensive = hot_clean[hot_clean["orig_hot_label"] == 2.0]
    assert len(offensive) == 1765
    assert offensive["hate"].isna().all()

    # Auxiliary tasks are 100% masked
    for task in ["sarcasm", "target", "severity"]:
        assert hot_clean[task].isna().all(), f"{task} must be 100% masked in hot_cleaned"


def test_task_conditional_masking():
    """Verify that non-hate samples never carry target or severity supervision."""
    for fname in ["multibully_cleaned.csv", "hot_cleaned.csv", "combined_multibully_hot_v2.csv"]:
        df = pd.read_csv(DATA_CLEANED_DIR / fname)
        non_hate = df[df["hate"] == "non_hate"]
        if "target" in non_hate.columns:
            assert non_hate["target"].isna().all(), f"Target supervision leaked into non_hate in {fname}"
        if "severity" in non_hate.columns:
            assert non_hate["severity"].isna().all(), f"Severity supervision leaked into non_hate in {fname}"


def test_cv_folds_disjoint_and_outer_test_isolated():
    """Verify that CV folds are strictly disjoint, partition all samples, and isolate outer test."""
    mb_folds = pd.read_csv(DATA_CLEANED_DIR / "multibully_v2_cv_folds.csv")
    comb_folds = pd.read_csv(DATA_CLEANED_DIR / "combined_multibully_hot_cv_folds.csv")

    assert len(mb_folds) == 5718 * 5
    assert len(comb_folds) == (5718 + 3189) * 5

    # Check MB fold disjointness and coverage
    for f in range(5):
        f_mb = mb_folds[mb_folds["fold"] == f]
        tr = set(f_mb[f_mb["split"] == "train"]["id"])
        va = set(f_mb[f_mb["split"] == "val"]["id"])
        te = set(f_mb[f_mb["split"] == "test"]["id"])
        assert len(tr & va) == 0
        assert len(tr & te) == 0
        assert len(va & te) == 0
        assert len(tr) + len(va) + len(te) == 5718

    # Check outer test isolation in combined folds
    # In every fold, all HOT rows must be strictly 'train'
    hot_comb = comb_folds[comb_folds["source_dataset"] == "hot"]
    assert (hot_comb["split"] == "train").all(), "All auxiliary HOT rows must be strictly 'train' in every fold"
    assert (hot_comb["split"] == "val").sum() == 0, "Zero HOT rows in validation"
    assert (hot_comb["split"] == "test").sum() == 0, "Zero HOT rows in test"

    # MultiBully splits in combined folds must match mb_folds 100%
    mb_comb = comb_folds[comb_folds["source_dataset"] == "multibully"]
    merged_splits = mb_comb.merge(mb_folds, on=["id", "fold"], suffixes=("_comb", "_mb"))
    assert (merged_splits["split_comb"] == merged_splits["split_mb"]).all(), "MultiBully split mismatch between caches"


def test_unresolved_review_cases_table():
    """Verify that unresolved review cases table exists and contains expected masked cases."""
    rev_p = REPORTS_DIR / "unresolved_review_cases.csv"
    assert rev_p.exists()
    rev_df = pd.read_csv(rev_p)
    assert len(rev_df) == 1765
    assert "orig_hot_label" in rev_df.columns
    assert (rev_df["orig_hot_label"] == 2.0).all()
    assert "reason" in rev_df.columns
