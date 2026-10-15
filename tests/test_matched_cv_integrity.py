"""Regression and integrity test suite for matched cross-validation splitting and cache safety."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from toxifusionnet.data.grouped_splits import (
    _normalize_label_val,
    _sample_stratum,
    aggregate_rare_group_strata,
    assign_group_ids,
    audit_fold_integrity,
    make_grouped_cv_splits,
    normalize_ocr_text,
)
from toxifusionnet.experiments.cv import validate_fold_cache


def test_nan_and_missing_label_normalization():
    """Verify that NaNs, None, and empty/whitespace values are normalized consistently to 'none'."""
    assert _normalize_label_val(np.nan) == "none"
    assert _normalize_label_val(float("nan")) == "none"
    assert _normalize_label_val(None) == "none"
    assert _normalize_label_val("") == "none"
    assert _normalize_label_val("   ") == "none"
    assert _normalize_label_val("nan") == "none"
    assert _normalize_label_val("NaN") == "none"
    assert _normalize_label_val("sarcastic") == "sarcastic"
    assert _normalize_label_val("high") == "high"

    # Verify _sample_stratum returns clean tuples without literal 'nan'
    row_with_nans = pd.Series({
        "hate": "hate",
        "sarcasm": np.nan,
        "target": "society",
        "severity": np.nan,
    })
    fine, base = _sample_stratum(row_with_nans)
    assert "nan" not in fine.lower()
    assert fine == "TGT_society_none"
    assert base == "TGT_society"


def test_singleton_and_rare_strata_aggregation():
    """Verify deterministic aggregation of singleton and rare strata at group level."""
    # Group strata where TGT_society_none has only 1 group, but siblings have sufficient support
    group_fine = {0: "TGT_society_none"}
    group_base = {0: "TGT_society"}

    for i in range(1, 25):
        group_fine[i] = "TGT_society_not_sarcastic"
        group_base[i] = "TGT_society"
    for i in range(25, 45):
        group_fine[i] = "TGT_society_sarcastic"
        group_base[i] = "TGT_society"
    for i in range(45, 100):
        group_fine[i] = "NONHATE_not_sarcastic"
        group_base[i] = "NONHATE"

    aggregated = aggregate_rare_group_strata(group_fine, group_base, min_groups=10, n_splits=5)

    # The singleton (group 0) must be merged into the most frequent sibling
    assert aggregated[0] == "TGT_society_not_sarcastic"
    counts = pd.Series(aggregated).value_counts()
    for stratum, c in counts.items():
        assert c >= 10, f"Stratum {stratum} has {c} < 10 members"


def test_inner_split_stratification_no_crash():
    """Verify that 5-fold CV with inner validation splitting succeeds on tricky edge-case distributions."""
    records = []
    # Create 60 distinct groups
    for g in range(60):
        # Insert a singleton society group
        if g == 0:
            records.append({"id": f"s_{g}", "text": f"text unique {g}", "hate": "hate", "target": "society", "sarcasm": np.nan, "severity": "medium"})
        elif g < 15:
            records.append({"id": f"s_{g}", "text": f"text unique {g}", "hate": "hate", "target": "society", "sarcasm": "not_sarcastic", "severity": "medium"})
        elif g < 30:
            records.append({"id": f"s_{g}", "text": f"text unique {g}", "hate": "hate", "target": "organization", "sarcasm": "sarcastic", "severity": "high"})
        else:
            records.append({"id": f"s_{g}", "text": f"text unique {g}", "hate": "none", "target": np.nan, "sarcasm": "not_sarcastic", "severity": "none"})

    df = pd.DataFrame(records)
    group_df, fold_dfs = make_grouped_cv_splits(df, n_splits=5, seed=42, val_frac=0.15)
    audit = audit_fold_integrity(fold_dfs, len(df))

    assert audit["passed"] is True
    assert len(fold_dfs) == 5
    for f, d in enumerate(fold_dfs):
        assert set(d["split"].unique()) == {"train", "val", "test"}
        assert len(d[d["split"] == "test"]) > 0
        assert len(d[d["split"] == "val"]) > 0
        assert len(d[d["split"] == "train"]) > 0


def test_cached_fold_reconstruction_preserves_group_id(tmp_path: Path):
    """Verify that cached fold dataframes preserve group_id and pass audit_fold_integrity."""
    records = []
    for g in range(50):
        # 2 rows per group to test grouping, including high severity
        sev = "high" if g < 15 else "low"
        records.append({"id": f"id_{g}_a", "text": f"meme template {g}", "hate": "hate" if g % 2 == 0 else "none", "sarcasm": "sarcastic", "severity": sev, "target": "community"})
        records.append({"id": f"id_{g}_b", "text": f"meme template {g}", "hate": "hate" if g % 2 == 0 else "none", "sarcasm": "sarcastic", "severity": sev, "target": "community"})
    df = pd.DataFrame(records)

    _, original_fold_dfs = make_grouped_cv_splits(df, n_splits=5, seed=42)

    # Save to cache
    cached_df = pd.concat([d[["id", "split", "fold", "group_id"]] for d in original_fold_dfs], ignore_index=True)
    cache_file = tmp_path / "test_folds_cache.csv"
    cached_df.to_csv(cache_file, index=False)

    # Reconstruct following cv.py logic
    loaded_cache = pd.read_csv(cache_file)
    validate_fold_cache(loaded_cache, expected_ids=set(df["id"]), folds=5)

    reconstructed_dfs = []
    merge_cols = ["id", "split"]
    if "group_id" in loaded_cache.columns:
        merge_cols.append("group_id")
    for f in range(5):
        d = df.drop(columns=["split", "group_id"], errors="ignore").merge(
            loaded_cache[loaded_cache["fold"] == f][merge_cols], on="id"
        )
        if "group_id" not in d.columns:
            d = assign_group_ids(d)
        d["fold"] = f
        reconstructed_dfs.append(d)

    audit = audit_fold_integrity(reconstructed_dfs, len(df))
    assert audit["passed"] is True, "Reconstructed folds must pass integrity audit with group_id intact"


def test_fold_cache_rejects_missing_and_unexpected_ids():
    """Verify that validate_fold_cache strictly rejects missing, unexpected, duplicate, or overlapping IDs."""
    ids = [f"id_{i}" for i in range(20)]
    fold_records = []
    for f in range(5):
        for i, sample_id in enumerate(ids):
            split = "test" if i % 5 == f else ("val" if i % 5 == (f + 1) % 5 else "train")
            fold_records.append({"id": sample_id, "split": split, "fold": f})
    valid_cache = pd.DataFrame(fold_records)

    # Valid check must pass
    test_dict = validate_fold_cache(valid_cache, expected_ids=set(ids), folds=5)
    assert len(test_dict) == 5

    # 1. Missing expected ID
    with pytest.raises(ValueError, match="missing.*IDs"):
        validate_fold_cache(valid_cache, expected_ids=set(ids) | {"extra_expected"}, folds=5)

    # 2. Unexpected extraneous ID in cache
    bad_unexpected = valid_cache.copy()
    bad_unexpected = pd.concat([
        bad_unexpected,
        pd.DataFrame([{"id": "alien_id", "split": "test", "fold": 0}])
    ], ignore_index=True)
    with pytest.raises(ValueError, match="unexpected"):
        validate_fold_cache(bad_unexpected, expected_ids=set(ids), folds=5)

    # 3. Duplicate assignment within the same fold
    bad_dup_fold = valid_cache.copy()
    bad_dup_fold = pd.concat([
        bad_dup_fold,
        pd.DataFrame([{"id": "id_0", "split": "train", "fold": 0}])
    ], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate IDs within fold"):
        validate_fold_cache(bad_dup_fold, expected_ids=set(ids), folds=5)

    # 4. Duplicate outer-test assignment across folds
    bad_dup_test = valid_cache.copy()
    # Put id_0 in test for fold 1 as well
    bad_dup_test.loc[(bad_dup_test["fold"] == 1) & (bad_dup_test["id"] == "id_0"), "split"] = "test"
    with pytest.raises(ValueError, match="Duplicate test IDs detected across folds"):
        validate_fold_cache(bad_dup_test, expected_ids=set(ids), folds=5)


def test_normalized_text_group_leakage_prevention():
    """Verify that samples with identical OCR text (regardless of case or whitespace) never cross splits."""
    samples = [
        {"id": "1", "text": "Modi ji is GREAT   ", "hate": "none", "sarcasm": "not_sarcastic"},
        {"id": "2", "text": "modi JI is great", "hate": "none", "sarcasm": "not_sarcastic"},
        {"id": "3", "text": "   MODI JI IS GREAT", "hate": "none", "sarcasm": "not_sarcastic"},
    ]
    # Add other filler groups
    for i in range(4, 55):
        samples.append({"id": str(i), "text": f"unique text sample {i}", "hate": "none", "sarcasm": "not_sarcastic"})

    df = pd.DataFrame(samples)
    _, fold_dfs = make_grouped_cv_splits(df, n_splits=5, seed=42)

    for f, d in enumerate(fold_dfs):
        # Look at the three meme variants (ids 1, 2, 3)
        subset_splits = d[d["id"].isin(["1", "2", "3"])]["split"].unique()
        assert len(subset_splits) == 1, f"Fold {f}: OCR variants were split across multiple partitions: {subset_splits}"
