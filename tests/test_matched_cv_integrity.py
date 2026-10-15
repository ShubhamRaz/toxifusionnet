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


def test_impossible_synthetic_dataset_fails_clearly():
    """Verify that small synthetic datasets fail clearly with a ValueError when a valid split is impossible."""
    # Only 4 groups for 5-fold CV
    samples = [
        {"id": f"s_{i}", "text": f"text {i}", "hate": "none", "sarcasm": "not_sarcastic"}
        for i in range(4)
    ]
    df = pd.DataFrame(samples)
    with pytest.raises(ValueError, match="Cannot construct valid 5-fold splits"):
        make_grouped_cv_splits(df, n_splits=5, seed=42)


def test_standalone_fold_generator_synthetic(tmp_path: Path):
    """Verify that generate_combined_folds runs standalone on synthetic data without model training."""
    from scripts.generate_combined_folds import generate_combined_folds

    # Create synthetic dataset CSV with 60 groups and 2 sources
    records = []
    for i in range(60):
        src = "multibully" if i < 40 else "hot"
        records.append({
            "id": f"synth_{i}",
            "text": f"synthetic text group {i}",
            "hate": "hate" if i % 2 == 0 else "none",
            "sarcasm": "sarcastic" if i % 3 == 0 else "not_sarcastic",
            "target": "society" if i < 15 else "individual",
            "severity": "high" if i < 15 else "low",
            "source_dataset": src,
        })
    synth_df = pd.DataFrame(records)
    synth_csv = tmp_path / "synthetic_dataset.csv"
    synth_df.to_csv(synth_csv, index=False)

    out_csv = tmp_path / "synthetic_cv_folds.csv"
    cached = generate_combined_folds(synth_csv, out_csv, folds=5, seed=42)

    assert out_csv.exists()
    assert len(cached) == 60 * 5
    assert set(cached["fold"].unique()) == {0, 1, 2, 3, 4}
    assert set(cached["split"].unique()) == {"train", "val", "test"}

    # Test overwrite protection without force
    cached_cached = generate_combined_folds(synth_csv, out_csv, folds=5, seed=42, force=False)
    assert len(cached_cached) == 60 * 5


def test_source_aware_composite_id_handling():
    """Verify composite (source_dataset, id) key support when IDs collide across sources."""
    # Source A has id_0..id_9, Source B has id_0..id_9 (colliding IDs)
    records = []
    for f in range(5):
        for i in range(10):
            # Source A
            split_a = "test" if i % 5 == f else "train"
            records.append({"source_dataset": "source_a", "id": f"id_{i}", "split": split_a, "fold": f})
            # Source B
            split_b = "test" if (i + 1) % 5 == f else "train"
            records.append({"source_dataset": "source_b", "id": f"id_{i}", "split": split_b, "fold": f})

    fold_cache = pd.DataFrame(records)
    # validate_fold_cache with id_col=['source_dataset', 'id']
    test_dict = validate_fold_cache(fold_cache, folds=5, id_col=["source_dataset", "id"])
    assert len(test_dict) == 5
    # Total unique test instances across folds should be 20
    all_tests = []
    for f in range(5):
        all_tests.extend(test_dict[f])
    assert len(all_tests) == 20


def test_end_to_end_composite_key_generator_and_audit(tmp_path: Path):
    """Verify end-to-end composite key handling across fold generation, cache validation, and audit."""
    from scripts.generate_combined_folds import generate_combined_folds

    # 60 total rows: 30 in source_a (ids 0..29), 30 in source_b (colliding ids 0..29)
    records = []
    for src in ["source_a", "source_b"]:
        for i in range(30):
            records.append({
                "id": f"id_{i}",
                "text": f"text {src} item {i}",
                "hate": "hate" if i % 2 == 0 else "none",
                "sarcasm": "sarcastic" if i % 3 == 0 else "not_sarcastic",
                "target": "society" if i < 10 else "individual",
                "severity": "high" if i < 10 else "low",
                "source_dataset": src,
            })
    df = pd.DataFrame(records)
    assert not df["id"].is_unique
    assert not df.duplicated(subset=["source_dataset", "id"]).any()

    synth_csv = tmp_path / "colliding_ids_dataset.csv"
    df.to_csv(synth_csv, index=False)

    out_csv = tmp_path / "colliding_ids_cv_folds.csv"
    cached = generate_combined_folds(synth_csv, out_csv, folds=5, seed=42)

    assert out_csv.exists()
    assert len(cached) == 60 * 5
    # Verify cached validation without force works
    cached_reval = generate_combined_folds(synth_csv, out_csv, folds=5, seed=42, force=False)
    assert len(cached_reval) == 60 * 5


def test_unresolvable_duplicate_ids_rejected(tmp_path: Path):
    """Verify that unresolvable duplicate IDs (within the same source) raise ValueError."""
    from scripts.generate_combined_folds import generate_combined_folds

    # Duplicates within source_a
    records = [
        {"id": "id_0", "text": "text 0", "hate": "none", "sarcasm": "none", "source_dataset": "source_a"},
        {"id": "id_0", "text": "text 1", "hate": "none", "sarcasm": "none", "source_dataset": "source_a"},
    ]
    df = pd.DataFrame(records)
    csv_p = tmp_path / "dup_within_source.csv"
    df.to_csv(csv_p, index=False)

    with pytest.raises(ValueError, match="Global 'id' column must be unique"):
        generate_combined_folds(csv_p, tmp_path / "out.csv", folds=5)


def test_cv_reconstruction_with_colliding_ids(tmp_path: Path, monkeypatch):
    """Exercise actual run_cv() reconstruction with colliding IDs across two source datasets.

    Verifies that:
    1. Every original row appears exactly once per fold.
    2. Split assignments match the cache precisely for each source.
    3. No source's data receives another source's assignment.
    4. Suffix collisions (_x, _y) do not occur.
    5. All metadata columns survive untouched.
    6. Mocked train_or_resume prevents any model training.
    """
    from scripts.generate_combined_folds import generate_combined_folds
    from toxifusionnet.experiments.cv import run_cv

    # Create dataset with colliding IDs between source_a and source_b
    records = []
    for src in ["source_a", "source_b"]:
        for i in range(30):
            records.append({
                "id": f"id_{i}",
                "text": f"unique text for {src} item {i}",
                "hate": "hate" if i % 2 == 0 else "none",
                "sarcasm": "sarcastic" if i % 3 == 0 else "not_sarcastic",
                "target": "society" if i < 10 else "individual",
                "severity": "high" if i < 10 else "low",
                "source_dataset": src,
                "meta_token": f"token_{src}_{i}",
            })
    df = pd.DataFrame(records)
    synth_csv = tmp_path / "cv_colliding_dataset.csv"
    df.to_csv(synth_csv, index=False)

    folds_cache_csv = tmp_path / "cv_colliding_folds_cache.csv"
    cached = generate_combined_folds(synth_csv, folds_cache_csv, folds=5, seed=42)

    captured_fold_dfs = []

    def mock_train_or_resume(cfg, d, fold_dir):
        captured_fold_dfs.append(d.copy())
        f_dir = Path(fold_dir)
        f_dir.mkdir(parents=True, exist_ok=True)
        # Create minimal dummy predictions for test split so run_cv finishes slice metrics
        test_part = d[d["split"] == "test"].copy()
        test_part["gold_hate"] = (test_part["hate"] == "hate").astype(int)
        test_part["pred_hate"] = (test_part["hate"] == "hate").astype(int)
        test_part["gold_sarcasm"] = (test_part["sarcasm"] == "sarcastic").astype(int)
        test_part["pred_sarcasm"] = (test_part["sarcasm"] == "sarcastic").astype(int)
        test_part.to_csv(f_dir / "preds_test.csv", index=False)
        return {
            "best_epoch": 1,
            "best_step": 1,
            "best_val_score": 0.5,
            "total_steps": 1,
            "successful_updates": 1,
            "attempted_batches": 1,
            "skipped_updates": 0,
            "steps_budget": 1,
            "eval_checkpoint": "dummy.pt",
            "eval_checkpoint_type": "best",
            "class_balancing": {},
        }

    monkeypatch.setattr("toxifusionnet.experiments.cv.train_or_resume", mock_train_or_resume)

    cv_out_dir = tmp_path / "cv_run_output"
    cv_metrics = run_cv(
        config="configs/exp2_multidataset_text.yaml",
        overrides=[],
        data_path=str(synth_csv),
        out_dir=cv_out_dir,
        folds=5,
        seed=42,
        folds_cache=str(folds_cache_csv),
    )

    # 1. Verification of fold count
    assert len(captured_fold_dfs) == 5

    # 2. Per-fold integrity and correct reconstruction
    outer_test_appearances = {}
    for f, d in enumerate(captured_fold_dfs):
        # Exactly 60 rows (no row dropped, no Cartesian product)
        assert len(d) == len(df), f"Fold {f}: expected {len(df)} rows, got {len(d)}"

        # Suffix collision check: no _x or _y columns
        bad_cols = [c for c in d.columns if c.endswith("_x") or c.endswith("_y")]
        assert not bad_cols, f"Fold {f}: found suffix collisions: {bad_cols}"

        # Metadata preservation
        assert "meta_token" in d.columns
        for _, row in d.iterrows():
            expected_token = f"token_{row['source_dataset']}_{row['id'].replace('id_', '')}"
            assert row["meta_token"] == expected_token, f"Metadata corruption for {row['source_dataset']} {row['id']}"

        # Distinct source data check: text belongs to correct source
        src_a_rows = d[d["source_dataset"] == "source_a"]
        src_b_rows = d[d["source_dataset"] == "source_b"]
        assert len(src_a_rows) == 30
        assert len(src_b_rows) == 30
        assert src_a_rows["text"].str.startswith("unique text for source_a").all()
        assert src_b_rows["text"].str.startswith("unique text for source_b").all()

        # Split assignment matches cache precisely per (source_dataset, id)
        f_cached = cached[cached["fold"] == f]
        for _, c_row in f_cached.iterrows():
            d_split = d.loc[(d["source_dataset"] == c_row["source_dataset"]) & (d["id"] == c_row["id"]), "split"].item()
            assert d_split == c_row["split"], f"Fold {f}: split mismatch for ({c_row['source_dataset']}, {c_row['id']})"

        # Track outer test appearances
        test_keys = list(d[d["split"] == "test"]["source_dataset"] + "::" + d[d["split"] == "test"]["id"])
        for k in test_keys:
            outer_test_appearances[k] = outer_test_appearances.get(k, 0) + 1

    # 3. Outer test coverage: exactly 1 test appearance per sample across all 5 folds
    assert len(outer_test_appearances) == len(df)
    for k, count in outer_test_appearances.items():
        assert count == 1, f"Sample {k} appeared in outer-test {count} times (expected 1)"

    # 4. Source-specific reporting generated in cv_metrics
    assert "sources" in cv_metrics["model"]
    assert "source_a" in cv_metrics["model"]["sources"]
    assert "source_b" in cv_metrics["model"]["sources"]
