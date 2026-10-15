"""Standalone canonical fold generator for the combined MultiBully + HOT dataset.

Generates reproducible, group-aware, multi-task stratified 5-fold cross-validation
partitions without triggering model training or modifying model checkpoints.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from toxifusionnet.data.grouped_splits import audit_fold_integrity, make_grouped_cv_splits
from toxifusionnet.experiments.cv import compute_normalized_fold_cache_sha256, validate_fold_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_combined_folds")


def generate_combined_folds(
    data_path: str | Path,
    out_path: str | Path,
    folds: int = 5,
    seed: int = 42,
    val_frac: float = 0.15,
    force: bool = False,
) -> pd.DataFrame:
    data_p = Path(data_path)
    if not data_p.exists():
        raise FileNotFoundError(f"Input dataset not found at: {data_p}")

    df = pd.read_csv(data_p)
    log.info("Loaded dataset from %s: %d rows, %d columns", data_p, len(df), len(df.columns))

    if "id" not in df.columns:
        raise ValueError("Input dataset must contain an 'id' column.")

    use_composite_keys = False
    if not df["id"].is_unique:
        if "source_dataset" in df.columns and not df.duplicated(subset=["source_dataset", "id"]).any():
            use_composite_keys = True
            log.info("Dataset 'id' column contains non-unique IDs across sources; using (source_dataset, id) composite keys.")
        else:
            raise ValueError(
                "Global 'id' column must be unique across all rows, or uniquely identified by (source_dataset, id)."
            )
    else:
        log.info("Dataset 'id' column is globally unique across all %d rows.", len(df))

    if "source_dataset" in df.columns:
        if df.duplicated(subset=["source_dataset", "id"]).any():
            raise ValueError("(source_dataset, id) composite keys must be strictly unique.")
        log.info("Source breakdown: %s", df["source_dataset"].value_counts().to_dict())

    out_p = Path(out_path)
    expected_keys = (
        set(df["source_dataset"].astype(str) + "::" + df["id"].astype(str))
        if use_composite_keys
        else set(df["id"].astype(str))
    )
    id_col = ["source_dataset", "id"] if use_composite_keys else "id"

    if out_p.exists() and not force:
        log.info("Existing fold cache found at %s. Validating existing cache...", out_p)
        cached = pd.read_csv(out_p)
        validate_fold_cache(cached, expected_ids=expected_keys, folds=folds, id_col=id_col)
        sha = compute_normalized_fold_cache_sha256(out_p)
        log.info("Existing fold cache is VALID (SHA-256 [CRLF-normalized]: %s). Use --force to regenerate.", sha)
        return cached

    # Build group-aware splits with rare-stratum aggregation
    log.info("Generating %d group-aware stratified folds (seed=%d, val_frac=%.2f)...", folds, seed, val_frac)
    group_df, fold_dfs = make_grouped_cv_splits(df, n_splits=folds, seed=seed, val_frac=val_frac)

    # Strict integrity audit
    log.info("Auditing fold integrity (leakage, coverage, rare classes)...")
    audit = audit_fold_integrity(fold_dfs, len(df), id_col=id_col)
    if not audit["passed"]:
        raise ValueError(f"Fold integrity audit failed: {audit}")
    log.info(
        "Audit PASSED: total_test=%d, unique_test=%d, 0 leakage across all folds.",
        audit["total_test_evaluated"],
        audit["unique_test_evaluated"],
    )

    # Assemble canonical cache dataframe
    cols = ["id"]
    if "source_dataset" in fold_dfs[0].columns:
        cols.append("source_dataset")
    cols.extend(["fold", "split", "group_id"])

    cached = pd.concat([d[[c for c in cols if c in d.columns]] for d in fold_dfs], ignore_index=True)

    # Validate the generated cache table
    log.info("Validating generated fold cache table against expected IDs...")
    validate_fold_cache(cached, expected_ids=expected_keys, folds=folds, id_col=id_col)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    cached.to_csv(out_p, index=False)
    log.info("Saved canonical fold assignments to %s (%d rows)", out_p, len(cached))

    sha = compute_normalized_fold_cache_sha256(out_p)
    log.info("Canonical fold cache SHA-256 (CRLF-normalized): %s", sha)

    # Print summary statistics per fold
    for f, f_df in enumerate(fold_dfs):
        f_sub = cached[cached["fold"] == f]
        split_counts = f_sub["split"].value_counts().to_dict()
        if "source_dataset" in f_sub.columns:
            src_split = f_sub.groupby(["source_dataset", "split"]).size().to_dict()
            log.info("Fold %d: splits=%s, by_source=%s", f, split_counts, src_split)
        else:
            log.info("Fold %d: splits=%s", f, split_counts)

        # Label distribution report
        for sname in ["train", "val", "test"]:
            s_part = f_df[f_df["split"] == sname]
            h_cnt = int((s_part["hate"] == "hate").sum()) if "hate" in s_part else 0
            sarc_cnt = int((s_part["sarcasm"] == "sarcastic").sum()) if "sarcasm" in s_part else 0
            sev_cnt = int((s_part["severity"] == "high").sum()) if "severity" in s_part else 0
            log.info("  Fold %d [%s, N=%d]: hate=%d, sarcasm=%d, high_sev=%d", f, sname, len(s_part), h_cnt, sarc_cnt, sev_cnt)

    return cached


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/processed/combined_multibully_hot.csv", help="Path to input dataset CSV")
    p.add_argument("--out", default="data/processed/combined_multibully_hot_cv_folds.csv", help="Path to output fold cache CSV")
    p.add_argument("--folds", type=int, default=5, help="Number of outer cross-validation folds")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--val-frac", type=float, default=0.15, help="Fraction of training groups for inner validation")
    p.add_argument("--force", action="store_true", help="Force overwrite existing cache file")
    args = p.parse_args()

    generate_combined_folds(args.data, args.out, folds=args.folds, seed=args.seed, val_frac=args.val_frac, force=args.force)


if __name__ == "__main__":
    main()
