#!/usr/bin/env python3
"""Reproducible generation and verification of derived v3 datasets for ToxiFusionNet.

Generates:
1. data/processed/hot_gold_only.csv:
   Auxiliary text hate dataset strictly containing human gold hate annotations (Normal -> non_hate, Hate -> hate),
   with the offensive class masked to NaN, and all auxiliary tasks (sarcasm, target, severity) masked to NaN.

2. data/processed/multibully_memehate.csv:
   MultiBully dataset evaluating whole-meme hate (Img-Text-Label -> hate), preserving all other human annotations.

Includes extensive validation checks:
- Missing source file detection
- Unexpected schema columns
- Unmapped IDs
- Null labels where not expected
- Exact distribution validation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_v3_datasets")


def prepare_hot_gold_only(
    hot_source_path: str | Path = "data/processed/hot_v2.csv",
    out_path: str | Path = "data/processed/hot_gold_only.csv"
) -> pd.DataFrame:
    """Generate hot_gold_only.csv containing strictly human ground truth for hate."""
    src = Path(hot_source_path)
    if not src.exists():
        # Check fallback
        fallback = Path("data/raw/hot.csv")
        if fallback.exists():
            src = fallback
        else:
            raise FileNotFoundError(
                f"Source HOT dataset not found at {src} or {fallback}. "
                f"Ensure HOT source data is present in data/processed/ or data/raw/."
            )

    log.info("Loading source HOT data from %s...", src)
    df = pd.read_csv(src)

    # Validate required source columns
    if "orig_hot_label" not in df.columns and "label" not in df.columns:
        raise ValueError(f"Expected 'orig_hot_label' or 'label' column in {src}; found {list(df.columns)}")

    # Extract original label
    orig_label = df["orig_hot_label"] if "orig_hot_label" in df.columns else df["label"]

    # Map human gold hate:
    # 0.0 / 'normal' -> non_hate
    # 1.0 / 'hate'   -> hate
    # 2.0 / 'offensive' -> NaN (masked to avoid label noise)
    def map_gold_hate(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in ("0", "0.0", "normal"):
            return "non_hate"
        elif s in ("1", "1.0", "hate"):
            return "hate"
        elif s in ("2", "2.0", "offensive"):
            return np.nan
        return np.nan

    hate_gold = orig_label.map(map_gold_hate)

    # Build clean gold-only DataFrame
    out_df = pd.DataFrame({
        "id": [f"hot_{i}" if not str(x).startswith("hot_") else str(x) for i, x in enumerate(df["id"] if "id" in df.columns else range(len(df)))],
        "text": df["text"].astype(str).str.strip(),
        "orig_hot_label": orig_label,
        "source_dataset": "hot",
        "split": "train",  # Auxiliary data is strictly train-only
        "hate": hate_gold,
        "sarcasm": np.nan,   # 100% masked - zero pseudo-labels
        "target": np.nan,    # 100% masked - zero pseudo-labels
        "severity": np.nan,  # 100% masked - zero pseudo-labels
        "label_source": json.dumps({
            "hate": "hot_human_gold",
            "sarcasm": "masked",
            "target": "masked",
            "severity": "masked"
        })
    })

    # Validation Checks
    log.info("Running validation checks on hot_gold_only...")
    if len(out_df) != 3189:
        log.warning("Expected 3,189 rows in HOT dataset, got %d", len(out_df))

    # Verify ID uniqueness
    if out_df["id"].nunique() != len(out_df):
        raise ValueError(f"Duplicate IDs detected in {out_path}: {len(out_df) - out_df['id'].nunique()} duplicates")

    # Verify no empty texts
    if (out_df["text"].str.len() == 0).any():
        raise ValueError("Empty text entries detected in HOT dataset")

    # Verify auxiliary tasks are strictly 100% NaN
    for aux_task in ("sarcasm", "target", "severity"):
        non_null_count = out_df[aux_task].notna().sum()
        if non_null_count != 0:
            raise ValueError(f"Violation: {aux_task} has {non_null_count} non-null values in hot_gold_only!")

    # Verify exact human hate label counts
    hate_counts = out_df["hate"].value_counts(dropna=False).to_dict()
    log.info("HOT Gold Hate label distribution: %s", hate_counts)

    # In standard HOT: normal=1121, hate=303, offensive/masked=1765
    if hate_counts.get("non_hate") != 1121 or hate_counts.get("hate") != 303:
        raise ValueError(
            f"Unexpected label distribution in HOT gold hate! "
            f"Expected non_hate=1121, hate=303; got {hate_counts}"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    log.info("Successfully generated %s (%d rows)", out, len(out_df))
    return out_df


def prepare_multibully_memehate(
    raw_path: str | Path = "data/raw/multibully.csv",
    proc_path: str | Path = "data/processed/multibully.csv",
    out_path: str | Path = "data/processed/multibully_memehate.csv"
) -> pd.DataFrame:
    """Generate multibully_memehate.csv where hate is mapped from whole-meme Img-Text-Label."""
    raw_p = Path(raw_path)
    proc_p = Path(proc_path)

    if not raw_p.exists():
        raise FileNotFoundError(f"Missing raw MultiBully dataset at {raw_p}")
    if not proc_p.exists():
        raise FileNotFoundError(f"Missing canonical processed MultiBully dataset at {proc_p}")

    log.info("Loading MultiBully raw (%s) and processed (%s)...", raw_p, proc_p)
    raw_df = pd.read_csv(raw_p)
    proc_df = pd.read_csv(proc_p)

    # Verify raw has Img-Name and Img-Text-Label
    if "Img-Name" not in raw_df.columns or "Img-Text-Label" not in raw_df.columns:
        raise ValueError(f"Raw MultiBully missing 'Img-Name' or 'Img-Text-Label'; columns: {list(raw_df.columns)}")

    # Map raw Img-Name to Img-Text-Label
    meme_map = dict(zip(raw_df["Img-Name"].astype(str), raw_df["Img-Text-Label"]))

    # Check for unmapped IDs
    unmapped = [item_id for item_id in proc_df["id"].astype(str) if item_id not in meme_map]
    if unmapped:
        raise ValueError(f"Found {len(unmapped)} IDs in processed MultiBully missing from raw! Examples: {unmapped[:5]}")

    out_df = proc_df.copy()

    # Map Img-Text-Label to binary hate: Bully -> hate, Nonbully -> non_hate
    raw_labels = out_df["id"].astype(str).map(meme_map)
    out_df["hate"] = raw_labels.map({"Bully": "hate", "Nonbully": "non_hate"})

    # Check for null hate labels
    null_hate = out_df["hate"].isna().sum()
    if null_hate > 0:
        bad_ids = out_df[out_df["hate"].isna()]["id"].tolist()
        raise ValueError(f"Detected {null_hate} unmapped/null meme-hate labels! Examples: {bad_ids[:5]}")

    # Update label_source metadata if column exists
    if "label_source" in out_df.columns:
        def update_source(s):
            try:
                d = json.loads(s) if isinstance(s, str) and s.startswith("{") else {}
                d["hate"] = "gold_meme_label"
                return json.dumps(d)
            except Exception:
                return s
        out_df["label_source"] = out_df["label_source"].apply(update_source)

    # Validation Checks
    log.info("Running validation checks on multibully_memehate...")
    if len(out_df) != len(proc_df):
        raise ValueError(f"Row count mismatch: processed has {len(proc_df)}, output has {len(out_df)}")

    meme_hate_counts = out_df["hate"].value_counts().to_dict()
    log.info("Meme-Hate label distribution: %s", meme_hate_counts)

    # Authoritative distribution: hate: 3139 (54.90%), non_hate: 2579 (45.10%)
    if meme_hate_counts.get("hate") != 3139 or meme_hate_counts.get("non_hate") != 2579:
        raise ValueError(
            f"Unexpected label distribution in meme-level hate! "
            f"Expected hate=3139, non_hate=2579; got {meme_hate_counts}"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    log.info("Successfully generated %s (%d rows)", out, len(out_df))
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Prepare reproducible v3 derived datasets.")
    parser.add_argument("--hot-source", default="data/processed/hot_v2.csv", help="Path to source HOT CSV")
    parser.add_argument("--hot-out", default="data/processed/hot_gold_only.csv", help="Path to output HOT gold-only CSV")
    parser.add_argument("--mb-raw", default="data/raw/multibully.csv", help="Path to raw MultiBully CSV")
    parser.add_argument("--mb-proc", default="data/processed/multibully.csv", help="Path to processed MultiBully CSV")
    parser.add_argument("--mb-out", default="data/processed/multibully_memehate.csv", help="Path to output MultiBully MemeHate CSV")
    args = parser.parse_args()

    log.info("=== Preparing v3 Derived Datasets ===")
    prepare_hot_gold_only(args.hot_source, args.hot_out)
    prepare_multibully_memehate(args.mb_raw, args.mb_proc, args.mb_out)
    log.info("=== All v3 Derived Datasets Prepared and Validated Successfully ===")


if __name__ == "__main__":
    main()
