"""Clean and curate the HOT dataset from dataset.csv without modifying the original file."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def clean_hot_dataset(raw_path: str | Path, out_path: str | Path, seed: int = 42) -> pd.DataFrame:
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    total_lines = 0
    blank_lines = 0
    split_lines = 0

    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            total_lines += 1
            if not any(c.strip() for c in row):
                blank_lines += 1
                continue
            orig_label = row[0].strip()
            # If text had unquoted commas, text was split across middle columns
            text_parts = [c for c in row[1:-6] if c != ""]
            if len(text_parts) > 1:
                split_lines += 1
                text = ", ".join(text_parts).strip()
            else:
                text = row[1].strip() if len(row) > 1 else ""

            hate_l = row[-6].strip()
            sarc_l = row[-5].strip()
            targ_l = row[-4].strip()
            sev_l = row[-3].strip()
            model_name = row[-2].strip()
            status = row[-1].strip()

            records.append({
                "id": f"hot_{len(records)}",
                "text": text,
                "orig_hot_label": orig_label,
                "qwen_hate": hate_l,
                "qwen_sarcasm": sarc_l,
                "qwen_target": targ_l,
                "qwen_severity": sev_l,
                "annotation_model": model_name,
                "annotation_status": status,
                "source_dataset": "hot"
            })

    df = pd.DataFrame(records)

    # Standard canonical columns for multitask training:
    # hate: 'hate' or 'non_hate'
    # sarcasm: 'sarcastic' or 'not_sarcastic', 'uncertain' -> NaN
    # target: 'individual', 'community', 'organization', 'society', others -> NaN
    # severity: 'low', 'moderate', 'high', others -> NaN
    df["hate"] = df["qwen_hate"]
    df["sarcasm"] = df["qwen_sarcasm"].apply(lambda s: s if s in ("sarcastic", "not_sarcastic") else np.nan)
    df["target"] = df["qwen_target"].apply(lambda t: t if t in ("individual", "community", "organization", "society") else np.nan)
    df["severity"] = df["qwen_severity"].apply(lambda v: v if v in ("low", "moderate", "high") else np.nan)

    # Group-aware stratified split to guarantee identical texts never cross train/val/test
    # Groups: text strings
    text_groups = pd.factorize(df["text"].str.strip().str.lower())[0]
    # Stratification target: hate label
    y_strat = df["hate"].astype(str)

    # 10 folds: folds 0..6 = train (70%), 7..8 = val (15% approx), 9 = test (15% approx)
    # Using 7 splits: 5 train (71.4%), 1 val (14.3%), 1 test (14.3%)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = np.zeros(len(df), dtype=int)
    for fold, (_, test_idx) in enumerate(sgkf.split(df, y_strat, text_groups)):
        folds[test_idx] = fold

    # Assign splits: fold 0 -> val, fold 1 -> test, folds 2..6 -> train
    df["split"] = "train"
    df.loc[folds == 0, "split"] = "val"
    df.loc[folds == 1, "split"] = "test"

    df["label_source"] = json.dumps({"hate": "qwen3:8b", "sarcasm": "qwen3:8b", "target": "qwen3:8b", "severity": "qwen3:8b", "orig": "hot"})

    df.to_csv(out_path, index=False, encoding="utf-8")

    meta = {
        "dataset_name": "HOT (Hindi Offensive Tweets)",
        "source_file": str(raw_path),
        "total_raw_lines": total_lines,
        "blank_lines_removed": blank_lines,
        "valid_records_retained": len(df),
        "split_lines_reconstructed": split_lines,
        "splits": df["split"].value_counts().to_dict(),
        "tasks": {
            "hate": df["hate"].value_counts(dropna=False).to_dict(),
            "sarcasm": df["sarcasm"].value_counts(dropna=False).to_dict(),
            "target": df["target"].value_counts(dropna=False).to_dict(),
            "severity": df["severity"].value_counts(dropna=False).to_dict(),
        }
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Cleaned HOT dataset saved to {out_path} ({len(df)} rows)")
    print(f"Metadata saved to {out_path.with_suffix('.meta.json')}")
    return df


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    raw_csv = repo_root.parent / "dataset.csv"
    if not raw_csv.exists():
        raw_csv = repo_root / "data" / "raw" / "hot.csv"
    out_csv = repo_root / "data" / "processed" / "hot_cleaned.csv"
    clean_hot_dataset(raw_csv, out_csv)
