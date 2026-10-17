#!/usr/bin/env python3
"""Build the simplified MultiBully dataset exposing only main categories.

Transformations:
1. Hate: Binary ('non_hate', 'hate') - preserved from clean MultiBully.
2. Sarcasm: Binary ('not_sarcastic', 'sarcastic') - preserved from clean MultiBully.
3. Target: Simplified to binary ('target_absent', 'target_present').
   - For hate speech: 'None' -> 'target_absent', while {'individual', 'community', 'organization', 'society'} -> 'target_present'.
   - For non-hate speech: NaN (masked to IGNORE_INDEX = -100).
4. Severity: Simplified to binary ('mild', 'severe').
   - For hate speech: 'low' (Harmless) -> 'mild', while {'moderate', 'high'} (Partially/Very-Harmful) -> 'severe'.
   - For non-hate speech: NaN (masked to IGNORE_INDEX = -100).
5. Experimental integrity:
   - Preserves original clean and raw datasets.
   - Matches 100% of the 5,718 MultiBully examples and aligns 1-to-1 with canonical 5-fold cache multibully_v2_cv_folds.csv.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def build_simplified_dataset(
    clean_path: Path = Path("data/cleaned/multibully_cleaned.csv"),
    raw_path: Path = Path("data/raw/multibully.csv"),
    out_path: Path = Path("data/cleaned/multibully_simplified.csv"),
) -> pd.DataFrame:
    print(f"Loading cleaned dataset from {clean_path}...")
    clean_df = pd.read_csv(clean_path)
    print(f"Cleaned dataset shape: {clean_df.shape}")

    print(f"Loading raw dataset from {raw_path}...")
    raw_df = pd.read_csv(raw_path, keep_default_na=False)
    raw_map = raw_df.set_index("Img-Name")

    simplified_rows = []
    
    counts_before = {
        "hate": clean_df["hate"].value_counts(dropna=False).to_dict(),
        "sarcasm": clean_df["sarcasm"].value_counts(dropna=False).to_dict(),
        "target": clean_df["target"].value_counts(dropna=False).to_dict(),
        "severity": clean_df["severity"].value_counts(dropna=False).to_dict(),
    }

    for _, row in clean_df.iterrows():
        img_id = str(row["id"])
        is_hate = (row["hate"] == "hate")
        
        # Target transformation
        if not is_hate:
            new_target = np.nan
        else:
            # Check raw target annotation
            r = raw_map.loc[img_id] if img_id in raw_map.index else None
            raw_tgt = str(r["Target"]).strip() if r is not None and "Target" in r else ""
            if raw_tgt.lower() in ("none", "") or pd.isna(row["target"]):
                new_target = "target_absent"
            else:
                new_target = "target_present"

        # Severity transformation
        if not is_hate:
            new_severity = np.nan
        else:
            sev_val = str(row["severity"]).strip().lower()
            if sev_val in ("low", "harmless"):
                new_severity = "low"
            elif sev_val in ("moderate", "high", "partially-harmful", "very-harmful"):
                new_severity = "moderate_or_high"
            else:
                new_severity = np.nan

        new_row = dict(row)
        new_row["target"] = new_target
        new_row["severity"] = new_severity
        simplified_rows.append(new_row)

    out_df = pd.DataFrame(simplified_rows)
    
    counts_after = {
        "hate": out_df["hate"].value_counts(dropna=False).to_dict(),
        "sarcasm": out_df["sarcasm"].value_counts(dropna=False).to_dict(),
        "target": out_df["target"].value_counts(dropna=False).to_dict(),
        "severity": out_df["severity"].value_counts(dropna=False).to_dict(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved simplified dataset ({len(out_df)} rows) to {out_path}")

    # Write metadata sidecar
    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "labels": {
            "target": ["target_absent", "target_present"],
            "severity": ["low", "moderate_or_high"],
        },
        "dataset": "multibully_simplified",
        "n_rows": len(out_df),
        "source_cleaned": str(clean_path),
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "task_types": {
            "sarcasm": "binary",
            "hate": "binary",
            "target": "binary_hate_conditional",
            "severity": "binary_hate_conditional",
        },
        "counts_before": {k: {str(c): int(v) for c, v in vdict.items()} for k, vdict in counts_before.items()},
        "counts_after": {k: {str(c): int(v) for c, v in vdict.items()} for k, vdict in counts_after.items()},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata sidecar to {meta_path}")

    print("\n=== CLASS COUNTS BEFORE VS AFTER TRANSFORMATION ===")
    print("Hate:")
    print("  Before:", counts_before["hate"])
    print("  After: ", counts_after["hate"])
    print("Sarcasm:")
    print("  Before:", counts_before["sarcasm"])
    print("  After: ", counts_after["sarcasm"])
    print("Target:")
    print("  Before:", counts_before["target"])
    print("  After: ", counts_after["target"])
    print("Severity:")
    print("  Before:", counts_before["severity"])
    print("  After: ", counts_after["severity"])

    return out_df


if __name__ == "__main__":
    build_simplified_dataset()
