"""Build canonical v2 combined dataset (MultiBully + curated HOT v2) with rigorous metadata."""
import json
import sys
from pathlib import Path
sys.path.insert(0, "src")

import pandas as pd
import numpy as np

mb_path = Path("data/processed/multibully.csv")
hot_path = Path("data/processed/hot_v2.csv")
out_comb_path = Path("data/processed/combined_multibully_hot_v2.csv")

df_mb = pd.read_csv(mb_path)
df_mb["source_dataset"] = "multibully"

df_hot = pd.read_csv(hot_path)
df_hot["source_dataset"] = "hot"

for col in ["sarcasm_rationale", "hate_rationale", "narrative"]:
    if col not in df_hot:
        df_hot[col] = None

for col in ["orig_hot_label", "qwen_hate", "qwen_sarcasm", "qwen_target", "qwen_severity", "annotation_status"]:
    if col not in df_mb:
        df_mb[col] = None

canonical_cols = [
    "id", "text", "sarcasm", "hate", "target", "severity",
    "sarcasm_rationale", "hate_rationale", "narrative",
    "label_source", "source_dataset",
    "orig_hot_label", "qwen_hate", "qwen_sarcasm", "qwen_target", "qwen_severity", "annotation_status"
]

df_comb = pd.concat([df_mb[canonical_cols], df_hot[canonical_cols]], ignore_index=True)
df_comb.to_csv(out_comb_path, index=False, encoding="utf-8")
print(f"Saved combined v2 dataset to {out_comb_path}: {len(df_comb)} rows")

meta = {
    "total_records": len(df_comb),
    "sources": df_comb["source_dataset"].value_counts().to_dict(),
    "tasks": {
        t: df_comb[t].value_counts(dropna=False).to_dict() for t in ["hate", "sarcasm", "target", "severity"]
    },
    "provenance": {
        "multibully_hate": "Human gold (Text-Label from Maity et al., SIGIR 2022)",
        "multibully_sarcasm": "Human gold (Sarcasm column)",
        "multibully_target": "Human gold (Target column, conditional on hate)",
        "multibully_severity": "Human gold (Harmful-Score, conditional on hate)",
        "hot_hate": "Human gold from HOT: 0.0=non_hate (Normal, N=1121), 1.0=hate (Hate Speech, N=303), 2.0=NaN (Offensive profanity excluded, N=1765)",
        "hot_sarcasm": "Qwen3:8B pseudo-label (N=2457 valid; N=732 uncertain/NaN)",
        "hot_target": "Qwen3:8B pseudo-label (N=1035 valid; N=2154 not_applicable/NaN)",
        "hot_severity": "Qwen3:8B pseudo-label (N=1020 valid; N=2169 not_applicable/NaN)",
    }
}
out_comb_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
print(f"Saved metadata to {out_comb_path.with_suffix('.meta.json')}")
