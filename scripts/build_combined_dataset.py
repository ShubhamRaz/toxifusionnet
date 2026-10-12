"""Create the combined MultiBully + HOT dataset for multi-dataset experiments."""
from pathlib import Path
import json
import pandas as pd

workspace = Path(r"c:\Users\SHUBHAM RAJ\Desktop\Others\Toxic\toxifusionnet\toxifusionnet")
mb_path = workspace / "data" / "processed" / "multibully.csv"
hot_path = workspace / "data" / "processed" / "hot_cleaned.csv"
combined_path = workspace / "data" / "processed" / "combined_multibully_hot.csv"

df_mb = pd.read_csv(mb_path)
df_mb["source_dataset"] = "multibully"

df_hot = pd.read_csv(hot_path)
df_hot["source_dataset"] = "hot"
if "sarcasm_rationale" not in df_hot:
    df_hot["sarcasm_rationale"] = None
if "hate_rationale" not in df_hot:
    df_hot["hate_rationale"] = None
if "narrative" not in df_hot:
    df_hot["narrative"] = None

# Canonical common columns
canonical_cols = [
    "id", "text", "sarcasm", "hate", "target", "severity",
    "sarcasm_rationale", "hate_rationale", "narrative",
    "label_source", "split", "source_dataset"
]

# Keep metadata columns from HOT for full provenance
hot_extra_cols = ["orig_hot_label", "qwen_hate", "qwen_sarcasm", "qwen_target", "qwen_severity", "annotation_status"]
for c in hot_extra_cols:
    if c not in df_mb:
        df_mb[c] = None

all_cols = canonical_cols + [c for c in hot_extra_cols if c not in canonical_cols]

df_combined = pd.concat([df_mb[all_cols], df_hot[all_cols]], ignore_index=True)
df_combined.to_csv(combined_path, index=False, encoding="utf-8")

meta = {
    "total_records": len(df_combined),
    "sources": df_combined["source_dataset"].value_counts().to_dict(),
    "splits": {
        "overall": df_combined["split"].value_counts().to_dict(),
        "multibully": df_mb["split"].value_counts().to_dict(),
        "hot": df_hot["split"].value_counts().to_dict()
    },
    "tasks": {
        t: df_combined[t].value_counts(dropna=False).to_dict() for t in ["hate", "sarcasm", "target", "severity"]
    }
}
combined_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
print(f"Saved combined dataset to {combined_path} ({len(df_combined)} rows)")
print(f"Sources: {meta['sources']}")
print(f"MultiBully splits: {meta['splits']['multibully']}")
print(f"HOT splits: {meta['splits']['hot']}")
