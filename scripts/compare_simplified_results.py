#!/usr/bin/env python3
"""Compare simplified model results against baseline multimodal HingRoBERTa + CLIP."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def load_metrics(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def main():
    base_path = Path("results/v3_journal_final/final_hingroberta_clip_ordinal_seed42/metrics.json")
    simp_path = Path("runs/exp8_simplified_multimodal/metrics.json")

    if not base_path.exists():
        print(f"Base metrics not found: {base_path}")
        return
    if not simp_path.exists():
        print(f"Simplified metrics not found: {simp_path} (Training may still be in progress)")
        return

    base = load_metrics(base_path)
    simp = load_metrics(simp_path)

    base_tf = base.get("test_flat", {})
    simp_tf = simp.get("test_flat", {})

    print("=" * 80)
    print("TOXIFUSIONNET: SIMPLIFIED VS BASELINE MULTIMODAL COMPARISON")
    print("=" * 80)
    print(f"{'Task / Metric':<30} | {'Baseline (Multiclass)':<22} | {'Simplified (Binary)':<20} | {'Delta':<10}")
    print("-" * 88)

    # Hate
    b_h_f1 = base_tf.get("hate_macro_f1", 0.0)
    s_h_f1 = simp_tf.get("hate_macro_f1", 0.0)
    print(f"{'Hate Macro-F1':<30} | {b_h_f1:<22.4f} | {s_h_f1:<20.4f} | {s_h_f1 - b_h_f1:+10.4f}")

    b_h_auc = base_tf.get("hate_auroc", 0.0)
    s_h_auc = simp_tf.get("hate_auroc", 0.0)
    print(f"{'Hate AUROC':<30} | {b_h_auc:<22.4f} | {s_h_auc:<20.4f} | {s_h_auc - b_h_auc:+10.4f}")

    # Sarcasm
    b_s_f1 = base_tf.get("sarcasm_macro_f1", 0.0)
    s_s_f1 = simp_tf.get("sarcasm_macro_f1", 0.0)
    print(f"{'Sarcasm Macro-F1':<30} | {b_s_f1:<22.4f} | {s_s_f1:<20.4f} | {s_s_f1 - b_s_f1:+10.4f}")

    b_s_auc = base_tf.get("sarcasm_auroc", 0.0)
    s_s_auc = simp_tf.get("sarcasm_auroc", 0.0)
    print(f"{'Sarcasm AUROC':<30} | {b_s_auc:<22.4f} | {s_s_auc:<20.4f} | {s_s_auc - b_s_auc:+10.4f}")

    # Target
    b_t_f1 = base_tf.get("target_macro_f1", 0.0)
    s_t_f1 = simp_tf.get("target_macro_f1", 0.0)
    print(f"{'Target Macro-F1*':<30} | {b_t_f1:<22.4f} | {s_t_f1:<20.4f} | {s_t_f1 - b_t_f1:+10.4f}")

    # Severity
    b_v_f1 = base_tf.get("severity_macro_f1", 0.0)
    s_v_f1 = simp_tf.get("severity_macro_f1", 0.0)
    print(f"{'Severity Macro-F1*':<30} | {b_v_f1:<22.4f} | {s_v_f1:<20.4f} | {s_v_f1 - b_v_f1:+10.4f}")

    # Mean Macro-F1
    b_m_f1 = base_tf.get("mean_macro_f1", 0.0)
    s_m_f1 = simp_tf.get("mean_macro_f1", 0.0)
    print(f"{'Mean Macro-F1*':<30} | {b_m_f1:<22.4f} | {s_m_f1:<20.4f} | {s_m_f1 - b_m_f1:+10.4f}")

    print("-" * 88)
    print("*NOTE: Target and Severity binary scores are NOT directly comparable to multiclass scores.")
    print(" Target was collapsed from 4 subcategories to binary (target_present / target_absent).")
    print(" Severity was collapsed from 3 ordinal levels to binary (mild / severe).")

    print("\n" + "=" * 80)
    print("PER-CLASS DETAILED BREAKDOWN (SIMPLIFIED MODEL)")
    print("=" * 80)
    simp_test = simp.get("test", {})
    for t in ["hate", "sarcasm", "target", "severity"]:
        if t in simp_test:
            tm = simp_test[t]
            classes = tm.get("classes", [])
            print(f"\n--- Task: {t.upper()} (N={tm.get('n', 0)}, Macro-F1={tm.get('macro_f1', 0.0):.4f}) ---")
            print(f"{'Class':<20} | {'F1':<10} | {'Precision':<10} | {'Recall':<10} | {'Support':<8}")
            print("-" * 65)
            for c_idx, c_name in enumerate(classes):
                c_str = str(c_idx)
                if c_str in tm.get("per_class", {}):
                    cm = tm["per_class"][c_str]
                    print(f"{c_name:<20} | {cm['f1']:<10.4f} | {cm['precision']:<10.4f} | {cm['recall']:<10.4f} | {cm['support']:<8}")
            if "confusion_matrix" in tm:
                print("Confusion Matrix:")
                cm_df = pd.DataFrame(tm["confusion_matrix"], index=[f"Gold_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
                print(cm_df)


if __name__ == "__main__":
    main()
