#!/usr/bin/env python3
"""Evaluate ToxiFusionNet models to reach target accuracy (e.g. >= 0.85).

Computes:
1. Selective prediction operating points (confident triage) where accuracy >= target_acc.
2. Threshold-tuned operating points.
3. Summary of all tasks achieving >= 85% accuracy.

Usage:
    python scripts/evaluate_at_accuracy.py --target-acc 0.85
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_target_accuracy(pooled_csv: str, target_acc: float = 0.85):
    p = Path(pooled_csv)
    if not p.exists():
        print(f"Error: {pooled_csv} not found.")
        return

    df = pd.read_csv(p)
    print(f"=== Evaluating {p.name} (N = {len(df)}) for Target Accuracy >= {target_acc*100:.1f}% ===\n")

    # 1. Hate Speech - Full vs Selective Classification
    if "p_hate" in df.columns and "gold_hate" in df.columns:
        y_true = df["gold_hate"].values
        p_hate = df["p_hate"].values
        conf = np.abs(p_hate - 0.5)

        base_acc = accuracy_score(y_true, (p_hate >= 0.5).astype(int))
        print(f"--- 1. Hate Detection ---")
        print(f"Base Unfiltered Accuracy (tau=0.50): {base_acc*100:.2f}%\n")

        print("Selective Classification Operating Table (|p - 0.5| >= margin):")
        print(f"{'Margin':<8} | {'P(Hate) Range':<16} | {'Coverage':<10} | {'Accuracy':<10} | {'Status'}")
        print("-" * 62)

        qualifying_margin = None
        for margin in np.arange(0.0, 0.45, 0.05):
            mask = conf >= margin
            preds = (p_hate[mask] >= 0.5).astype(int)
            acc = accuracy_score(y_true[mask], preds)
            cov = mask.mean()
            status = "TARGET MET" if acc >= target_acc else "Below Target"
            if acc >= target_acc and qualifying_margin is None:
                qualifying_margin = margin
            p_low = 0.5 - margin
            p_high = 0.5 + margin
            rng_str = f"<= {p_low:.2f} or >= {p_high:.2f}" if margin > 0 else "All"
            print(f"{margin:<8.2f} | {rng_str:<16} | {cov*100:<9.1f}% | {acc*100:<9.2f}% | {status}")

        if qualifying_margin is not None:
            print(f"\n>> RESULT: Target >= {target_acc*100:.1f}% is achieved at Margin >= {qualifying_margin:.2f}")
            print(f"   Automated Decision Rule: Classify when P(hate) <= {0.5 - qualifying_margin:.2f} or P(hate) >= {0.5 + qualifying_margin:.2f}.")
            print(f"   Uncertain cases ({0.5 - qualifying_margin:.2f} < P(hate) < {0.5 + qualifying_margin:.2f}) routed to human review.")
        print()

    # 2. Severity Classification
    if "pred_severity" in df.columns and "gold_severity" in df.columns:
        valid = (df["gold_severity"] != -100) & pd.notna(df["gold_severity"])
        y_sev = df.loc[valid, "gold_severity"].astype(int)
        pred_sev = df.loc[valid, "pred_severity"].astype(int)
        sev_acc = accuracy_score(y_sev, pred_sev)
        print(f"--- 2. Severity Classification ---")
        print(f"Severity Raw Accuracy: {sev_acc*100:.2f}% | Target Met: {sev_acc >= target_acc}\n")

    # 3. Baseline Comparisons
    baselines_path = Path("results/baselines.csv")
    if baselines_path.exists():
        print("--- 3. Other Benchmark Models with >= 85% Accuracy ---")
        base_df = pd.read_csv(baselines_path, index_col=0)
        for col in ["severity_acc", "target_acc", "hate_acc"]:
            if col in base_df.columns:
                for idx, val in base_df[col].dropna().items():
                    if val >= target_acc:
                        print(f"  - {idx.upper()}: {col} = {val*100:.2f}% (>= {target_acc*100:.0f}%)")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-csv", default="results/v3_journal_final/e1d_text_weighted/pooled_model.csv",
                        help="Path to pooled predictions CSV")
    parser.add_argument("--target-acc", type=float, default=0.85,
                        help="Target accuracy threshold (default: 0.85)")
    args = parser.parse_args()

    # Fallback to cv_pooled_predictions_full if path doesn't exist
    path = args.pooled_csv
    if not Path(path).exists():
        path = "results/cv_pooled_predictions_full.csv"

    evaluate_target_accuracy(path, args.target_acc)


if __name__ == "__main__":
    main()
