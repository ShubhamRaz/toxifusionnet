#!/usr/bin/env python3
"""Run leakage-safe validation-only decision tuning on ToxiFusionNet experiments.

Usage:
    python scripts/run_decision_tuning.py \
        --exp-dir results/v3_journal_final/e1d_text_weighted \
        --exp-name E1d_weighted \
        --out-dir results/v3_journal_final/decision_tuning \
        --min-support 20
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

import pandas as pd
from toxifusionnet.analysis.decision_tuning import run_full_decision_tuning_experiment


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", default="results/v3_journal_final/e1d_text_weighted",
                        help="Path to experiment directory containing fold subdirectories")
    parser.add_argument("--exp-name", default="E1d_weighted",
                        help="Identifier name for the experiment")
    parser.add_argument("--out-dir", default="results/v3_journal_final/decision_tuning",
                        help="Directory to save decision tuning reports and summaries")
    parser.add_argument("--min-support", type=int, default=20,
                        help="Minimum validation instances required per class for tuning (default: 20)")

    args = parser.parse_args()

    print("=== ToxiFusionNet: Running Leakage-Safe Decision Tuning ===")
    results = run_full_decision_tuning_experiment(
        exp_dir=args.exp_dir,
        exp_name=args.exp_name,
        out_dir=args.out_dir,
        min_support=args.min_support,
    )

    out_p = Path(args.out_dir)
    print("\nSuccessfully generated decision tuning artifacts in:")
    print(f"  1. JSON Results:    {out_p / 'decision_tuning_results.json'}")
    print(f"  2. Summary Table:   {out_p / 'decision_tuning_summary.csv'}")
    print(f"  3. Markdown Report: {out_p / 'decision_tuning_report.md'}\n")

    summary_df = pd.read_csv(out_p / "decision_tuning_summary.csv")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
