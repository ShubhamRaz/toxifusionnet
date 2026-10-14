#!/usr/bin/env python3
"""Run publication-ready statistical comparison and report generation.

Usage:
    python scripts/generate_publication_analysis.py \
        --e1c-dir results/v3_journal_final/e1c_text_unweighted \
        --e1d-dir results/v3_journal_final/e1d_text_weighted \
        --e6b-dir results/v3_journal_final/e6b_text_gold_weighted \
        --out-dir results/v3_journal_final/publication_analysis
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from toxifusionnet.experiments.compare import run_publication_analysis


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1c-dir", default="results/v3_journal_final/e1c_text_unweighted",
                        help="Path to E1c unweighted baseline output directory")
    parser.add_argument("--e1d-dir", default="results/v3_journal_final/e1d_text_weighted",
                        help="Path to E1d square-root weighted output directory")
    parser.add_argument("--e6b-dir", default="results/v3_journal_final/e6b_text_gold_weighted",
                        help="Path to E6b combined gold weighted output directory")
    parser.add_argument("--out-dir", default="results/v3_journal_final/publication_analysis",
                        help="Directory to save final publication analysis reports")
    parser.add_argument("--n-perm", type=int, default=10000, help="Number of permutations (default: 10,000)")
    parser.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap iterations (default: 10,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutations and bootstrap")
    parser.add_argument("--multibully-path", default="data/processed/multibully.csv",
                        help="Path to multibully.csv for text cluster resolution")

    args = parser.parse_args()

    print("=== ToxiFusionNet: Generating Publication-Grade Statistical Analysis ===")
    res = run_publication_analysis(
        e1c_dir=args.e1c_dir,
        e1d_dir=args.e1d_dir,
        e6b_dir=args.e6b_dir,
        out_dir=args.out_dir,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        seed=args.seed,
        multibully_path=args.multibully_path
    )

    out_p = Path(args.out_dir)
    print("\nSuccessfully generated publication artifacts in:")
    print(f"  1. JSON Report:     {out_p / 'statistical_comparison.json'}")
    print(f"  2. Summary Table:   {out_p / 'task_comparison_summary.csv'}")
    print(f"  3. Markdown Report: {out_p / 'publication_statistical_report.md'}\n")

    # Print summary table to console
    summary_df = pd.read_csv(out_p / "task_comparison_summary.csv")
    print(summary_df[["contrast", "task", "metric_a", "metric_b", "delta", "raw_p_value", "holm_p_value", "p_significant_05"]].to_string(index=False))


if __name__ == "__main__":
    import pandas as pd
    main()
