#!/usr/bin/env python3
"""Run multi-task diagnostic analysis across E1c, E1d, E6b and TF-IDF baseline.

Usage:
    python scripts/run_diagnostic_report.py \
        --e1c-dir results/v3_journal_final/e1c_text_unweighted \
        --e1d-dir results/v3_journal_final/e1d_text_weighted \
        --e6b-dir results/v3_journal_final/e6b_text_gold_weighted \
        --out-dir results/v3_journal_final/diagnostic_analysis
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
from toxifusionnet.analysis.diagnostic import run_full_diagnostic_analysis


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1c-dir", default="results/v3_journal_final/e1c_text_unweighted",
                        help="Path to E1c unweighted baseline output directory")
    parser.add_argument("--e1d-dir", default="results/v3_journal_final/e1d_text_weighted",
                        help="Path to E1d square-root weighted output directory")
    parser.add_argument("--e6b-dir", default="results/v3_journal_final/e6b_text_gold_weighted",
                        help="Path to E6b combined gold weighted output directory")
    parser.add_argument("--out-dir", default="results/v3_journal_final/diagnostic_analysis",
                        help="Directory to save diagnostic reports")

    args = parser.parse_args()

    print("=== ToxiFusionNet: Running Multi-Task Diagnostic Analysis ===")
    report = run_full_diagnostic_analysis(
        e1c_dir=args.e1c_dir,
        e1d_dir=args.e1d_dir,
        e6b_dir=args.e6b_dir,
        out_dir=args.out_dir,
    )

    out_p = Path(args.out_dir)
    print("\nSuccessfully generated diagnostic artifacts in:")
    print(f"  1. JSON Report:     {out_p / 'diagnostic_report.json'}")
    print(f"  2. Summary Table:   {out_p / 'diagnostic_summary.csv'}")
    print(f"  3. Markdown Report: {out_p / 'diagnostic_report.md'}\n")

    summary_df = pd.read_csv(out_p / "diagnostic_summary.csv")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
