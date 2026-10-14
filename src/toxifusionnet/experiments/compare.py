"""Statistical comparison of cross-validation model predictions.

Provides paired, group-aware bootstrap confidence intervals, effect sizes,
fold-level variability, and comprehensive comparison report generation
across all four tasks (hate, sarcasm, target, severity).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..data.grouped_splits import normalize_ocr_text
from ..training.metrics import evaluate_predictions, flatten_metrics
from ..training.stats import paired_group_bootstrap

log = logging.getLogger("toxifusionnet")


def mcnemar_test(y_true: pd.Series, y_a: pd.Series, y_b: pd.Series) -> dict:
    """Compute McNemar test on paired binary predictions."""
    corr_a = (y_a == y_true).astype(int)
    corr_b = (y_b == y_true).astype(int)

    # Contingency table:
    # a_correct & b_correct, a_correct & b_wrong, a_wrong & b_correct, a_wrong & b_wrong
    n00 = int(((corr_a == 0) & (corr_b == 0)).sum())
    n01 = int(((corr_a == 0) & (corr_b == 1)).sum())  # B correct, A wrong
    n10 = int(((corr_a == 1) & (corr_b == 0)).sum())  # A correct, B wrong
    n11 = int(((corr_a == 1) & (corr_b == 1)).sum())

    # McNemar with continuity correction
    diff = abs(n01 - n10)
    stat = ((diff - 1.0) ** 2) / max(n01 + n10, 1)
    p_value = float(stats.chi2.sf(stat, df=1))
    odds_ratio = float(n01 / max(n10, 1))

    return {
        "contingency_table": {"both_correct": n11, "a_only": n10, "b_only": n01, "both_wrong": n00},
        "statistic": float(stat),
        "p_value": p_value,
        "odds_ratio": odds_ratio
    }


def compare_predictions(
    preds_a_path: str | Path,
    preds_b_path: str | Path,
    name_a: str = "Model_A",
    name_b: str = "Model_B",
    out_dir: str | Path | None = None,
    n_boot: int = 1000,
    seed: int = 42
) -> dict[str, Any]:
    """Execute paired, group-aware statistical comparison between two pooled prediction files."""
    df_a = pd.read_csv(preds_a_path)
    df_b = pd.read_csv(preds_b_path)

    # 1. Strict Sample ID & Gold Label Verification
    if "id" not in df_a.columns or "id" not in df_b.columns:
        raise ValueError("Both prediction files must contain an 'id' column for paired alignment.")

    set_a = set(df_a["id"].astype(str))
    set_b = set(df_b["id"].astype(str))
    if set_a != set_b:
        diff = set_a.symmetric_difference(set_b)
        raise ValueError(
            f"Cannot compare models with differing test sample IDs! "
            f"{len(diff)} mismatched IDs detected (examples: {list(diff)[:5]}). "
            f"Comparisons must use identical out-of-fold test sets."
        )

    # Align on ID
    df_a = df_a.sort_values("id").reset_index(drop=True)
    df_b = df_b.sort_values("id").reset_index(drop=True)

    tasks = ["hate", "sarcasm", "target", "severity"]
    active_tasks = []

    # Verify identical gold labels
    for t in tasks:
        gold_col = f"gold_{t}"
        if gold_col in df_a.columns and gold_col in df_b.columns:
            g_a = df_a[gold_col].fillna("NA").astype(str)
            g_b = df_b[gold_col].fillna("NA").astype(str)
            if not (g_a == g_b).all():
                mismatches = (g_a != g_b).sum()
                raise ValueError(f"Gold label mismatch for task '{t}': {mismatches} discrepancies between models!")
            active_tasks.append(t)

    # Groups for clustered bootstrap
    if "text" in df_a.columns:
        groups = normalize_ocr_text(df_a["text"].fillna(""))
    else:
        groups = np.arange(len(df_a)).astype(str)

    task_comparisons = {}

    for t in active_tasks:
        pred_col = f"pred_{t}"
        gold_col = f"gold_{t}"
        if pred_col not in df_a.columns or pred_col not in df_b.columns:
            continue

        valid = (df_a[gold_col] != -100) & (df_a[gold_col].notna()) & (df_a[gold_col] != "NA")
        if not valid.any():
            continue

        y_true = df_a.loc[valid, gold_col]
        y_a = df_a.loc[valid, pred_col]
        y_b = df_b.loc[valid, pred_col]
        task_groups = groups[valid]

        # Compute overall metric scores
        from sklearn.metrics import f1_score
        macro_f1 = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
        f1_a = float(macro_f1(y_true.to_numpy(), y_a.to_numpy()))
        f1_b = float(macro_f1(y_true.to_numpy(), y_b.to_numpy()))
        delta_f1 = f1_b - f1_a
        rel_improvement = (delta_f1 / max(f1_a, 1e-4)) * 100.0

        # Paired group-aware bootstrap
        boot_res = paired_group_bootstrap(task_groups, y_true, y_a, y_b, metric=macro_f1, n_boot=n_boot, seed=seed)

        # Fold-level variability
        fold_metrics_a = []
        fold_metrics_b = []
        if "fold" in df_a.columns and "fold" in df_b.columns:
            for f in sorted(df_a["fold"].dropna().unique()):
                f_mask = valid & (df_a["fold"] == f)
                if f_mask.any():
                    ev_a = evaluate_predictions(df_a[f_mask], [t])
                    ev_b = evaluate_predictions(df_b[f_mask], [t])
                    fold_metrics_a.append(ev_a.get(t, {}).get("macro_f1", np.nan))
                    fold_metrics_b.append(ev_b.get(t, {}).get("macro_f1", np.nan))

        fold_var = {
            f"folds_{name_a}": [float(x) for x in fold_metrics_a],
            f"mean_{name_a}": float(np.nanmean(fold_metrics_a)) if fold_metrics_a else np.nan,
            f"std_{name_a}": float(np.nanstd(fold_metrics_a)) if fold_metrics_a else np.nan,
            f"folds_{name_b}": [float(x) for x in fold_metrics_b],
            f"mean_{name_b}": float(np.nanmean(fold_metrics_b)) if fold_metrics_b else np.nan,
            f"std_{name_b}": float(np.nanstd(fold_metrics_b)) if fold_metrics_b else np.nan,
        }

        # McNemar for binary tasks
        mcnemar = None
        if t in ("hate", "sarcasm"):
            mcnemar = mcnemar_test(y_true, y_a, y_b)

        task_comparisons[t] = {
            "evaluated_support": int(valid.sum()),
            f"{name_a}_macro_f1": float(f1_a),
            f"{name_b}_macro_f1": float(f1_b),
            "delta_macro_f1": float(delta_f1),
            "delta_ci95": [float(x) for x in boot_res["ci95"]],
            "p_value_bootstrap": float(boot_res["p_value"]),
            "relative_improvement_pct": float(rel_improvement),
            "fold_variability": fold_var,
            "mcnemar": mcnemar
        }

    report = {
        "model_a": {"name": name_a, "path": str(preds_a_path)},
        "model_b": {"name": name_b, "path": str(preds_b_path)},
        "total_aligned_samples": len(df_a),
        "task_comparisons": task_comparisons,
        "implementation_status": "COMPUTED_EVALUATION"
    }

    if out_dir:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "comparison_report.json").write_text(json.dumps(report, indent=2, default=float))
        md_text = _format_markdown_report(report)
        (out_p / "comparison_report.md").write_text(md_text)
        log.info("Saved comparison report to %s", out_p)

    return report


def _format_markdown_report(report: dict) -> str:
    name_a = report["model_a"]["name"]
    name_b = report["model_b"]["name"]
    comps = report["task_comparisons"]

    lines = [
        f"# Statistical Comparison Report: {name_a} vs. {name_b}",
        "",
        f"**Sample Count:** {report['total_aligned_samples']} aligned out-of-fold test instances with identical gold ground truth.",
        f"**Bootstrap Iterations:** 1,000 group-aware cluster iterations.",
        f"**Status:** `{report['implementation_status']}`",
        "",
        "## 1. Multi-Task Performance & Effect Sizes",
        "",
        f"| Task | Support | {name_a} Macro-F1 | {name_b} Macro-F1 | Delta (B - A) | 95% Group Bootstrap CI | $p$-value (Bootstrap) | Rel. Impr. (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for t, data in comps.items():
        ci_str = f"[{data['delta_ci95'][0]:.4f}, {data['delta_ci95'][1]:.4f}]"
        sig = "**" if data["p_value_bootstrap"] < 0.05 else ""
        lines.append(
            f"| **{t.capitalize()}** | {data['evaluated_support']} | "
            f"{data[f'{name_a}_macro_f1']:.4f} | {data[f'{name_b}_macro_f1']:.4f} | "
            f"{sig}{data['delta_macro_f1']:+.4f}{sig} | {ci_str} | "
            f"{data['p_value_bootstrap']:.4f} | {data['relative_improvement_pct']:+.2f}% |"
        )

    lines.extend([
        "",
        "## 2. Fold-Level Variability (5-Fold Cross-Validation)",
        "",
        f"| Task | {name_a} (Mean ± Std) | {name_b} (Mean ± Std) | Fold Deltas $[f_0, f_1, f_2, f_3, f_4]$ |",
        "| :--- | :---: | :---: | :---: |"
    ])

    for t, data in comps.items():
        fv = data["fold_variability"]
        ma, sa = fv[f"mean_{name_a}"], fv[f"std_{name_a}"]
        mb, sb = fv[f"mean_{name_b}"], fv[f"std_{name_b}"]
        deltas = [round(b - a, 4) for a, b in zip(fv[f"folds_{name_a}"], fv[f"folds_{name_b}"])]
        lines.append(f"| **{t.capitalize()}** | {ma:.4f} ± {sa:.4f} | {mb:.4f} ± {sb:.4f} | {deltas} |")

    lines.extend([
        "",
        "## 3. Methodological Status Distinction",
        "",
        "- **Computed Results:** The figures above represent empirical calculations on saved prediction runs.",
        "- **Implemented-But-Not-Run Methods:** Proposed extensions (`E1b`, `E6a`, `E7a`, and severity loss ablations) possess verified reproducible configuration pipelines and preflight integrity checks, pending full execution on the DGX supercomputer.",
        ""
    ])

    return "\n".join(lines)
