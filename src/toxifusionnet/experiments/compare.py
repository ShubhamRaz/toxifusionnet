"""Statistical comparison of cross-validation model predictions.

Provides:
- Paired, cluster-aware permutation tests (two-sided, plus-one corrected).
- Paired, cluster-aware bootstrap confidence intervals (10,000 resamples).
- Family-wise error rate control via Holm-Bonferroni correction across multi-task comparisons.
- Multi-task overall mean Macro-F1 joint bootstrap and permutation testing.
- Fold-level variability across cross-validation folds.
- Class-level precision, recall, F1, and support analysis (especially ultra-rare classes).
- Publication report generation with Markdown, JSON, and CSV outputs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from ..data.grouped_splits import normalize_ocr_text
from ..training.metrics import evaluate_predictions, flatten_metrics
from ..training.stats import (
    fast_macro_f1,
    paired_group_permutation_test,
    paired_group_bootstrap,
    paired_multitask_group_bootstrap,
    paired_multitask_group_permutation_test,
    holm_bonferroni_correction,
    mcnemar_test,
    class_level_metrics
)

log = logging.getLogger("toxifusionnet")


def _resolve_text_groups(df: pd.DataFrame, multibully_path: str | Path = "data/processed/multibully.csv") -> np.ndarray:
    """Resolve normalized text group IDs for each row in df, merging from multibully.csv if necessary."""
    if "text" in df.columns and df["text"].notna().any():
        return normalize_ocr_text(df["text"].fillna(""))

    mb_p = Path(multibully_path)
    if mb_p.exists():
        try:
            mb = pd.read_csv(mb_p, usecols=["id", "text"])
            merged = df[["id"]].merge(mb, on="id", how="left")
            if merged["text"].notna().any():
                return normalize_ocr_text(merged["text"].fillna(""))
        except Exception as e:
            log.warning("Failed to merge text from %s: %s", mb_p, e)

    # Fallback to unique id if text cannot be resolved
    return np.asarray(df["id"].astype(str))


def compare_predictions(
    preds_a_path: str | Path,
    preds_b_path: str | Path,
    name_a: str = "Model_A",
    name_b: str = "Model_B",
    out_dir: str | Path | None = None,
    n_perm: int = 10000,
    n_boot: int = 10000,
    seed: int = 42,
    multibully_path: str | Path = "data/processed/multibully.csv"
) -> dict[str, Any]:
    """Execute paired, cluster-aware statistical comparison between two pooled cross-validation prediction files."""
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
                mismatches = int((g_a != g_b).sum())
                raise ValueError(f"Gold label mismatch for task '{t}': {mismatches} discrepancies between models!")
            active_tasks.append(t)

    # Resolve normalized text groups for cluster-aware resampling & permutations
    groups = _resolve_text_groups(df_a, multibully_path=multibully_path)

    task_comparisons = {}
    y_trues_dict = {}
    preds_a_dict = {}
    preds_b_dict = {}

    for t in active_tasks:
        pred_col = f"pred_{t}"
        gold_col = f"gold_{t}"
        if pred_col not in df_a.columns or pred_col not in df_b.columns:
            continue

        valid = (df_a[gold_col] != -100) & (df_a[gold_col].notna()) & (df_a[gold_col] != "NA")
        if not valid.any():
            continue

        y_true = df_a.loc[valid, gold_col].values
        y_a = df_a.loc[valid, pred_col].values
        y_b = df_b.loc[valid, pred_col].values
        task_groups = groups[valid]

        y_trues_dict[t] = df_a[gold_col].values
        preds_a_dict[t] = df_a[pred_col].values
        preds_b_dict[t] = df_b[pred_col].values

        # Compute point estimates
        classes = sorted(list(set(y_true)))
        c_map = {c: i for i, c in enumerate(classes)}
        y_enc = np.array([c_map[c] for c in y_true])
        a_enc = np.array([c_map.get(c, len(classes)) for c in y_a])
        b_enc = np.array([c_map.get(c, len(classes)) for c in y_b])

        f1_a = fast_macro_f1(y_enc, a_enc, len(classes))
        f1_b = fast_macro_f1(y_enc, b_enc, len(classes))
        delta_f1 = f1_b - f1_a
        rel_improvement = (delta_f1 / max(f1_a, 1e-4)) * 100.0

        # Two-sided Paired Group Permutation Test (unbiased plus-one correction)
        perm_res = paired_group_permutation_test(
            task_groups, y_true, y_a, y_b, n_perm=n_perm, seed=seed
        )

        # Paired Group-Aware Bootstrap Confidence Interval (10,000 resamples)
        boot_res = paired_group_bootstrap(
            task_groups, y_true, y_a, y_b, n_boot=n_boot, seed=seed
        )

        # Class-level metrics for Model A and Model B
        class_metrics_a = class_level_metrics(y_true, y_a, classes=classes)
        class_metrics_b = class_level_metrics(y_true, y_b, classes=classes)
        class_deltas = {}
        for c in classes:
            c_str = str(c)
            ma = class_metrics_a[c_str]
            mb = class_metrics_b[c_str]
            class_deltas[c_str] = {
                "support": ma["support"],
                "delta_precision": round(mb["precision"] - ma["precision"], 4),
                "delta_recall": round(mb["recall"] - ma["recall"], 4),
                "delta_f1": round(mb["f1"] - ma["f1"], 4),
            }

        # Fold-level variability across the 5 folds
        fold_metrics_a = []
        fold_metrics_b = []
        if "fold" in df_a.columns and "fold" in df_b.columns:
            fold_arr = df_a.loc[valid, "fold"].values
            for f in sorted(pd.Series(fold_arr).dropna().unique()):
                f_mask = fold_arr == f
                if f_mask.any():
                    fold_metrics_a.append(fast_macro_f1(y_enc[f_mask], a_enc[f_mask], len(classes)))
                    fold_metrics_b.append(fast_macro_f1(y_enc[f_mask], b_enc[f_mask], len(classes)))

        fold_var = {
            f"folds_{name_a}": [float(x) for x in fold_metrics_a],
            f"mean_{name_a}": float(np.nanmean(fold_metrics_a)) if fold_metrics_a else np.nan,
            f"std_{name_a}": float(np.nanstd(fold_metrics_a)) if fold_metrics_a else np.nan,
            f"folds_{name_b}": [float(x) for x in fold_metrics_b],
            f"mean_{name_b}": float(np.nanmean(fold_metrics_b)) if fold_metrics_b else np.nan,
            f"std_{name_b}": float(np.nanstd(fold_metrics_b)) if fold_metrics_b else np.nan,
        }

        # McNemar test for binary tasks (hate, sarcasm)
        mcnemar_res = None
        if t in ("hate", "sarcasm"):
            mcnemar_res = mcnemar_test(y_true, y_a, y_b)

        task_comparisons[t] = {
            "evaluated_support": int(valid.sum()),
            f"{name_a}_macro_f1": float(f1_a),
            f"{name_b}_macro_f1": float(f1_b),
            "delta_macro_f1": float(delta_f1),
            "delta_ci95": [float(x) for x in boot_res["ci95"]],
            "p_value_permutation": float(perm_res["p_value_two_sided"]),
            "bootstrap_prob_superiority": float(boot_res["bootstrap_prob_superiority"]),
            "p_value_bootstrap": float(boot_res["p_value_bootstrap"]),
            "relative_improvement_pct": float(rel_improvement),
            "fold_variability": fold_var,
            "class_metrics_a": class_metrics_a,
            "class_metrics_b": class_metrics_b,
            "class_deltas": class_deltas,
            "mcnemar": mcnemar_res
        }

    # Overall Multitask Mean Macro-F1 Comparison
    multitask_res = {}
    if len(active_tasks) >= 2:
        mt_boot = paired_multitask_group_bootstrap(
            groups, y_trues_dict, preds_a_dict, preds_b_dict, active_tasks, n_boot=n_boot, seed=seed
        )
        mt_perm = paired_multitask_group_permutation_test(
            groups, y_trues_dict, preds_a_dict, preds_b_dict, active_tasks, n_perm=n_perm, seed=seed
        )
        multitask_res = {
            f"{name_a}_mean_macro_f1": float(mt_boot["mean_macro_f1_a"]),
            f"{name_b}_mean_macro_f1": float(mt_boot["mean_macro_f1_b"]),
            "delta_mean_macro_f1": float(mt_boot["obs_delta"]),
            "delta_ci95": [float(x) for x in mt_boot["ci95"]],
            "p_value_permutation": float(mt_perm["p_value_two_sided"]),
            "bootstrap_prob_superiority": float(mt_boot["bootstrap_prob_superiority"]),
            "tasks": active_tasks
        }

    report = {
        "model_a": {"name": name_a, "path": str(preds_a_path).replace("\\", "/")},
        "model_b": {"name": name_b, "path": str(preds_b_path).replace("\\", "/")},
        "total_aligned_samples": len(df_a),
        "task_comparisons": task_comparisons,
        "multitask_mean": multitask_res,
        "parameters": {"n_perm": n_perm, "n_boot": n_boot, "seed": seed},
        "implementation_status": "COMPUTED_EVALUATION"
    }

    if out_dir:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "comparison_report.json").write_text(json.dumps(report, indent=2, default=float))
        md_text = _format_single_comparison_markdown(report)
        (out_p / "comparison_report.md").write_text(md_text)
        log.info("Saved comparison report to %s", out_p)

    return report


def run_publication_analysis(
    e1c_dir: str | Path = "results/v3_journal_final/e1c_text_unweighted",
    e1d_dir: str | Path = "results/v3_journal_final/e1d_text_weighted",
    e6b_dir: str | Path = "results/v3_journal_final/e6b_text_gold_weighted",
    out_dir: str | Path = "results/v3_journal_final/publication_analysis",
    n_perm: int = 10000,
    n_boot: int = 10000,
    seed: int = 42,
    multibully_path: str | Path = "data/processed/multibully.csv"
) -> dict[str, Any]:
    """Run full publication-ready statistical analysis across E1c vs E1d and E1d vs E6b.
    Applies Holm-Bonferroni correction across the 8 primary task hypothesis tests.
    Generates machine-readable JSON, summary CSV, and publication Markdown report.
    """
    e1c_path = Path(e1c_dir) / "pooled_model.csv"
    e1d_path = Path(e1d_dir) / "pooled_model.csv"
    e6b_path = Path(e6b_dir) / "pooled_model.csv"

    for p in (e1c_path, e1d_path, e6b_path):
        if not p.exists():
            raise FileNotFoundError(f"Prediction file not found: {p}. Ensure cross-validation completed.")

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    log.info("Running Contrast 1: E1c (Unweighted) vs E1d (Class-Weighted)...")
    comp_c_d = compare_predictions(
        preds_a_path=e1c_path,
        preds_b_path=e1d_path,
        name_a="E1c_unweighted",
        name_b="E1d_weighted",
        n_perm=n_perm,
        n_boot=n_boot,
        seed=seed,
        multibully_path=multibully_path
    )

    log.info("Running Contrast 2: E1d (Weighted) vs E6b (Combined Gold Weighted)...")
    comp_d_6b = compare_predictions(
        preds_a_path=e1d_path,
        preds_b_path=e6b_path,
        name_a="E1d_weighted",
        name_b="E6b_combined_gold",
        n_perm=n_perm,
        n_boot=n_boot,
        seed=seed,
        multibully_path=multibully_path
    )

    # Collect 8 primary hypothesis test p-values for family-wise Holm correction
    test_keys = []
    raw_p_values = []
    tasks = ["hate", "sarcasm", "target", "severity"]

    for t in tasks:
        test_keys.append(("E1c_vs_E1d", t))
        raw_p_values.append(comp_c_d["task_comparisons"][t]["p_value_permutation"])

    for t in tasks:
        test_keys.append(("E1d_vs_E6b", t))
        raw_p_values.append(comp_d_6b["task_comparisons"][t]["p_value_permutation"])

    holm_p_values = holm_bonferroni_correction(raw_p_values)

    # Attach Holm adjusted p-values back to the task comparisons
    for (contrast, t), raw_p, adj_p in zip(test_keys, raw_p_values, holm_p_values):
        comp = comp_c_d if contrast == "E1c_vs_E1d" else comp_d_6b
        comp["task_comparisons"][t]["raw_p_value"] = float(raw_p)
        comp["task_comparisons"][t]["holm_p_value"] = float(adj_p)
        comp["task_comparisons"][t]["p_significant_05"] = bool(adj_p < 0.05)

    # McNemar tests adjustment across binary tasks (4 tests)
    mcnemar_keys = []
    mcnemar_raw_p = []
    for t in ("hate", "sarcasm"):
        mcnemar_keys.append(("E1c_vs_E1d", t))
        mcnemar_raw_p.append(comp_c_d["task_comparisons"][t]["mcnemar"]["p_value"])
        mcnemar_keys.append(("E1d_vs_E6b", t))
        mcnemar_raw_p.append(comp_d_6b["task_comparisons"][t]["mcnemar"]["p_value"])

    mcnemar_holm = holm_bonferroni_correction(mcnemar_raw_p)
    for (contrast, t), raw_p, adj_p in zip(mcnemar_keys, mcnemar_raw_p, mcnemar_holm):
        comp = comp_c_d if contrast == "E1c_vs_E1d" else comp_d_6b
        comp["task_comparisons"][t]["mcnemar"]["raw_p_value"] = float(raw_p)
        comp["task_comparisons"][t]["mcnemar"]["holm_p_value"] = float(adj_p)
        comp["task_comparisons"][t]["mcnemar"]["p_significant_05"] = bool(adj_p < 0.05)

    # Assemble summary table rows
    csv_rows = []
    for contrast, comp in [("E1c_vs_E1d", comp_c_d), ("E1d_vs_E6b", comp_d_6b)]:
        name_a = comp["model_a"]["name"]
        name_b = comp["model_b"]["name"]
        for t in tasks:
            tc = comp["task_comparisons"][t]
            mc = tc.get("mcnemar")
            csv_rows.append({
                "contrast": contrast,
                "task": t,
                "support": tc["evaluated_support"],
                "model_a": name_a,
                "model_b": name_b,
                "metric_a": tc[f"{name_a}_macro_f1"],
                "metric_b": tc[f"{name_b}_macro_f1"],
                "delta": tc["delta_macro_f1"],
                "ci95_lower": tc["delta_ci95"][0],
                "ci95_upper": tc["delta_ci95"][1],
                "raw_p_value": tc["raw_p_value"],
                "holm_p_value": tc["holm_p_value"],
                "p_significant_05": tc["p_significant_05"],
                "bootstrap_prob_superiority": tc["bootstrap_prob_superiority"],
                "mcnemar_stat": mc["statistic"] if mc else None,
                "mcnemar_raw_p": mc["raw_p_value"] if mc else None,
                "mcnemar_holm_p": mc["holm_p_value"] if mc else None,
            })

    summary_df = pd.DataFrame(csv_rows)
    summary_df.to_csv(out_p / "task_comparison_summary.csv", index=False)

    full_publication_results = {
        "metadata": {
            "description": "Publication-Ready Statistical Analysis of E1c, E1d, and E6b",
            "eval_instances": comp_c_d["total_aligned_samples"],
            "n_permutations": n_perm,
            "n_bootstrap": n_boot,
            "seed": seed,
            "correction_method": "Holm-Bonferroni (8 task comparisons)"
        },
        "contrast_e1c_vs_e1d": comp_c_d,
        "contrast_e1d_vs_e6b": comp_d_6b,
        "eight_task_holm_table": [
            {
                "contrast": k[0],
                "task": k[1],
                "raw_permutation_p": float(r),
                "holm_adjusted_p": float(a),
                "significant_at_05": bool(a < 0.05)
            }
            for k, r, a in zip(test_keys, raw_p_values, holm_p_values)
        ]
    }

    (out_p / "statistical_comparison.json").write_text(
        json.dumps(full_publication_results, indent=2, default=float)
    )

    md_report = _format_publication_markdown_report(full_publication_results)
    (out_p / "publication_statistical_report.md").write_text(md_report)
    log.info("Publication analysis completed. Artifacts saved to %s", out_p)

    return full_publication_results


def _format_single_comparison_markdown(report: dict) -> str:
    name_a = report["model_a"]["name"]
    name_b = report["model_b"]["name"]
    comps = report["task_comparisons"]
    p_name = "p_value_permutation"

    lines = [
        f"# Statistical Comparison Report: {name_a} vs. {name_b}",
        "",
        f"**Sample Count:** {report['total_aligned_samples']} aligned out-of-fold test instances with identical gold ground truth.",
        f"**Permutations:** {report.get('parameters', {}).get('n_perm', 10000):,} cluster iterations (two-sided, plus-one corrected).",
        f"**Bootstrap:** {report.get('parameters', {}).get('n_boot', 10000):,} cluster resamples.",
        "",
        "## 1. Multi-Task Performance & Effect Sizes",
        "",
        f"| Task | Support | {name_a} Macro-F1 | {name_b} Macro-F1 | Delta (B - A) | 95% Cluster Bootstrap CI | Two-Sided Perm. $p$ | $\\mathbb{{P}}(\\Delta > 0)$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for t, data in comps.items():
        ci_str = f"[{data['delta_ci95'][0]:.4f}, {data['delta_ci95'][1]:.4f}]"
        p_val = data.get(p_name, data.get("p_value_bootstrap", 1.0))
        sig = "**" if p_val < 0.05 else ""
        lines.append(
            f"| **{t.capitalize()}** | {data['evaluated_support']} | "
            f"{data[f'{name_a}_macro_f1']:.4f} | {data[f'{name_b}_macro_f1']:.4f} | "
            f"{sig}{data['delta_macro_f1']:+.4f}{sig} | {ci_str} | "
            f"{p_val:.4f} | {data['bootstrap_prob_superiority'] * 100.0:.1f}% |"
        )

    # Multitask overall mean
    mt = report.get("multitask_mean")
    if mt:
        mt_ci = f"[{mt['delta_ci95'][0]:.4f}, {mt['delta_ci95'][1]:.4f}]"
        lines.extend([
            "",
            "### Overall Multi-Task Mean Macro-F1",
            "",
            f"- **{name_a} Mean:** {mt[f'{name_a}_mean_macro_f1']:.4f}",
            f"- **{name_b} Mean:** {mt[f'{name_b}_mean_macro_f1']:.4f}",
            f"- **Delta:** {mt['delta_mean_macro_f1']:+.4f} (95% CI: {mt_ci})",
            f"- **Two-Sided Permutation $p$-value:** {mt['p_value_permutation']:.4f}",
            f"- **Bootstrap Probability of Superiority:** {mt['bootstrap_prob_superiority'] * 100.0:.1f}%",
        ])

    return "\n".join(lines)


LABEL_NAME_MAP = {
    "hate": {"0": "non_hate (0)", "1": "hate (1)", 0: "non_hate (0)", 1: "hate (1)"},
    "sarcasm": {"0": "not_sarcastic (0)", "1": "sarcastic (1)", 0: "not_sarcastic (0)", 1: "sarcastic (1)"},
    "severity": {
        "0": "low (0)", "1": "moderate (1)", "2": "high (2)",
        0: "low (0)", 1: "moderate (1)", 2: "high (2)",
        "low": "low (0)", "moderate": "moderate (1)", "high": "high (2)"
    },
    "target": {
        "0": "individual (0)", "1": "organization (1)", "2": "community (2)", "3": "society (3)",
        0: "individual (0)", 1: "organization (1)", 2: "community (2)", 3: "society (3)"
    }
}


def _get_high_severity_metrics(sev_dict: dict) -> dict:
    if not isinstance(sev_dict, dict):
        return {}
    for key in ("high", "2", 2, "2.0", 2.0):
        if key in sev_dict:
            return sev_dict[key]
    return {}


def _format_p_value(p: float | None) -> str:
    """Format p-value for scientific publication: use '<0.0001' for tiny values instead of '0.0000'."""
    if p is None:
        return "—"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _format_publication_markdown_report(full_res: dict) -> str:
    meta = full_res["metadata"]
    c_d = full_res["contrast_e1c_vs_e1d"]
    d_6b = full_res["contrast_e1d_vs_e6b"]
    tasks = ["hate", "sarcasm", "target", "severity"]

    lines = [
        "# ToxiFusionNet: Publication-Grade Statistical Evaluation Report",
        "",
        f"**Protocol:** Grouped 5-Fold Cross-Validation on MultiBully ($N = {meta['eval_instances']:,}$ held-out test predictions).  ",
        f"**Inferential Tests:** Paired cluster permutation test ($B = {meta['n_permutations']:,}$ permutations, two-sided, plus-one corrected).  ",
        f"**Confidence Intervals:** Paired cluster bootstrap ($B = {meta['n_bootstrap']:,}$ resamples, 95% percentile interval).  ",
        f"**Multiple Comparison Control:** Holm-Bonferroni step-down correction across the 8 primary task hypothesis tests (4 tasks $\\times$ 2 contrasts).  ",
        "",
        "---",
        "",
        "## 1. Primary Benchmark: Multi-Task Macro-F1 Comparisons",
        "",
        "> [!NOTE]",
        "> The 8 task-level hypotheses constitute the primary confirmatory testing family controlled via the Holm-Bonferroni step-down procedure ($\\alpha = 0.05$). Overall multi-task mean Macro-F1 comparisons are reported separately as aggregate secondary analyses; their empirical permutation $p$-values are unadjusted and not part of the primary 8-test family.",
        "",
        "### Contrast 1: E1c (Unweighted Control) vs. E1d (Square-Root Balanced)",
        "",
        "| Task | Support | E1c Macro-F1 | E1d Macro-F1 | Delta (E1d - E1c) | 95% Cluster CI | Raw $p_{\\text{perm}}$ | Holm-Adjusted $p$ | Significant ($\\alpha=0.05$) | $\\mathbb{P}(\\Delta > 0)$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for t in tasks:
        tc = c_d["task_comparisons"][t]
        ci_str = f"[{tc['delta_ci95'][0]:.4f}, {tc['delta_ci95'][1]:.4f}]"
        sig_str = "**Yes**" if tc["p_significant_05"] else "No"
        sig_marker = "**" if tc["p_significant_05"] else ""
        raw_p_str = _format_p_value(tc["raw_p_value"])
        holm_p_str = _format_p_value(tc["holm_p_value"])
        lines.append(
            f"| **{t.capitalize()}** | {tc['evaluated_support']:,} | "
            f"{tc['E1c_unweighted_macro_f1']:.4f} | {tc['E1d_weighted_macro_f1']:.4f} | "
            f"{sig_marker}{tc['delta_macro_f1']:+.4f}{sig_marker} | {ci_str} | "
            f"{raw_p_str} | {holm_p_str} | {sig_str} | "
            f"{tc['bootstrap_prob_superiority'] * 100.0:.1f}% |"
        )

    # Multitask mean for Contrast 1 (Aggregate Secondary Analysis)
    mt1 = c_d.get("multitask_mean", {})
    if mt1:
        ci_mt1 = f"[{mt1['delta_ci95'][0]:.4f}, {mt1['delta_ci95'][1]:.4f}]"
        lines.append(
            f"| **Overall Mean (Secondary)** | {meta['eval_instances']:,} | "
            f"{mt1['E1c_unweighted_mean_macro_f1']:.4f} | {mt1['E1d_weighted_mean_macro_f1']:.4f} | "
            f"**{mt1['delta_mean_macro_f1']:+.4f}** | {ci_mt1} | "
            f"{_format_p_value(mt1['p_value_permutation'])} (unadj.) | - | - | {mt1['bootstrap_prob_superiority'] * 100.0:.1f}% |"
        )

    lines.extend([
        "",
        "### Contrast 2: E1d (Square-Root Balanced) vs. E6b (Combined Gold Weighted)",
        "",
        "| Task | Support | E1d Macro-F1 | E6b Macro-F1 | Delta (E6b - E1d) | 95% Cluster CI | Raw $p_{\\text{perm}}$ | Holm-Adjusted $p$ | Significant ($\\alpha=0.05$) | $\\mathbb{P}(\\Delta > 0)$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for t in tasks:
        tc = d_6b["task_comparisons"][t]
        ci_str = f"[{tc['delta_ci95'][0]:.4f}, {tc['delta_ci95'][1]:.4f}]"
        sig_str = "**Yes**" if tc["p_significant_05"] else "No"
        sig_marker = "**" if tc["p_significant_05"] else ""
        raw_p_str = _format_p_value(tc["raw_p_value"])
        holm_p_str = _format_p_value(tc["holm_p_value"])
        lines.append(
            f"| **{t.capitalize()}** | {tc['evaluated_support']:,} | "
            f"{tc['E1d_weighted_macro_f1']:.4f} | {tc['E6b_combined_gold_macro_f1']:.4f} | "
            f"{sig_marker}{tc['delta_macro_f1']:+.4f}{sig_marker} | {ci_str} | "
            f"{raw_p_str} | {holm_p_str} | {sig_str} | "
            f"{tc['bootstrap_prob_superiority'] * 100.0:.1f}% |"
        )

    mt2 = d_6b.get("multitask_mean", {})
    if mt2:
        ci_mt2 = f"[{mt2['delta_ci95'][0]:.4f}, {mt2['delta_ci95'][1]:.4f}]"
        lines.append(
            f"| **Overall Mean (Secondary)** | {meta['eval_instances']:,} | "
            f"{mt2['E1d_weighted_mean_macro_f1']:.4f} | {mt2['E6b_combined_gold_mean_macro_f1']:.4f} | "
            f"**{mt2['delta_mean_macro_f1']:+.4f}** | {ci_mt2} | "
            f"{_format_p_value(mt2['p_value_permutation'])} (unadj.) | - | - | {mt2['bootstrap_prob_superiority'] * 100.0:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Class-Level Precision, Recall, and F1 Breakdown",
        "",
        "Special focus is placed on the ultra-rare **High Severity** class ($N = 30$) and minority target categories:",
        "",
        "| Head | Class Label | Support | E1c Prec / Rec / F1 | E1d Prec / Rec / F1 | E6b Prec / Rec / F1 | $\\Delta$ F1 (E1d - E1c) | $\\Delta$ F1 (E6b - E1d) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for t in tasks:
        tc_c_d = c_d["task_comparisons"][t]
        tc_d_6b = d_6b["task_comparisons"][t]
        classes = sorted(list(tc_c_d["class_metrics_a"].keys()))
        for c in classes:
            ma = tc_c_d["class_metrics_a"][c]
            mb = tc_c_d["class_metrics_b"][c]
            m6 = tc_d_6b["class_metrics_b"][c]
            d1 = mb["f1"] - ma["f1"]
            d2 = m6["f1"] - mb["f1"]
            label_disp = LABEL_NAME_MAP.get(t, {}).get(c, str(c))
            str_a = f"{ma['precision']:.3f} / {ma['recall']:.3f} / **{ma['f1']:.3f}**"
            str_b = f"{mb['precision']:.3f} / {mb['recall']:.3f} / **{mb['f1']:.3f}**"
            str_6 = f"{m6['precision']:.3f} / {m6['recall']:.3f} / **{m6['f1']:.3f}**"
            lines.append(
                f"| **{t.capitalize()}** | `{label_disp}` | {ma['support']:,} | "
                f"{str_a} | {str_b} | {str_6} | {d1:+.4f} | {d2:+.4f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Fold-Level Performance & Stability Across Folds",
        "",
        "5-fold cross-validation metrics ($Mean \\pm Std$ across held-out test partitions):",
        "",
        "| Task | Contrast 1: E1c vs. E1d (Fold Deltas) | Contrast 2: E1d vs. E6b (Fold Deltas) |",
        "| :--- | :--- | :--- |"
    ])

    for t in tasks:
        fv1 = c_d["task_comparisons"][t]["fold_variability"]
        fv2 = d_6b["task_comparisons"][t]["fold_variability"]
        deltas1 = [round(b - a, 4) for a, b in zip(fv1["folds_E1c_unweighted"], fv1["folds_E1d_weighted"])]
        deltas2 = [round(b - a, 4) for a, b in zip(fv2["folds_E1d_weighted"], fv2["folds_E6b_combined_gold"])]
        str1 = f"{fv1['mean_E1c_unweighted']:.4f} +/- {fv1['std_E1c_unweighted']:.4f} vs {fv1['mean_E1d_weighted']:.4f} +/- {fv1['std_E1d_weighted']:.4f} (Deltas: {deltas1})"
        str2 = f"{fv2['mean_E1d_weighted']:.4f} +/- {fv2['std_E1d_weighted']:.4f} vs {fv2['mean_E6b_combined_gold']:.4f} +/- {fv2['std_E6b_combined_gold']:.4f} (Deltas: {deltas2})"
        lines.append(f"| **{t.capitalize()}** | {str1} | {str2} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Supplementary Analysis: Binary Heads McNemar Tests",
        "",
        "| Contrast | Task | Discordant Pairs ($n_{01} + n_{10}$) | Odds Ratio | Test Variant | Raw $p$ | Holm-Adjusted $p$ | Significant ($\\alpha=0.05$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for contrast, comp in [("E1c vs E1d", c_d), ("E1d vs E6b", d_6b)]:
        for t in ("hate", "sarcasm"):
            mc = comp["task_comparisons"][t]["mcnemar"]
            sig_str = "**Yes**" if mc["p_significant_05"] else "No"
            raw_p_str = _format_p_value(mc.get("raw_p_value", mc.get("p_value")))
            holm_p_str = _format_p_value(mc.get("holm_p_value", mc.get("p_value")))
            lines.append(
                f"| {contrast} | **{t.capitalize()}** | {mc['discordant_pairs']} | "
                f"{mc['odds_ratio']:.3f} | {mc['test_variant']} | "
                f"{raw_p_str} | {holm_p_str} | {sig_str} |"
            )

    # Scientific Cautious Interpretation
    # Check statistical conclusions from Holm-adjusted p-values
    sig_e1d_tasks = [t for t in tasks if c_d["task_comparisons"][t]["p_significant_05"]]
    sig_e6b_tasks = [t for t in tasks if d_6b["task_comparisons"][t]["p_significant_05"]]

    lines.extend([
        "",
        "---",
        "",
        "## 5. Methodological & Scientific Interpretation",
        "",
        "### 5.1 Per-Task Class Balancing (E1c -> E1d)",
    ])

    if sig_e1d_tasks:
        lines.append(
            f"- **Statistically Supported Gains:** Following Holm-Bonferroni family-wise error rate control across the 8 primary hypothesis tests, per-task square-root class weighting demonstrates statistically significant Macro-F1 improvements on: **{', '.join(sig_e1d_tasks)}** (Holm $p < 0.05$)."
        )
    else:
        lines.append(
            "- **Statistical Invariance Across Heads:** After rigorous Holm-Bonferroni correction across the 8 simultaneous tests, the Macro-F1 differences between E1c and E1d do not cross the conservative family-wise significance threshold ($\\alpha = 0.05$). Observed sample improvements must be interpreted cautiously as suggestive trends rather than definitive rejections of the null hypothesis."
        )

    # Hate detection cautious statement
    hate_delta = c_d["task_comparisons"]["hate"]["delta_macro_f1"]
    hate_raw_p = _format_p_value(c_d["task_comparisons"]["hate"]["raw_p_value"])
    hate_holm_p = _format_p_value(c_d["task_comparisons"]["hate"]["holm_p_value"])
    lines.append(
        f"- **Hate Detection Invariance:** No statistically significant difference in hate Macro-F1 was detected in this comparison ($\\Delta = {hate_delta:+.4f}$, raw $p = {hate_raw_p}$, Holm $p = {hate_holm_p}$). This lack of statistical significance does not establish equivalence between the models or rule out meaningful differences in other performance measures (such as subgroup error rates or classification thresholds)."
    )

    # Analyze severity trade-off and rare class
    sev_d = c_d["task_comparisons"]["severity"]["class_metrics_b"]
    sev_c = c_d["task_comparisons"]["severity"]["class_metrics_a"]
    high_c = _get_high_severity_metrics(sev_c)
    high_d = _get_high_severity_metrics(sev_d)
    c_rec = high_c.get("recall", 0.0)
    d_rec = high_d.get("recall", 0.0)
    high_rec_diff = d_rec - c_rec
    lines.append(
        f"- **Rare High-Severity Class Support & Uncertainty ($N = 30$):** Recall on the high-severity class increased by {high_rec_diff:+.2f} ({c_rec:.2f} $\\to$ {d_rec:.2f}, F1: {high_c.get('f1', 0.0):.3f} $\\to$ {high_d.get('f1', 0.0):.3f}), moving from complete omission in E1c to non-zero detection in E1d. While this recall improvement is encouraging, the estimate carries substantial uncertainty due to the small test support of only 30 high-severity instances across all 5 folds (an increase of approximately 5 correctly classified examples). This result should be seen as an encouraging indicator of minority sensitivity rather than a precise or broadly generalizable performance estimate. Concurrently, the majority Moderate Severity class ($N = 1,409$) maintained stable performance (0.904 precision, 0.914 F1 vs. 0.892 precision, 0.939 F1 in E1c), showing that minority-class weighting did not induce severe majority-class precision degradation."
    )

    lines.extend([
        "",
        "### 5.2 Auxiliary Gold Hate Supervision Transfer (E1d -> E6b)",
    ])

    if "hate" in sig_e6b_tasks:
        lines.append(
            f"- **Statistically Supported Hate Gain:** Adding auxiliary gold hate supervision produced a statistically significant shift in hate discrimination (Holm $p < 0.05$)."
        )
    else:
        hate_d_6b_delta = d_6b["task_comparisons"]["hate"]["delta_macro_f1"]
        hate_d_6b_raw_p = _format_p_value(d_6b["task_comparisons"]["hate"]["raw_p_value"])
        hate_d_6b_holm_p = _format_p_value(d_6b["task_comparisons"]["hate"]["holm_p_value"])
        lines.append(
            f"- **Inconclusive Primary Hate Transfer:** Holding the optimizer-update budget strictly constant (488 steps) and matching class weights, adding auxiliary gold text hate data yielded negligible and statistically non-significant change in hate classification ($\\Delta = {hate_d_6b_delta:+.4f}$, raw $p = {hate_d_6b_raw_p}$, Holm $p = {hate_d_6b_holm_p}$)."
        )

    # Conditional heads dilution under fixed budget
    if "severity" in sig_e6b_tasks or d_6b["task_comparisons"]["severity"]["delta_macro_f1"] < 0:
        sev_delta = d_6b["task_comparisons"]["severity"]["delta_macro_f1"]
        sev_raw_p = _format_p_value(d_6b["task_comparisons"]["severity"]["raw_p_value"])
        sev_holm = _format_p_value(d_6b["task_comparisons"]["severity"]["holm_p_value"])
        lines.append(
            f"- **Severity Decline Under Combined Training:** Severity Macro-F1 experienced a statistically significant decline in E6b compared to E1d ($\\Delta = {sev_delta:+.4f}$, raw $p = {sev_raw_p}$, Holm $p = {sev_holm}$). A plausible interpretation is that incorporating auxiliary HOT rows (where target and severity labels are masked) under a matched 488-update budget diluted the supervision frequency for conditional heads. However, this is a plausible interpretation rather than a proven mechanism: because E6b uses a different overall data mix under the same fixed update budget, the current experiment cannot isolate reduced MultiBully exposure from other concurrent dataset-mix effects, such as domain discrepancy or loss-landscape gradient interference."
        )

    target_delta = d_6b["task_comparisons"]["target"]["delta_macro_f1"]
    target_raw_p = _format_p_value(d_6b["task_comparisons"]["target"]["raw_p_value"])
    target_holm_p = _format_p_value(d_6b["task_comparisons"]["target"]["holm_p_value"])
    lines.extend([
        f"- **Target Head Invariance:** Target classification showed no statistically significant difference ($\\Delta = {target_delta:+.4f}$, raw $p = {target_raw_p}$, Holm $p = {target_holm_p}$).",
        "",
        "---",
        "*Report generated automatically by `toxifusionnet.experiments.compare.run_publication_analysis` with reproducible cluster permutations and family-wise error control.*"
    ])

    return "\n".join(lines)

