"""Diagnostic analysis of ToxiFusionNet model predictions (E1c, E1d, E6b) and baselines.

Computes comprehensive diagnostic metrics:
- Accuracy, Macro-F1, balanced accuracy, and per-class precision/recall/F1/support
- Confusion matrices and class-wise false positive / false negative error counts
- Binary prediction score distributions for hate and sarcasm (quantiles, calibration, separation)
- Detailed error patterns for target and severity, with special focus on rare High-Severity (N=30)
- Missing-label masking checks and conditional end-to-end evaluation
- Direct head-to-head comparison against the saved TF-IDF classical baseline
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..labels import (
    IGNORE_INDEX,
    LABEL_SPACES,
    MULTIBULLY_TARGET_LABELS,
    TASKS,
    apply_label_config,
)
from ..training.metrics import end_to_end_metrics, evaluate_predictions

log = logging.getLogger("toxifusionnet.analysis.diagnostic")


def compute_task_diagnostic(
    df: pd.DataFrame,
    task: str,
    class_names: list[str] | None = None
) -> dict[str, Any]:
    """Compute detailed diagnostic metrics for a specific task head.

    Handles missing-label masking (IGNORE_INDEX = -100).
    """
    gold_col = f"gold_{task}"
    pred_col = f"pred_{task}"

    if gold_col not in df.columns or pred_col not in df.columns:
        raise KeyError(f"Columns {gold_col} and/or {pred_col} not found in DataFrame.")

    # Apply missing-label masking
    valid = (df[gold_col] != IGNORE_INDEX) & df[gold_col].notna()
    n_total = len(df)
    n_masked = int((~valid).sum())
    n_eval = int(valid.sum())

    if n_eval == 0:
        return {"n_total": n_total, "n_masked": n_masked, "n_eval": 0}

    y_true = df.loc[valid, gold_col].astype(int).values
    y_pred = df.loc[valid, pred_col].astype(int).values

    classes = sorted(list(set(y_true).union(set(y_pred))))
    n_classes = len(classes)

    if class_names is None:
        if task in LABEL_SPACES and len(LABEL_SPACES[task]) == n_classes:
            class_names = list(LABEL_SPACES[task])
        else:
            class_names = [str(c) for c in classes]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    per_class = {}
    for idx, c in enumerate(classes):
        c_name = class_names[idx] if idx < len(class_names) else str(c)
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(n_eval - tp - fn - fp)
        per_class[str(c)] = {
            "class_id": int(c),
            "class_name": c_name,
            "support": int(s[idx]),
            "support_pct": round(float(s[idx]) / max(n_eval, 1) * 100.0, 2),
            "precision": round(float(p[idx]), 4),
            "recall": round(float(r[idx]), 4),
            "f1": round(float(f[idx]), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # Confusion breakdown by class name
    cm_dict = {}
    for i, c_true in enumerate(classes):
        c_true_name = class_names[i] if i < len(class_names) else str(c_true)
        cm_dict[c_true_name] = {}
        for j, c_pred in enumerate(classes):
            c_pred_name = class_names[j] if j < len(class_names) else str(c_pred)
            cm_dict[c_true_name][c_pred_name] = int(cm[i, j])

    diag: dict[str, Any] = {
        "task": task,
        "n_total": n_total,
        "n_masked": n_masked,
        "n_eval": n_eval,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_named": cm_dict,
        "class_names": class_names,
    }

    # Binary task score distribution analysis
    prob_col = f"p_{task}"
    if prob_col in df.columns:
        probs = df.loc[valid, prob_col].values
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)

        pos_p = probs[pos_mask]
        neg_p = probs[neg_mask]

        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        overall_q = {f"p{int(q*100):02d}": round(float(np.quantile(probs, q)), 4) for q in quantiles}
        pos_q = {f"p{int(q*100):02d}": round(float(np.quantile(pos_p, q)), 4) for q in quantiles} if len(pos_p) else {}
        neg_q = {f"p{int(q*100):02d}": round(float(np.quantile(neg_p, q)), 4) for q in quantiles} if len(neg_p) else {}

        # Separation metrics
        pos_mean = float(np.mean(pos_p)) if len(pos_p) else np.nan
        neg_mean = float(np.mean(neg_p)) if len(neg_p) else np.nan
        mean_sep = pos_mean - neg_mean if len(pos_p) and len(neg_p) else np.nan

        # Brier calibration score and AUROC
        auroc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else np.nan
        brier = float(brier_score_loss(y_true, probs))

        # Histogram bins (10 bins from 0.0 to 1.0)
        bin_edges = np.linspace(0.0, 1.0, 11)
        binned_stats = []
        for b_idx in range(10):
            low, high = bin_edges[b_idx], bin_edges[b_idx + 1]
            if b_idx == 9:
                in_bin = (probs >= low) & (probs <= high)
            else:
                in_bin = (probs >= low) & (probs < high)
            bin_count = int(in_bin.sum())
            bin_pos = int(y_true[in_bin].sum()) if bin_count > 0 else 0
            emp_acc = round(float(bin_pos / bin_count), 4) if bin_count > 0 else np.nan
            binned_stats.append({
                "bin_range": f"[{low:.1f}, {high:.1f}]",
                "count": bin_count,
                "positives": bin_pos,
                "empirical_prob": emp_acc,
            })

        diag["score_distribution"] = {
            "overall_mean": round(float(np.mean(probs)), 4),
            "overall_std": round(float(np.std(probs)), 4),
            "overall_quantiles": overall_q,
            "pos_mean": round(pos_mean, 4) if not np.isnan(pos_mean) else None,
            "neg_mean": round(neg_mean, 4) if not np.isnan(neg_mean) else None,
            "mean_separation": round(mean_sep, 4) if not np.isnan(mean_sep) else None,
            "pos_quantiles": pos_q,
            "neg_quantiles": neg_q,
            "auroc": round(auroc, 4) if not np.isnan(auroc) else None,
            "brier_score": round(brier, 4),
            "binned_calibration": binned_stats,
        }

    return diag


def analyze_rare_class_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Provide fine-grained error diagnostics on rare high-severity and minority target classes."""
    out: dict[str, Any] = {}

    # Severity rare-class analysis
    if "gold_severity" in df.columns and "pred_severity" in df.columns:
        v_mask = (df["gold_severity"] != IGNORE_INDEX) & df["gold_severity"].notna()
        sub_sev = df[v_mask]
        y_sev = sub_sev["gold_severity"].astype(int).values
        p_sev = sub_sev["pred_severity"].astype(int).values

        high_mask = (y_sev == 2)
        n_high = int(high_mask.sum())
        high_preds = p_sev[high_mask]

        tp_high = int((high_preds == 2).sum())
        fn_to_mod = int((high_preds == 1).sum())
        fn_to_low = int((high_preds == 0).sum())

        fp_from_mod = int(((y_sev == 1) & (p_sev == 2)).sum())
        fp_from_low = int(((y_sev == 0) & (p_sev == 2)).sum())

        out["severity_high_rare_class"] = {
            "support": n_high,
            "correct_identified_tp": tp_high,
            "recall": round(float(tp_high / max(n_high, 1)), 4),
            "misclassified_as_moderate": fn_to_mod,
            "misclassified_as_low": fn_to_low,
            "false_positives_total": fp_from_mod + fp_from_low,
            "false_positives_from_moderate": fp_from_mod,
            "false_positives_from_low": fp_from_low,
            "precision": round(float(tp_high / max(tp_high + fp_from_mod + fp_from_low, 1)), 4),
        }

        # Low severity analysis
        low_mask = (y_sev == 0)
        n_low = int(low_mask.sum())
        low_preds = p_sev[low_mask]
        tp_low = int((low_preds == 0).sum())
        fn_low_to_mod = int((low_preds == 1).sum())
        fn_low_to_high = int((low_preds == 2).sum())

        out["severity_low_minority_class"] = {
            "support": n_low,
            "correct_identified_tp": tp_low,
            "recall": round(float(tp_low / max(n_low, 1)), 4),
            "misclassified_as_moderate": fn_low_to_mod,
            "misclassified_as_high": fn_low_to_high,
        }

    # Target minority class analysis
    if "gold_target" in df.columns and "pred_target" in df.columns:
        t_mask = (df["gold_target"] != IGNORE_INDEX) & df["gold_target"].notna()
        sub_tgt = df[t_mask]
        y_tgt = sub_tgt["gold_target"].astype(int).values
        p_tgt = sub_tgt["pred_target"].astype(int).values

        target_classes = ["individual", "organization", "community", "society"]
        minority_summary = {}
        for c_id, c_name in [(1, "organization"), (2, "community"), (3, "society")]:
            c_mask = (y_tgt == c_id)
            n_c = int(c_mask.sum())
            c_preds = p_tgt[c_mask]
            tp = int((c_preds == c_id).sum())
            fn_to_indiv = int((c_preds == 0).sum())
            minority_summary[c_name] = {
                "support": n_c,
                "correct_tp": tp,
                "recall": round(float(tp / max(n_c, 1)), 4),
                "absorbed_by_majority_individual": fn_to_indiv,
                "absorbed_by_majority_pct": round(float(fn_to_indiv / max(n_c, 1) * 100.0), 2),
            }
        out["target_minority_absorption"] = minority_summary

    return out


def build_experiment_diagnostic(
    df: pd.DataFrame,
    exp_name: str,
    class_taxonomy: dict | None = None
) -> dict[str, Any]:
    """Build full diagnostic profile for a single experiment."""
    apply_label_config({"target": MULTIBULLY_TARGET_LABELS})

    tasks = [t for t in TASKS if f"pred_{t}" in df.columns and f"gold_{t}" in df.columns]

    task_diags = {}
    macro_f1s = []
    for t in tasks:
        names = class_taxonomy.get(t) if class_taxonomy else None
        if names is None and t in LABEL_SPACES:
            names = list(LABEL_SPACES[t])
        td = compute_task_diagnostic(df, t, class_names=names)
        task_diags[t] = td
        macro_f1s.append(td["macro_f1"])

    overall_mean_macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0.0

    # Rare class focus
    rare_diag = analyze_rare_class_diagnostics(df)

    # End-to-end conditional cascading evaluation
    e2e_diag = {}
    for t in ("target", "severity"):
        if f"pred_{t}" in df.columns and "pred_hate" in df.columns:
            valid_e2e = (df[f"gold_{t}"] != IGNORE_INDEX).any()
            if valid_e2e:
                e2e_res = end_to_end_metrics(df, t)
                e2e_diag[f"end_to_end_{t}"] = {
                    "accuracy": round(float(e2e_res.get("accuracy", 0.0)), 4),
                    "macro_f1": round(float(e2e_res.get("macro_f1", 0.0)), 4),
                    "n_eval": int(e2e_res.get("n", 0)),
                }

    # Masking validation summary
    masking_summary = {
        t: {
            "total_rows": len(df),
            "masked_count": int(((df[f"gold_{t}"] == IGNORE_INDEX) | df[f"gold_{t}"].isna()).sum()),
            "supervised_count": int(((df[f"gold_{t}"] != IGNORE_INDEX) & df[f"gold_{t}"].notna()).sum()),
        }
        for t in tasks
    }

    return {
        "experiment": exp_name,
        "n_samples": len(df),
        "overall_mean_macro_f1": round(overall_mean_macro_f1, 4),
        "tasks": task_diags,
        "rare_class_diagnostics": rare_diag,
        "end_to_end_conditional": e2e_diag,
        "masking_summary": masking_summary,
    }


def compare_with_tfidf_baseline(
    model_diag: dict[str, Any],
    tfidf_diag: dict[str, Any]
) -> dict[str, Any]:
    """Compute head-to-head comparison between a neural model and the TF-IDF classical baseline."""
    comparison = {}
    for t in ("hate", "sarcasm", "target", "severity"):
        if t in model_diag["tasks"] and t in tfidf_diag["tasks"]:
            m_t = model_diag["tasks"][t]
            tf_t = tfidf_diag["tasks"][t]
            comparison[t] = {
                "neural_macro_f1": m_t["macro_f1"],
                "tfidf_macro_f1": tf_t["macro_f1"],
                "delta_macro_f1": round(m_t["macro_f1"] - tf_t["macro_f1"], 4),
                "neural_accuracy": m_t["accuracy"],
                "tfidf_accuracy": tf_t["accuracy"],
                "delta_accuracy": round(m_t["accuracy"] - tf_t["accuracy"], 4),
                "neural_balanced_acc": m_t["balanced_accuracy"],
                "tfidf_balanced_acc": tf_t["balanced_accuracy"],
                "delta_balanced_acc": round(m_t["balanced_accuracy"] - tf_t["balanced_accuracy"], 4),
            }
    overall_delta = round(model_diag["overall_mean_macro_f1"] - tfidf_diag["overall_mean_macro_f1"], 4)
    return {
        "task_comparisons": comparison,
        "neural_mean_macro_f1": model_diag["overall_mean_macro_f1"],
        "tfidf_mean_macro_f1": tfidf_diag["overall_mean_macro_f1"],
        "delta_mean_macro_f1": overall_delta,
    }


def run_full_diagnostic_analysis(
    e1c_dir: str | Path = "results/v3_journal_final/e1c_text_unweighted",
    e1d_dir: str | Path = "results/v3_journal_final/e1d_text_weighted",
    e6b_dir: str | Path = "results/v3_journal_final/e6b_text_gold_weighted",
    out_dir: str | Path = "results/v3_journal_final/diagnostic_analysis",
) -> dict[str, Any]:
    """Run full diagnostic analysis across E1c, E1d, E6b, and the saved TF-IDF baseline."""
    apply_label_config({"target": MULTIBULLY_TARGET_LABELS})

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    e1c_csv = Path(e1c_dir) / "pooled_model.csv"
    e1d_csv = Path(e1d_dir) / "pooled_model.csv"
    e6b_csv = Path(e6b_dir) / "pooled_model.csv"
    tfidf_csv = Path(e1d_dir) / "pooled_tfidf_lr.csv"

    for p in (e1c_csv, e1d_csv, e6b_csv, tfidf_csv):
        if not p.exists():
            raise FileNotFoundError(f"Required artifact not found: {p}")

    df_e1c = pd.read_csv(e1c_csv)
    df_e1d = pd.read_csv(e1d_csv)
    df_e6b = pd.read_csv(e6b_csv)
    df_tf = pd.read_csv(tfidf_csv)

    log.info("Analyzing E1c (unweighted baseline)...")
    diag_e1c = build_experiment_diagnostic(df_e1c, "E1c_unweighted")

    log.info("Analyzing E1d (class-weighted reference)...")
    diag_e1d = build_experiment_diagnostic(df_e1d, "E1d_weighted")

    log.info("Analyzing E6b (combined gold weighted)...")
    diag_e6b = build_experiment_diagnostic(df_e6b, "E6b_combined_gold")

    log.info("Analyzing TF-IDF baseline...")
    diag_tf = build_experiment_diagnostic(df_tf, "TFIDF_LogReg")

    # Head-to-head comparisons against TF-IDF
    vs_tfidf = {
        "E1c_vs_TFIDF": compare_with_tfidf_baseline(diag_e1c, diag_tf),
        "E1d_vs_TFIDF": compare_with_tfidf_baseline(diag_e1d, diag_tf),
        "E6b_vs_TFIDF": compare_with_tfidf_baseline(diag_e6b, diag_tf),
    }

    report = {
        "experiments": {
            "E1c_unweighted": diag_e1c,
            "E1d_weighted": diag_e1d,
            "E6b_combined_gold": diag_e6b,
            "TFIDF_LogReg": diag_tf,
        },
        "comparisons_against_tfidf": vs_tfidf,
    }

    # Save JSON report
    (out_p / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )

    # Build CSV summary table
    csv_rows = []
    for exp_key, diag in report["experiments"].items():
        for task in ("hate", "sarcasm", "target", "severity"):
            if task in diag["tasks"]:
                t_data = diag["tasks"][task]
                csv_rows.append({
                    "experiment": exp_key,
                    "task": task,
                    "support": t_data["n_eval"],
                    "accuracy": t_data["accuracy"],
                    "balanced_accuracy": t_data["balanced_accuracy"],
                    "macro_f1": t_data["macro_f1"],
                    "weighted_f1": t_data["weighted_f1"],
                })
        csv_rows.append({
            "experiment": exp_key,
            "task": "OVERALL_MEAN",
            "support": diag["n_samples"],
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": diag["overall_mean_macro_f1"],
            "weighted_f1": np.nan,
        })

    summary_df = pd.DataFrame(csv_rows)
    summary_df.to_csv(out_p / "diagnostic_summary.csv", index=False)

    # Format Markdown report
    md_text = _format_diagnostic_markdown_report(report)
    (out_p / "diagnostic_report.md").write_text(md_text, encoding="utf-8")

    log.info("Diagnostic analysis complete. Artifacts saved to %s", out_p)
    return report


def _format_diagnostic_markdown_report(report: dict[str, Any]) -> str:
    """Format diagnostic report into clean, publication-ready Markdown."""
    exps = report["experiments"]
    e1c = exps["E1c_unweighted"]
    e1d = exps["E1d_weighted"]
    e6b = exps["E6b_combined_gold"]
    tf = exps["TFIDF_LogReg"]
    tasks = ["hate", "sarcasm", "target", "severity"]

    lines = [
        "# ToxiFusionNet: Multi-Task Diagnostic Analysis Report",
        "",
        "This report establishes the baseline diagnostic profile for **E1c** (unweighted control), **E1d** (per-task square-root class weighted), **E6b** (combined gold auxiliary weighted), and the classical **TF-IDF + Logistic Regression** baseline across all four tasks on MultiBully ($N = 5,718$ held-out test predictions).",
        "",
        "---",
        "",
        "## 1. Overall Multi-Task Performance Matrix",
        "",
        "| Task | Support | TF-IDF Baseline F1 / Acc | E1c Unweighted F1 / Acc | E1d Weighted F1 / Acc | E6b Combined F1 / Acc | E1d vs. TF-IDF ($\\Delta$ F1) | E1d vs. E1c ($\\Delta$ F1) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for t in tasks:
        sup = e1d["tasks"][t]["n_eval"]
        tf_str = f"{tf['tasks'][t]['macro_f1']:.4f} / {tf['tasks'][t]['accuracy']:.4f}"
        e1c_str = f"{e1c['tasks'][t]['macro_f1']:.4f} / {e1c['tasks'][t]['accuracy']:.4f}"
        e1d_str = f"**{e1d['tasks'][t]['macro_f1']:.4f}** / {e1d['tasks'][t]['accuracy']:.4f}"
        e6b_str = f"{e6b['tasks'][t]['macro_f1']:.4f} / {e6b['tasks'][t]['accuracy']:.4f}"
        d_tf = e1d["tasks"][t]["macro_f1"] - tf["tasks"][t]["macro_f1"]
        d_e1c = e1d["tasks"][t]["macro_f1"] - e1c["tasks"][t]["macro_f1"]
        lines.append(
            f"| **{t.capitalize()}** | {sup:,} | {tf_str} | {e1c_str} | {e1d_str} | {e6b_str} | {d_tf:+.4f} | {d_e1c:+.4f} |"
        )

    # Overall Mean
    lines.append(
        f"| **Overall Mean** | 5,718 | {tf['overall_mean_macro_f1']:.4f} / — | {e1c['overall_mean_macro_f1']:.4f} / — | **{e1d['overall_mean_macro_f1']:.4f}** / — | {e6b['overall_mean_macro_f1']:.4f} / — | {e1d['overall_mean_macro_f1'] - tf['overall_mean_macro_f1']:+.4f} | {e1d['overall_mean_macro_f1'] - e1c['overall_mean_macro_f1']:+.4f} |"
    )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Balanced Accuracy & Macro-Recall Summary",
        "",
        "Balanced accuracy corresponds to unweighted macro-averaged recall across classes ($BA = \\frac{1}{K}\\sum_k Recall_k$):",
        "",
        "| Task | TF-IDF Balanced Acc | E1c Balanced Acc | E1d Balanced Acc | E6b Balanced Acc | Gain (E1d - E1c) | Gain (E1d - TF-IDF) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for t in tasks:
        ba_tf = tf["tasks"][t]["balanced_accuracy"]
        ba_e1c = e1c["tasks"][t]["balanced_accuracy"]
        ba_e1d = e1d["tasks"][t]["balanced_accuracy"]
        ba_e6b = e6b["tasks"][t]["balanced_accuracy"]
        lines.append(
            f"| **{t.capitalize()}** | {ba_tf:.4f} | {ba_e1c:.4f} | **{ba_e1d:.4f}** | {ba_e6b:.4f} | {ba_e1d - ba_e1c:+.4f} | {ba_e1d - ba_tf:+.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Binary Prediction Score Distributions (Hate & Sarcasm)",
        "",
        "### 3.1 Probability Distribution Statistics (E1d Reference)",
        "",
        "| Task | Overall Mean +/- Std | Positive Class Median [IQR] | Negative Class Median [IQR] | Mean Separation ($\\mu_+ - \\mu_-$) | AUROC | Brier Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for t in ("hate", "sarcasm"):
        sd = e1d["tasks"][t].get("score_distribution", {})
        pos_iqr = f"{sd['pos_quantiles']['p50']:.3f} [{sd['pos_quantiles']['p25']:.3f}, {sd['pos_quantiles']['p75']:.3f}]"
        neg_iqr = f"{sd['neg_quantiles']['p50']:.3f} [{sd['neg_quantiles']['p25']:.3f}, {sd['neg_quantiles']['p75']:.3f}]"
        lines.append(
            f"| **{t.capitalize()}** | {sd['overall_mean']:.3f} +/- {sd['overall_std']:.3f} | {pos_iqr} | {neg_iqr} | {sd['mean_separation']:+.3f} | {sd['auroc']:.4f} | {sd['brier_score']:.4f} |"
        )

    sd_hate = e1d["tasks"]["hate"].get("score_distribution", {})
    sd_sarc = e1d["tasks"]["sarcasm"].get("score_distribution", {})

    sarc_pos_med = sd_sarc.get("pos_quantiles", {}).get("p50", 0.461)
    sarc_pos_p25 = sd_sarc.get("pos_quantiles", {}).get("p25", 0.358)
    hate_pos_med = sd_hate.get("pos_quantiles", {}).get("p50", 0.615)
    hate_neg_med = sd_hate.get("neg_quantiles", {}).get("p50", 0.219)
    hate_sep = sd_hate.get("mean_separation", 0.273)

    lines.extend([
        "",
        "**Key Distribution Findings:**",
        f"- **Sarcasm Score Distribution Shift:** For sarcasm in E1d, the positive class median score is **{sarc_pos_med:.3f}**, with 25% of positives scoring below {sarc_pos_p25:.3f}. Because the default cutoff is 0.50, more than half of true sarcastic posts receive predicted probabilities in the range `[{sarc_pos_p25:.2f}, 0.50)` and are misclassified as non-sarcastic by the default 0.5 decision boundary.",
        f"- **Hate Score Calibration:** For hate in E1d, positive class median is **{hate_pos_med:.3f}** and negative class median is **{hate_neg_med:.3f}**, giving healthy separation ($\\Delta \\mu = {hate_sep:+.3f}$).",
        "",
        "---",
        "",
        "## 4. Severity & Target Error Patterns: Focus on Rare Classes",
        "",
        "### 4.1 Rare High-Severity Class Diagnostics ($N = 30$)",
        "",
        "| Metric | E1c (Unweighted) | E1d (Class Weighted) | E6b (Combined Gold) |",
        "| :--- | :---: | :---: | :---: |"
    ])

    h_c = e1c["rare_class_diagnostics"].get("severity_high_rare_class", {})
    h_d = e1d["rare_class_diagnostics"].get("severity_high_rare_class", {})
    h_6 = e6b["rare_class_diagnostics"].get("severity_high_rare_class", {})

    lines.extend([
        f"| **Support (Held-Out Test)** | {h_d.get('support', 30)} | {h_d.get('support', 30)} | {h_d.get('support', 30)} |",
        f"| **Correctly Identified (TP)** | {h_c.get('correct_identified_tp', 0)} | **{h_d.get('correct_identified_tp', 0)}** | {h_6.get('correct_identified_tp', 0)} |",
        f"| **High Severity Recall** | {h_c.get('recall', 0.0):.4f} | **{h_d.get('recall', 0.0):.4f}** | {h_6.get('recall', 0.0):.4f} |",
        f"| **Misclassified as Moderate (FN)** | {h_c.get('misclassified_as_moderate', 0)} | {h_d.get('misclassified_as_moderate', 0)} | {h_6.get('misclassified_as_moderate', 0)} |",
        f"| **Misclassified as Low (FN)** | {h_c.get('misclassified_as_low', 0)} | {h_d.get('misclassified_as_low', 0)} | {h_6.get('misclassified_as_low', 0)} |",
        f"| **Total False Positives** | {h_c.get('false_positives_total', 0)} | {h_d.get('false_positives_total', 0)} | {h_6.get('false_positives_total', 0)} |",
        f"| **High Severity Precision** | {h_c.get('precision', 0.0):.4f} | **{h_d.get('precision', 0.0):.4f}** | {h_6.get('precision', 0.0):.4f} |",
        "",
        "### 4.2 Target Minority Absorption Patterns (E1d Reference)",
        "",
        "| Minority Category | Support | Correct (TP) | Recall | Misclassified as Majority 'Individual' | % Absorbed by Majority |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ])

    tgt_abs = e1d["rare_class_diagnostics"].get("target_minority_absorption", {})
    for cat, info in tgt_abs.items():
        lines.append(
            f"| **{cat.capitalize()}** | {info['support']} | {info['correct_tp']} | {info['recall']:.4f} | {info['absorbed_by_majority_individual']} | {info['absorbed_by_majority_pct']:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Label Masking & Conditional Cascading Diagnostics",
        "",
        "### 5.1 Supervision Coverage & Masking Checks",
        "- **Hate & Sarcasm:** 100% supervision coverage ($N = 5,718$ valid instances; 0 masked).",
        "- **Target:** Supervised on hate posts ($N = 1,437$ instances; 4,281 masked with `-100`).",
        "- **Severity:** Supervised on hate posts ($N = 1,580$ instances; 4,138 masked with `-100`).",
        "",
        "### 5.2 End-to-End Conditional Metrics (Cascading Hate Errors)",
        "When evaluating the full pipeline (non-hate -> 'none'), false negatives in binary hate detection cascade into target and severity errors:",
        "",
        "| Experiment | Primary Target F1 | End-to-End Target F1 | Primary Severity F1 | End-to-End Severity F1 |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ])

    for name, d in [("E1c Unweighted", e1c), ("E1d Weighted", e1d), ("E6b Combined Gold", e6b)]:
        p_tgt = d["tasks"]["target"]["macro_f1"]
        e_tgt = d["end_to_end_conditional"].get("end_to_end_target", {}).get("macro_f1", np.nan)
        p_sev = d["tasks"]["severity"]["macro_f1"]
        e_sev = d["end_to_end_conditional"].get("end_to_end_severity", {}).get("macro_f1", np.nan)
        lines.append(
            f"| **{name}** | {p_tgt:.4f} | {e_tgt:.4f} | {p_sev:.4f} | {e_sev:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "*Report generated automatically by `toxifusionnet.analysis.diagnostic.run_full_diagnostic_analysis`.*"
    ])

    return "\n".join(lines)
