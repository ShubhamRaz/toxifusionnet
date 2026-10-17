"""Evaluation metrics: per-task classification, sarcastic-subset analysis (false-negative rate), joint end-to-end metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)

from ..labels import HATE_LABELS, IGNORE_INDEX, LABEL_SPACES, SARCASM_LABELS, TASKS


def classification_metrics(y_true, y_pred, n_classes: int, p_pos=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(n_classes))
    if len(y_true) == 0:
        return {"n": 0}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {str(c): {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]), "support": int(s[c])} for c in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if p_pos is not None and n_classes == 2 and len(set(y_true.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y_true, p_pos))
    return out


def task_metrics_from_arrays(task: str, y_true, y_pred, p_pos=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    keep = y_true != IGNORE_INDEX
    pp = None if p_pos is None else np.asarray(p_pos)[keep]
    m = classification_metrics(y_true[keep], y_pred[keep], len(LABEL_SPACES[task]), pp)
    m["classes"] = LABEL_SPACES[task]
    return m


def sarcasm_subset_metrics(df: pd.DataFrame) -> dict:
    """Hate-detection quality split by gold sarcasm. df needs gold_hate, gold_sarcasm, pred_hate (ids). The headline
    number is the false-negative rate on sarcastic hate, the failure mode motivating the whole project."""
    out = {}
    d = df[(df["gold_hate"] != IGNORE_INDEX) & (df["gold_sarcasm"] != IGNORE_INDEX)]
    for name, sub in (("sarcastic", d[d.gold_sarcasm == 1]), ("non_sarcastic", d[d.gold_sarcasm == 0])):
        pos = sub[sub.gold_hate == 1]
        neg = sub[sub.gold_hate == 0]
        tp = int((pos.pred_hate == 1).sum())
        fn = int((pos.pred_hate == 0).sum())
        fp = int((neg.pred_hate == 1).sum())
        out[name] = {
            "n": int(len(sub)), "n_hate": int(len(pos)),
            "hate_recall": tp / max(len(pos), 1), "false_negative_rate": fn / max(len(pos), 1),
            "false_positive_rate": fp / max(len(neg), 1),
            "hate_f1": float(f1_score(sub.gold_hate, sub.pred_hate, pos_label=1, zero_division=0)) if len(sub) else 0.0,
        }
    return out


def end_to_end_metrics(df: pd.DataFrame, task: str) -> dict:
    """Joint metric mirroring the flow chart output: non-hate -> 'none', else the target / severity class.
    Errors in hate detection propagate into the downstream labels."""
    none_id = len(LABEL_SPACES[task])
    g = df[f"gold_{task}"].where(df["gold_hate"] == 1, none_id).where(df["gold_hate"] != IGNORE_INDEX)
    pr = df[f"pred_{task}"].where(df["pred_hate"] == 1, none_id)
    keep = g.notna() & ((df["gold_hate"] == 0) | (df[f"gold_{task}"] != IGNORE_INDEX))
    m = classification_metrics(g[keep].astype(int), pr[keep].astype(int), none_id + 1)
    m["classes"] = LABEL_SPACES[task] + ["none"]
    return m


def evaluate_predictions(df: pd.DataFrame, tasks=TASKS) -> dict:
    """df columns: gold_<task>, pred_<task> (ints, IGNORE_INDEX for missing gold), p_sarcasm, p_hate."""
    res: dict = {}
    for t in tasks:
        if f"pred_{t}" not in df:
            continue
        pp = df.get(f"p_{t}") if f"p_{t}" in df else (df.get(f"p_{t}") if t in ("sarcasm", "hate") else None)
        res[t] = task_metrics_from_arrays(t, df[f"gold_{t}"], df[f"pred_{t}"], None if pp is None else pp.to_numpy())
    if "pred_hate" in df and "gold_sarcasm" in df and (df.gold_sarcasm != IGNORE_INDEX).any():
        res["sarcasm_subset"] = sarcasm_subset_metrics(df)
    if "narrative" in df and df["narrative"].notna().any():
        res["by_narrative"] = {}
        for v, sub in df.groupby("narrative"):
            k = task_metrics_from_arrays("hate", sub.gold_hate, sub.pred_hate, sub["p_hate"].to_numpy() if "p_hate" in sub else None)
            res["by_narrative"][str(v)] = {"n": k.get("n"), "macro_f1": k.get("macro_f1")}
    for t in ("target", "severity"):
        if f"pred_{t}" in df and "pred_hate" in df and (df[f"gold_{t}"] != IGNORE_INDEX).any():
            res[f"end_to_end_{t}"] = end_to_end_metrics(df, t)
    scores = [res[t]["macro_f1"] for t in tasks if t in res and res[t].get("n", 0) > 0]
    res["mean_macro_f1"] = float(np.mean(scores)) if scores else 0.0
    return res


def flatten_metrics(res: dict) -> dict:
    """Compact scalar summary for result tables."""
    flat = {"mean_macro_f1": res.get("mean_macro_f1")}
    for t in TASKS:
        if t in res and res[t].get("n", 0) > 0:
            flat[f"{t}_macro_f1"] = res[t]["macro_f1"]
            flat[f"{t}_acc"] = res[t]["accuracy"]
            flat[f"{t}_bal_acc"] = res[t]["macro_recall"]   # balanced accuracy = macro-averaged recall
            if "auroc" in res[t]:
                flat[f"{t}_auroc"] = res[t]["auroc"]
    for k in ("sarcastic", "non_sarcastic"):
        if "sarcasm_subset" in res and k in res["sarcasm_subset"]:
            flat[f"hate_FNR_{k}"] = res["sarcasm_subset"][k]["false_negative_rate"]
            flat[f"hate_F1_{k}"] = res["sarcasm_subset"][k]["hate_f1"]
    for t in ("target", "severity"):
        if f"end_to_end_{t}" in res:
            flat[f"e2e_{t}_macro_f1"] = res[f"end_to_end_{t}"]["macro_f1"]
    return flat
