"""Significance testing and inferential statistics for model comparisons.

Provides:
- Fast vectorized Macro-F1 computation (fast_macro_f1).
- Two-sided paired group-aware permutation tests with plus-one correction.
- Paired group-aware bootstrap confidence intervals (resampling text clusters).
- Multitask paired group bootstrap and permutation test for overall Mean Macro-F1.
- Family-wise error rate control via Holm-Bonferroni step-down correction.
- Edwards continuity-corrected McNemar tests for paired binary tasks.
- Class-level precision, recall, F1, and support calculation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Vectorized calculation of macro-averaged F1 score across n_classes (0 to n_classes-1).
    Numerically identical to sklearn.metrics.f1_score(average='macro', zero_division=0)
    over the evaluated label space, but optimized for high-throughput bootstrap and permutation loops.
    """
    # y_true is in [0, n_classes - 1]. y_pred is in [0, n_classes] where n_classes represents unknown/oov.
    k = n_classes + 1
    cm = np.bincount(y_true * k + y_pred, minlength=n_classes * k).reshape(n_classes, k)
    tp = np.diag(cm[:, :n_classes])
    fp = cm[:, :n_classes].sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    return float(f1.mean())


def _encode_classes(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Helper to map string/numeric labels to integer indices 0..C-1, with unknown predictions mapped to C."""
    uniq_classes = np.unique(y)
    n_classes = len(uniq_classes)
    c_map = {c: i for i, c in enumerate(uniq_classes)}
    y_enc = np.array([c_map[val] for val in y], dtype=np.int64)
    a_enc = np.array([c_map.get(val, n_classes) for val in a], dtype=np.int64)
    b_enc = np.array([c_map.get(val, n_classes) for val in b], dtype=np.int64)
    return y_enc, a_enc, b_enc, n_classes


def paired_group_permutation_test(
    groups: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
    pred_a: np.ndarray | pd.Series,
    pred_b: np.ndarray | pd.Series,
    n_perm: int = 10000,
    seed: int = 42,
    mask: np.ndarray | pd.Series | None = None
) -> dict:
    """Two-sided paired, group-aware permutation test for Macro-F1 differences (Model B - Model A).

    Parameters:
    - groups: Cluster / text group identifiers. Examples with identical text share the same group.
    - y_true: Ground truth gold labels.
    - pred_a: Model A predictions (paired with y_true).
    - pred_b: Model B predictions (paired with y_true).
    - n_perm: Number of random permutations (default: 10,000).
    - seed: Random seed for reproducible permutations.
    - mask: Optional boolean mask to restrict evaluation to a subset.

    Null Hypothesis (H0):
    Model A and Model B are exchangeable under H0. For each text group, model assignments
    are swapped with probability 0.5 across all examples sharing that group, preserving pairing
    and clustering structure.

    Two-sided empirical p-value uses the unbiased plus-one correction:
    p = (sum(|T_b| >= |T_obs|) + 1) / (B + 1).
    """
    g, y, a, b = (np.asarray(x) for x in (groups, y_true, pred_a, pred_b))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        g, y, a, b = g[m], y[m], a[m], b[m]

    valid = (y != -100) & (y != "NA") & pd.notna(y)
    g, y, a, b = g[valid], y[valid], a[valid], b[valid]

    if len(y) == 0:
        return {"obs_delta": 0.0, "p_value_two_sided": 1.0, "n_perm": n_perm, "seed": seed, "n_samples": 0, "n_groups": 0}

    y_enc, a_enc, b_enc, n_classes = _encode_classes(y, a, b)

    obs_a = fast_macro_f1(y_enc, a_enc, n_classes)
    obs_b = fast_macro_f1(y_enc, b_enc, n_classes)
    obs_delta = obs_b - obs_a
    abs_obs_delta = abs(obs_delta)

    uniq_groups, group_indices = np.unique(g, return_inverse=True)
    n_groups = len(uniq_groups)

    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=(n_perm, n_groups), dtype=bool)

    extreme_count = 0
    perm_deltas = np.empty(n_perm, dtype=np.float64)

    for i in range(n_perm):
        swap = flips[i, group_indices]
        cur_a = np.where(swap, b_enc, a_enc)
        cur_b = np.where(swap, a_enc, b_enc)
        d = fast_macro_f1(y_enc, cur_b, n_classes) - fast_macro_f1(y_enc, cur_a, n_classes)
        perm_deltas[i] = d
        if abs(d) >= abs_obs_delta - 1e-12:
            extreme_count += 1

    p_value = float((extreme_count + 1.0) / (n_perm + 1.0))
    return {
        "obs_delta": float(obs_delta),
        "p_value_two_sided": float(p_value),
        "extreme_count": int(extreme_count),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "n_samples": int(len(y)),
        "n_groups": int(n_groups)
    }


def paired_group_bootstrap(
    groups: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
    pred_a: np.ndarray | pd.Series,
    pred_b: np.ndarray | pd.Series,
    metric=None,
    n_boot: int = 10000,
    seed: int = 42,
    mask: np.ndarray | pd.Series | None = None
) -> dict:
    """Cluster/group-aware paired bootstrap for Macro-F1 difference (Model B - Model A).
    Resamples text groups with replacement to respect clustering and prevent standard error underestimation.

    Returns:
    - delta_mean: Empirical mean of bootstrap deltas.
    - ci95: [2.5th percentile, 97.5th percentile] confidence interval.
    - bootstrap_prob_superiority: Empirical probability P(delta > 0).
    - bootstrap_prob_inferiority: Empirical probability P(delta <= 0).
    - p_value_bootstrap: Provided for backward compatibility; represents P(delta <= 0).
    """
    g, y, a, b = (np.asarray(x) for x in (groups, y_true, pred_a, pred_b))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        g, y, a, b = g[m], y[m], a[m], b[m]

    valid = (y != -100) & (y != "NA") & pd.notna(y)
    g, y, a, b = g[valid], y[valid], a[valid], b[valid]

    if len(y) == 0:
        return {"delta_mean": 0.0, "ci95": [0.0, 0.0], "bootstrap_prob_superiority": 0.0,
                "bootstrap_prob_inferiority": 1.0, "p_value": 1.0, "p_value_bootstrap": 1.0, "n_samples": 0, "n_groups": 0}

    uniq_groups, group_indices = np.unique(g, return_inverse=True)
    n_groups = len(uniq_groups)
    group_to_rows = [np.where(group_indices == gi)[0] for gi in range(n_groups)]
    rng = np.random.default_rng(seed)

    if metric is None:
        y_enc, a_enc, b_enc, n_classes = _encode_classes(y, a, b)
        deltas = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            sampled_g = rng.integers(0, n_groups, n_groups)
            sampled_rows = np.concatenate([group_to_rows[gi] for gi in sampled_g])
            deltas[i] = fast_macro_f1(y_enc[sampled_rows], b_enc[sampled_rows], n_classes) - \
                        fast_macro_f1(y_enc[sampled_rows], a_enc[sampled_rows], n_classes)
    else:
        deltas = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            sampled_g = rng.integers(0, n_groups, n_groups)
            sampled_rows = np.concatenate([group_to_rows[gi] for gi in sampled_g])
            deltas[i] = metric(y[sampled_rows], b[sampled_rows]) - metric(y[sampled_rows], a[sampled_rows])

    prob_sup = float((deltas > 0).mean())
    prob_inf = float((deltas <= 0).mean())
    return {
        "delta_mean": float(deltas.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "bootstrap_prob_superiority": prob_sup,
        "bootstrap_prob_inferiority": prob_inf,
        "p_value": prob_inf,
        "p_value_bootstrap": prob_inf,
        "n_samples": int(len(y)),
        "n_groups": int(n_groups),
        "n_boot": int(n_boot),
        "seed": int(seed)
    }


def paired_multitask_group_bootstrap(
    groups: np.ndarray | pd.Series,
    y_trues: dict[str, np.ndarray | pd.Series],
    preds_a: dict[str, np.ndarray | pd.Series],
    preds_b: dict[str, np.ndarray | pd.Series],
    tasks: list[str],
    n_boot: int = 10000,
    seed: int = 42
) -> dict:
    """Paired group-aware cluster bootstrap for the overall mean of Macro-F1 scores across multiple tasks.
    Resamples unique text groups with replacement, evaluates Macro-F1 on each task's valid subset,
    and aggregates via simple arithmetic mean across tasks.
    """
    g = np.asarray(groups)
    uniq_groups, group_indices = np.unique(g, return_inverse=True)
    n_groups = len(uniq_groups)
    group_to_rows = [np.where(group_indices == gi)[0] for gi in range(n_groups)]

    task_data = {}
    obs_scores_a = {}
    obs_scores_b = {}
    for t in tasks:
        yt = np.asarray(y_trues[t])
        pa = np.asarray(preds_a[t])
        pb = np.asarray(preds_b[t])
        valid = (yt != -100) & (yt != "NA") & pd.notna(yt)
        yt_v, pa_v, pb_v = yt[valid], pa[valid], pb[valid]
        y_enc, a_enc, b_enc, nc = _encode_classes(yt_v, pa_v, pb_v)
        row_indices = np.where(valid)[0]
        task_data[t] = {
            "y_enc": y_enc, "a_enc": a_enc, "b_enc": b_enc, "nc": nc,
            "orig_rows": row_indices
        }
        obs_scores_a[t] = fast_macro_f1(y_enc, a_enc, nc)
        obs_scores_b[t] = fast_macro_f1(y_enc, b_enc, nc)

    obs_mean_a = float(np.mean(list(obs_scores_a.values())))
    obs_mean_b = float(np.mean(list(obs_scores_b.values())))
    obs_delta = obs_mean_b - obs_mean_a

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)

    # For mapping sampled global rows to task-specific rows fast:
    n_total_rows = len(g)
    for i in range(n_boot):
        sampled_g = rng.integers(0, n_groups, n_groups)
        sampled_rows = np.concatenate([group_to_rows[gi] for gi in sampled_g])
        # Count frequency of each row in the sample
        row_counts = np.bincount(sampled_rows, minlength=n_total_rows)

        task_deltas = []
        for t in tasks:
            td = task_data[t]
            orig_r = td["orig_rows"]
            sub_counts = row_counts[orig_r]
            # Repeat task indices according to sampled frequency
            if sub_counts.sum() == 0:
                continue
            rep_idx = np.repeat(np.arange(len(orig_r)), sub_counts)
            f1_a = fast_macro_f1(td["y_enc"][rep_idx], td["a_enc"][rep_idx], td["nc"])
            f1_b = fast_macro_f1(td["y_enc"][rep_idx], td["b_enc"][rep_idx], td["nc"])
            task_deltas.append(f1_b - f1_a)
        deltas[i] = np.mean(task_deltas)

    prob_sup = float((deltas > 0).mean())
    prob_inf = float((deltas <= 0).mean())
    return {
        "mean_macro_f1_a": obs_mean_a,
        "mean_macro_f1_b": obs_mean_b,
        "obs_delta": float(obs_delta),
        "delta_mean": float(deltas.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "bootstrap_prob_superiority": prob_sup,
        "bootstrap_prob_inferiority": prob_inf,
        "p_value_bootstrap": prob_inf,
        "tasks": tasks,
        "n_boot": int(n_boot),
        "n_groups": int(n_groups),
        "seed": int(seed)
    }


def paired_multitask_group_permutation_test(
    groups: np.ndarray | pd.Series,
    y_trues: dict[str, np.ndarray | pd.Series],
    preds_a: dict[str, np.ndarray | pd.Series],
    preds_b: dict[str, np.ndarray | pd.Series],
    tasks: list[str],
    n_perm: int = 10000,
    seed: int = 42
) -> dict:
    """Two-sided paired, group-aware permutation test for the overall mean Macro-F1 across tasks.
    Swaps Model A and Model B assignments at the group level across all tasks simultaneously,
    computing the two-sided empirical p-value with plus-one correction.
    """
    g = np.asarray(groups)
    uniq_groups, group_indices = np.unique(g, return_inverse=True)
    n_groups = len(uniq_groups)

    task_data = {}
    obs_scores_a = {}
    obs_scores_b = {}
    for t in tasks:
        yt = np.asarray(y_trues[t])
        pa = np.asarray(preds_a[t])
        pb = np.asarray(preds_b[t])
        valid = (yt != -100) & (yt != "NA") & pd.notna(yt)
        yt_v, pa_v, pb_v = yt[valid], pa[valid], pb[valid]
        y_enc, a_enc, b_enc, nc = _encode_classes(yt_v, pa_v, pb_v)
        task_data[t] = {
            "y_enc": y_enc, "a_enc": a_enc, "b_enc": b_enc, "nc": nc,
            "group_indices": group_indices[valid]
        }
        obs_scores_a[t] = fast_macro_f1(y_enc, a_enc, nc)
        obs_scores_b[t] = fast_macro_f1(y_enc, b_enc, nc)

    obs_mean_a = float(np.mean(list(obs_scores_a.values())))
    obs_mean_b = float(np.mean(list(obs_scores_b.values())))
    obs_delta = obs_mean_b - obs_mean_a
    abs_obs_delta = abs(obs_delta)

    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=(n_perm, n_groups), dtype=bool)

    extreme_count = 0
    perm_deltas = np.empty(n_perm, dtype=np.float64)

    for i in range(n_perm):
        flip_i = flips[i]
        task_deltas = []
        for t in tasks:
            td = task_data[t]
            swap = flip_i[td["group_indices"]]
            cur_a = np.where(swap, td["b_enc"], td["a_enc"])
            cur_b = np.where(swap, td["a_enc"], td["b_enc"])
            task_deltas.append(fast_macro_f1(td["y_enc"], cur_b, td["nc"]) - fast_macro_f1(td["y_enc"], cur_a, td["nc"]))
        d = float(np.mean(task_deltas))
        perm_deltas[i] = d
        if abs(d) >= abs_obs_delta - 1e-12:
            extreme_count += 1

    p_value = float((extreme_count + 1.0) / (n_perm + 1.0))
    return {
        "mean_macro_f1_a": obs_mean_a,
        "mean_macro_f1_b": obs_mean_b,
        "obs_delta": float(obs_delta),
        "p_value_two_sided": float(p_value),
        "extreme_count": int(extreme_count),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "tasks": tasks,
        "n_groups": int(n_groups)
    }


def holm_bonferroni_correction(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Apply Holm-Bonferroni step-down correction for family-wise error rate control.

    Parameters:
    - p_values: List or 1D array of raw p-values from M simultaneous hypothesis tests.

    Returns:
    - 1D numpy array of adjusted p-values in the same order as input.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    if m == 1:
        return np.clip(p, 0.0, 1.0)

    order = np.argsort(p)
    sorted_p = p[order]
    adj = np.empty(m, dtype=float)
    cum_max = 0.0
    for i in range(m):
        step_val = (m - i) * sorted_p[i]
        cum_max = max(cum_max, step_val)
        adj[i] = min(1.0, cum_max)

    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


def mcnemar_test(
    y_true: pd.Series | np.ndarray,
    pred_a: pd.Series | np.ndarray,
    pred_b: pd.Series | np.ndarray
) -> dict:
    """Edwards continuity-corrected McNemar test on paired binary predictions.
    Also computes exact two-sided binomial p-value when discordant count < 25.
    """
    from scipy.stats import chi2, binomtest
    y, a, b = (np.asarray(x) for x in (y_true, pred_a, pred_b))
    valid = (y != -100) & (y != "NA") & pd.notna(y)
    y, a, b = y[valid], a[valid], b[valid]

    corr_a = (a == y)
    corr_b = (b == y)

    n00 = int(((~corr_a) & (~corr_b)).sum())
    n01 = int(((~corr_a) & corr_b).sum())  # B correct, A incorrect
    n10 = int((corr_a & (~corr_b)).sum())  # A correct, B incorrect
    n11 = int((corr_a & corr_b).sum())

    discordant = n01 + n10
    diff = abs(n01 - n10)
    # Edwards continuity correction
    stat = ((diff - 1.0) ** 2) / max(discordant, 1) if discordant > 0 else 0.0
    p_chi2 = float(chi2.sf(stat, df=1)) if discordant > 0 else 1.0
    odds_ratio = float(n01 / max(n10, 1))

    # Exact binomial test for small sample discordant pairs
    p_exact = float(binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided").pvalue) if discordant > 0 else 1.0

    p_value = p_exact if discordant < 25 else p_chi2
    return {
        "test_variant": "Edwards continuity-corrected McNemar" if discordant >= 25 else "Exact binomial McNemar",
        "contingency_table": {"both_correct": n11, "a_only": n10, "b_only": n01, "both_wrong": n00},
        "discordant_pairs": discordant,
        "statistic": float(stat),
        "p_value": float(p_value),
        "p_value_chi2": float(p_chi2),
        "p_value_exact": float(p_exact),
        "odds_ratio": float(odds_ratio)
    }


def class_level_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    classes: list | None = None
) -> dict:
    """Compute per-class support, precision, recall, and F1 score."""
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    valid = (y != -100) & (y != "NA") & pd.notna(y)
    y, p = y[valid], p[valid]

    if classes is None:
        classes = sorted(list(set(y).union(set(p))))

    res = {}
    for c in classes:
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        support = int((y == c).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        res[str(c)] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    return res


# Backward-compatible baseline functions:
def paired_bootstrap(y_true, pred_a, pred_b, metric=None, n_boot: int = 1000, seed: int = 0, mask=None) -> dict:
    """Paired bootstrap over test examples: delta = metric(B) - metric(A). Returns mean delta, 95% CI and
    one-sided directional bootstrap probability P(delta <= 0). ``mask`` restricts to a subset."""
    y, a, b = (np.asarray(x) for x in (y_true, pred_a, pred_b))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        y, a, b = y[m], a[m], b[m]
    metric = metric or (lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = metric(y[idx], b[idx]) - metric(y[idx], a[idx])
    return {
        "delta_mean": float(deltas.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_value": float((deltas <= 0).mean()),
        "bootstrap_prob_superiority": float((deltas > 0).mean()),
        "n": int(n)
    }


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Legacy McNemar interface; forwards to mcnemar_test."""
    res = mcnemar_test(y_true, pred_a, pred_b)
    table = res["contingency_table"]
    return {
        "a_only_correct": table["a_only"],
        "b_only_correct": table["b_only"],
        "chi2": res["statistic"],
        "p_value": res["p_value"]
    }


def bootstrap_ci(y_true, y_pred, metric=None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Compute 95% bootstrap confidence interval for a single system metric."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    metric = metric or (lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return {"estimate": 0.0, "ci95": [0.0, 0.0], "n": 0}
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[i] = metric(y[idx], p[idx])
    return {
        "estimate": float(metric(y, p)),
        "ci95": [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))],
        "n": int(n)
    }


def group_bootstrap_ci(groups, y_true, y_pred, metric=None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Cluster/group-aware 95% bootstrap confidence interval for a single system metric."""
    g, y, p = (np.asarray(x) for x in (groups, y_true, y_pred))
    metric = metric or (lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)

    uniq_groups, group_indices = np.unique(g, return_inverse=True)
    n_groups = len(uniq_groups)
    group_to_rows = [np.where(group_indices == gi)[0] for gi in range(n_groups)]

    scores = np.empty(n_boot)
    for i in range(n_boot):
        sampled_g = rng.integers(0, n_groups, n_groups)
        sampled_rows = np.concatenate([group_to_rows[gi] for gi in sampled_g])
        scores[i] = metric(y[sampled_rows], p[sampled_rows])

    return {
        "estimate": float(metric(y, p)),
        "ci95": [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))],
        "n_samples": int(len(y)),
        "n_groups": int(n_groups)
    }
