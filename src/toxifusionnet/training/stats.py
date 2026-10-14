"""Significance testing for system comparisons (the project's success criterion is a *statistically significant* gain)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def paired_bootstrap(y_true, pred_a, pred_b, metric=None, n_boot: int = 1000, seed: int = 0, mask=None) -> dict:
    """Paired bootstrap over test examples: delta = metric(B) - metric(A). Returns mean delta, 95% CI and one-sided p
    (probability that B is not better than A). ``mask`` restricts to a subset (e.g. sarcastic hate instances)."""
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
    return {"delta_mean": float(deltas.mean()), "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_value": float((deltas <= 0).mean()), "n": int(n)}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    from scipy.stats import chi2
    y, a, b = (np.asarray(x) for x in (y_true, pred_a, pred_b))
    b01 = int(((a == y) & (b != y)).sum())
    b10 = int(((a != y) & (b == y)).sum())
    stat = (abs(b01 - b10) - 1) ** 2 / max(b01 + b10, 1)
    return {"a_only_correct": b01, "b_only_correct": b10, "chi2": float(stat), "p_value": float(chi2.sf(stat, 1))}


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


def paired_group_bootstrap(groups, y_true, pred_a, pred_b, metric=None, n_boot: int = 1000, seed: int = 0, mask=None) -> dict:
    """Cluster/group-aware paired bootstrap: resamples text clusters (groups) with replacement.
    Prevents standard error underestimation caused by duplicate-text clustering.
    Returns mean delta (B - A), 95% CI and one-sided p-value.
    """
    g, y, a, b = (np.asarray(x) for x in (groups, y_true, pred_a, pred_b))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        g, y, a, b = g[m], y[m], a[m], b[m]
    metric = metric or (lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)

    # Pre-index row indices by unique group
    uniq_groups, group_indices = np.unique(g, return_inverse=True)
    n_groups = len(uniq_groups)
    group_to_rows = [[] for _ in range(n_groups)]
    for row_idx, g_idx in enumerate(group_indices):
        group_to_rows[g_idx].append(row_idx)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        sampled_g = rng.integers(0, n_groups, n_groups)
        sampled_rows = np.concatenate([group_to_rows[gi] for gi in sampled_g])
        deltas[i] = metric(y[sampled_rows], b[sampled_rows]) - metric(y[sampled_rows], a[sampled_rows])

    return {
        "delta_mean": float(deltas.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_value": float((deltas <= 0).mean()),
        "n_samples": int(len(y)),
        "n_groups": int(n_groups)
    }


def group_bootstrap_ci(groups, y_true, y_pred, metric=None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Cluster/group-aware 95% bootstrap confidence interval for a single system metric."""
    g, y, p = (np.asarray(x) for x in (groups, y_true, y_pred))
    metric = metric or (lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)

    uniq_groups, group_indices = np.unique(g, return_inverse=True)
    n_groups = len(uniq_groups)
    group_to_rows = [[] for _ in range(n_groups)]
    for row_idx, g_idx in enumerate(group_indices):
        group_to_rows[g_idx].append(row_idx)

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
