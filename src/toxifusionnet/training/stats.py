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
