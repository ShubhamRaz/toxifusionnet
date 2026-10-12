"""Explainability metrics following the ERASER benchmark (DeYoung et al., 2020) and HateXplain.

Plausibility (needs gold rationales): token-level F1, IOU-F1 (partial-match span F1), AUPRC over soft token scores.
Faithfulness (needs only a probability function): comprehensiveness and sufficiency.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


def top_k_mask(scores: np.ndarray, k: int = 5) -> np.ndarray:
    m = np.zeros_like(scores, dtype=np.float32)
    if len(scores):
        m[np.argsort(-scores)[:k]] = 1.0
    return m


def token_prf(pred: np.ndarray, gold: np.ndarray) -> tuple[float, float, float]:
    tp = float((pred * gold).sum())
    p = tp / max(pred.sum(), 1e-9)
    r = tp / max(gold.sum(), 1e-9)
    f = 2 * p * r / max(p + r, 1e-9)
    return p, r, f


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, v in enumerate(list(mask) + [0]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i))
            start = None
    return spans


def iou_f1(pred: np.ndarray, gold: np.ndarray, thresh: float = 0.5) -> float:
    ps, gs = _spans(pred), _spans(gold)
    if not ps or not gs:
        return 0.0
    def iou(a, b):
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        return inter / max((a[1] - a[0]) + (b[1] - b[0]) - inter, 1e-9)
    tp_p = sum(any(iou(p, g) > thresh for g in gs) for p in ps)
    tp_g = sum(any(iou(p, g) > thresh for p in ps) for g in gs)
    prec, rec = tp_p / len(ps), tp_g / len(gs)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def plausibility(scores_list: Sequence[np.ndarray], gold_list: Sequence[np.ndarray], k: int = 5) -> dict:
    tf, iu, ap = [], [], []
    for sc, g in zip(scores_list, gold_list):
        if g.sum() == 0:
            continue
        pm = top_k_mask(sc, min(k, len(sc)))
        tf.append(token_prf(pm, g)[2])
        iu.append(iou_f1(pm, g))
        ap.append(average_precision_score(g, sc))
    n = max(len(tf), 1)
    return {"token_f1": sum(tf) / n, "iou_f1": sum(iu) / n, "auprc": sum(ap) / n, "n_eval": len(tf)}


def faithfulness(prob_fn: Callable[[list[str]], float], words: list[str], rationale_mask: np.ndarray) -> dict:
    """prob_fn maps a word list to P(predicted class). comprehensiveness = p(x) - p(x \\ r); sufficiency = p(x) - p(r)."""
    full = prob_fn(words)
    keep = [w for w, m in zip(words, rationale_mask) if not m] or ["<empty>"]
    only = [w for w, m in zip(words, rationale_mask) if m] or ["<empty>"]
    return {"comprehensiveness": full - prob_fn(keep), "sufficiency": full - prob_fn(only)}
