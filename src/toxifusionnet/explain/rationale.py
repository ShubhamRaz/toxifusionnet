"""Sarcasm + hate rationale -> ground-truth attention (after Mamun et al., Appl. Sci. 2024, 14(11), 4898).

Pipeline: annotator rationales (binary token masks) -> aggregate across annotators -> combine hate and sarcasm masks with
the paper's rule (Algorithm 1) -> map words to sub-words -> softmax with temperature tau.

Note on Algorithm 1: the paper's text states the combined values for binary masks are 0.5 (sarcasm only), 1 (hate only)
and 1.5 (both). The printed pseudo-code of the third branch (s == h) literally evaluates to h + (s+h)/2 = 2 for s=h=1.
``mode="text"`` (default) reproduces the values stated in the text; ``mode="algorithm"`` reproduces the printed pseudo-code.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np


def _as_obj(spec: Any):
    if spec is None or (isinstance(spec, float) and np.isnan(spec)):
        return None
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s.split()
    return spec


def to_binary_masks(words: Sequence[str], spec: Any) -> list[np.ndarray]:
    """Convert a rationale spec to a list of binary masks (one per annotator), aligned with ``words``.

    Accepted specs (or their JSON-string form): list of 0/1 ints with len(words); list of rationale words;
    list of such lists (several annotators). Word matching is case-insensitive.
    """
    obj = _as_obj(spec)
    if obj is None or obj == []:
        return []
    if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, (list, tuple)) for x in obj):
        return [m for sub in obj for m in to_binary_masks(words, list(sub))]
    obj = list(obj)
    if all(isinstance(x, (int, float, np.integer)) and x in (0, 1) for x in obj) and len(obj) == len(words):
        return [np.asarray(obj, dtype=np.float32)]
    vocab = {str(x).lower() for x in obj}
    mask = np.asarray([1.0 if w.lower() in vocab else 0.0 for w in words], dtype=np.float32)
    return [mask] if mask.any() else []


def aggregate_annotators(masks: list[np.ndarray], n_tokens: int, min_votes: int = 1) -> np.ndarray:
    """Average annotator masks. Returns zeros if fewer than ``min_votes`` annotators provided a rationale.

    Paper: hate/offensive rationales are used when >=2 of 3 annotators agree on the label; sarcasm rationales
    when >=1 annotator marks the post as sarcastic (sarcasm is more subjective).
    """
    if len(masks) < min_votes or not masks:
        return np.zeros(n_tokens, dtype=np.float32)
    return np.mean(np.stack(masks), axis=0).astype(np.float32)


def combine_rationales(h: np.ndarray, s: np.ndarray, mode: str = "text") -> np.ndarray:
    """Element-wise Algorithm 1: merge hate/offensive vector ``h`` and sarcasm vector ``s``."""
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    r = np.zeros_like(h)
    gt, lt, eq = s > h, s < h, s == h
    r[gt] = (s[gt] + h[gt]) / 2.0
    r[lt] = h[lt]
    r[eq] = h[eq] + (s[eq] / 2.0 if mode == "text" else (s[eq] + h[eq]) / 2.0)
    return r


def word_values_to_subwords(values: np.ndarray, word_ids: Sequence[int | None]) -> np.ndarray:
    out = np.zeros(len(word_ids), dtype=np.float32)
    for i, w in enumerate(word_ids):
        if w is not None and w < len(values):
            out[i] = values[w]
    return out


def temperature_softmax(values: np.ndarray, valid: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """Softmax(values/tau) restricted to ``valid`` positions (specials / padding get probability 0)."""
    z = np.where(valid, values / max(tau, 1e-6), -1e9)
    z = z - z.max()
    e = np.exp(z) * valid
    return (e / max(e.sum(), 1e-9)).astype(np.float32)


def ground_truth_attention(words: Sequence[str], word_ids: Sequence[int | None], hate_spec: Any, sarcasm_spec: Any,
                           tau: float = 0.5, mode: str = "text", hate_min_votes: int = 1,
                           sarcasm_min_votes: int = 1, use_sarcasm: bool = True):
    """Return ``(hate_target, sarcasm_target)`` sub-word attention distributions (each ``None`` if no rationale)."""
    n = len(words)
    valid = np.asarray([w is not None for w in word_ids])
    h = aggregate_annotators(to_binary_masks(words, hate_spec), n, hate_min_votes)
    s = aggregate_annotators(to_binary_masks(words, sarcasm_spec), n, sarcasm_min_votes) if use_sarcasm else np.zeros(n, np.float32)
    hate_t = sarc_t = None
    r = combine_rationales(h, s, mode) if use_sarcasm else h
    if r.any():
        hate_t = temperature_softmax(word_values_to_subwords(r, word_ids), valid, tau)
    if s.any():
        sarc_t = temperature_softmax(word_values_to_subwords(s, word_ids), valid, tau)
    return hate_t, sarc_t
