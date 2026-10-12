"""Hand-crafted sarcasm cue features (flow chart: *Sarcasm Detection* -> contextual, lexical, sentiment cues)."""
from __future__ import annotations

import math

import numpy as np

from .lexicons import NEGATIVE, POSITIVE
from .normalize import NormalizedText

CUE_NAMES = [
    "n_laugh", "emoji_pos", "emoji_neg", "n_exclaim", "n_question", "has_ellipsis", "n_quotes", "caps_ratio",
    "n_pos_words", "n_neg_words", "polarity_contrast", "n_markers", "hashtag_sarcasm", "cmi", "switch_rate", "log_len",
]
CUE_DIM = len(CUE_NAMES)


def cue_vector(nt: NormalizedText) -> np.ndarray:
    s = nt.stats
    words = [w.lower() for w in nt.words]
    n_pos = sum(w in POSITIVE for w in words)
    n_neg = sum(w in NEGATIVE for w in words)
    # sentiment contrast: positive wording with negative evidence (negative words / negative or sarcastic emoji)
    neg_evidence = n_neg + s.get("n_emoji_neg", 0) + s.get("n_emoji_sarc", 0)
    pos_evidence = n_pos + s.get("n_emoji_pos", 0)
    contrast = float(n_pos > 0 and neg_evidence > 0) or float(pos_evidence > 0 and s.get("n_emoji_sarc", 0) > 0)
    n = max(len(words), 1)
    sq = lambda x: math.log1p(max(x, 0))  # noqa: E731  (compress counts)
    v = [
        sq(s.get("n_laugh", 0) + s.get("n_emoji_laugh", 0)),
        sq(s.get("n_emoji_pos", 0)),
        sq(s.get("n_emoji_neg", 0) + s.get("n_emoji_sarc", 0)),
        sq(s.get("n_exclaim", 0)),
        sq(s.get("n_question", 0)),
        float(s.get("has_ellipsis", 0)),
        sq(s.get("n_quotes", 0)),
        float(s.get("caps_ratio", 0.0)),
        sq(n_pos),
        sq(n_neg),
        contrast,
        sq(s.get("n_markers", 0)),
        float(s.get("hashtag_sarcasm", 0)),
        nt.cmi / 100.0,
        nt.switches / n,
        math.log1p(n) / 5.0,
    ]
    return np.asarray(v, dtype=np.float32)
