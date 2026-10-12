"""Heuristic (weak) labels for sarcasm and severity, used ONLY when a dataset has no gold annotation.

Results measured against weak labels quantify agreement with these heuristics, not true sarcasm / severity
detection; they must be reported as such. ``label_source`` in the canonical file records which tasks are weak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..labels import HATE_LABELS, SARCASM_LABELS, SEVERITY_LABELS
from ..preprocessing.features import CUE_NAMES, cue_vector
from ..preprocessing.lexicons import ABUSE, THREAT_PHRASES
from ..preprocessing.normalize import NormalizedText, TextNormalizer

_IDX = {n: i for i, n in enumerate(CUE_NAMES)}


def sarcasm_score(nt: NormalizedText) -> float:
    v = cue_vector(nt)
    s = (1.0 * v[_IDX["polarity_contrast"]] + 0.6 * (v[_IDX["n_markers"]] > 0) + 0.8 * v[_IDX["hashtag_sarcasm"]]
         + 0.4 * (v[_IDX["n_laugh"]] > 0) * (v[_IDX["n_pos_words"]] > 0) + 0.3 * (nt.stats.get("n_emoji_sarc", 0) > 0)
         + 0.2 * (v[_IDX["n_quotes"]] > 0))
    return float(s)


def weak_sarcasm(nt: NormalizedText, threshold: float = 0.8) -> str:
    return SARCASM_LABELS[int(sarcasm_score(nt) >= threshold)]


def weak_severity(nt: NormalizedText) -> str:
    low = nt.raw.lower()
    if any(p in low for p in THREAT_PHRASES if " " in p) or any(w in {"kill", "murder"} for w in low.split()):
        return SEVERITY_LABELS[2]
    n_abuse = sum(w.lower() in ABUSE for w in nt.words)
    if n_abuse >= 2 or (n_abuse >= 1 and nt.stats.get("n_exclaim", 0) >= 2):
        return SEVERITY_LABELS[1]
    return SEVERITY_LABELS[0]


def apply_weak_labels(df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    norm = TextNormalizer()
    df = df.copy()
    nts = [norm(t) for t in df["text"]]
    if "sarcasm" in tasks:
        df["sarcasm"] = [weak_sarcasm(n) for n in nts]
    if "severity" in tasks:
        sev = [weak_severity(n) for n in nts]
        df["severity"] = [s if h == HATE_LABELS[1] else np.nan for s, h in zip(sev, df["hate"])]
    return df
