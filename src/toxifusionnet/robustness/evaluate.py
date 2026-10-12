"""Robustness evaluation: re-score the test set under each perturbation and report metric drops + prediction flip rates."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..labels import IGNORE_INDEX, label_to_id
from .perturbations import PERTURBATIONS


def evaluate_robustness(pipeline, test_df: pd.DataFrame, seed: int = 0, perturbations=None) -> pd.DataFrame:
    perturbations = perturbations or PERTURBATIONS
    y = np.array([label_to_id("hate", v) for v in test_df["hate"]])
    sarc = np.array([label_to_id("sarcasm", v) for v in test_df["sarcasm"]])
    keep = y != IGNORE_INDEX
    texts = test_df["text"].tolist()
    base = (pipeline.hate_probability(texts) >= pipeline.hate_threshold).astype(int)

    def summarize(name, pred):
        out = {"perturbation": name, "hate_macro_f1": f1_score(y[keep], pred[keep], average="macro", zero_division=0),
               "flip_rate": float((pred != base).mean())}
        m = keep & (y == 1) & (sarc == 1)
        out["FNR_sarcastic_hate"] = float((pred[m] == 0).mean()) if m.any() else np.nan
        m2 = keep & (y == 0)
        out["FPR_non_hate"] = float((pred[m2] == 1).mean()) if m2.any() else np.nan
        if "narrative" in test_df and test_df["narrative"].notna().any():
            nm = keep & (test_df["narrative"].fillna(0).astype(int).to_numpy() == 1)
            out["hate_macro_f1_narrative"] = f1_score(y[nm], pred[nm], average="macro", zero_division=0) if nm.any() else np.nan
        return out

    rows = [summarize("none (original)", base)]
    for name, fn in perturbations.items():
        rng = random.Random(seed)
        pert = [fn(t, rng) for t in texts]
        pred = (pipeline.hate_probability(pert) >= pipeline.hate_threshold).astype(int)
        rows.append(summarize(name, pred))
    df = pd.DataFrame(rows)
    df["delta_macro_f1"] = df["hate_macro_f1"] - df.loc[0, "hate_macro_f1"]
    return df
