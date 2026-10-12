"""Stratified train/val/test splitting."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def _strat_key(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    key = pd.Series("", index=df.index, dtype=object)
    for i, c in enumerate(cols):
        key = key + ("|" if i else "") + df[c].fillna("na").astype(str)
    return key


def make_splits(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, seed: int = 42,
                stratify_cols: tuple[str, ...] = ("hate", "target")) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    cols = tuple(c for c in stratify_cols if c in df.columns and df[c].notna().any())
    key = _strat_key(df, cols) if cols else pd.Series(["all"] * len(df))
    vc = key.value_counts()
    if (vc < 3).any():  # rare strata -> fall back to the first column only
        key = _strat_key(df, cols[:1]) if cols else key
        if (key.value_counts() < 3).any():
            key = None
    idx = df.index.to_numpy()
    tr_val, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=None if key is None else key.loc[idx])
    rel = val_size / (1 - test_size)
    tr, va = train_test_split(tr_val, test_size=rel, random_state=seed, stratify=None if key is None else key.loc[tr_val])
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    return df
