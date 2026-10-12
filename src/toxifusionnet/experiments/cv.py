"""Stratified k-fold cross-validation with POOLED out-of-fold predictions.

Why: rare classes (e.g. 81 organization, 68 society, 30 high-severity hate posts) have only a handful of examples in a
15% test split, so single-split per-class scores are very noisy. Pooling the k test folds gives every labelled post
exactly one prediction from a model that never saw it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..config import load_config
from ..data.grouped_splits import audit_fold_integrity, make_grouped_cv_splits
from ..data.schema import load_processed_csv
from ..training.metrics import evaluate_predictions, flatten_metrics
from ..training.stats import bootstrap_ci
from ..training.trainer import train_or_resume


def _strata(df: pd.DataFrame) -> pd.Series:
    return df["hate"].fillna("na").astype(str) + "|" + df["sarcasm"].fillna("na").astype(str)


def make_fold_splits(df: pd.DataFrame, k: int, seed: int, val_frac: float = 0.15):
    """Yield (fold_index, dataframe with a ``split`` column) for k stratified folds (test = fold, val carved from the rest)."""
    df = df.reset_index(drop=True)
    strata = _strata(df)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (rest, te) in enumerate(skf.split(df, strata)):
        tr, va = train_test_split(rest, test_size=val_frac, random_state=seed + f, stratify=strata.iloc[rest])
        d = df.copy()
        d["split"] = "train"
        d.loc[va, "split"] = "val"
        d.loc[te, "split"] = "test"
        yield f, d


def run_cv(config: str, overrides: list[str], data_path: str, out_dir: str | Path, folds: int = 5, seed: int = 42,
           include_tfidf: bool = True, grouped: bool = True, folds_cache: str | Path | None = None) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_processed_csv(data_path)

    if grouped:
        if folds_cache and Path(folds_cache).exists():
            fold_assign = pd.read_csv(folds_cache)
            # Reconstruct fold_dfs from cached assignment
            fold_dfs = []
            for f in range(folds):
                d = df.drop(columns=["split"], errors="ignore").merge(
                    fold_assign[fold_assign["fold"] == f][["id", "split"]], on="id"
                )
                d["fold"] = f
                fold_dfs.append(d)
        else:
            _, fold_dfs = make_grouped_cv_splits(df, n_splits=folds, seed=seed)
            if folds_cache:
                Path(folds_cache).parent.mkdir(parents=True, exist_ok=True)
                cached = pd.concat([d[["id", "split", "fold", "group_id"]] for d in fold_dfs])
                cached.to_csv(folds_cache, index=False)
        fold_gen = enumerate(fold_dfs)
    else:
        fold_gen = make_fold_splits(df, folds, seed)

    pooled, tf_pooled = [], []
    for f, d in fold_gen:
        cfg = load_config(config, overrides + [f"seed={seed + f}"])
        res = train_or_resume(cfg, d, out_dir / f"fold{f}")
        p = pd.read_csv(out_dir / f"fold{f}" / "preds_test.csv"); p["fold"] = f
        pooled.append(p)
        if include_tfidf:
            from ..baselines.classical import TfidfLogReg, with_gold
            te = d[d.split == "test"].reset_index(drop=True)
            tf = TfidfLogReg().fit(d[d.split == "train"])
            q = with_gold(te, tf.predict(te)); q["fold"] = f
            tf_pooled.append(q)
    out = {}
    for name, parts in (("model", pooled), ("tfidf_lr", tf_pooled)):
        if not parts:
            continue
        P = pd.concat(parts, ignore_index=True)
        P.to_csv(out_dir / f"pooled_{name}.csv", index=False)
        tasks = [t for t in ("sarcasm", "hate", "target", "severity") if f"pred_{t}" in P]
        res = evaluate_predictions(P, tasks)
        per_fold = [flatten_metrics(evaluate_predictions(g, tasks)) for _, g in P.groupby("fold")]

        # Compute 95% bootstrap confidence intervals for primary task Macro-F1s
        cis = {}
        for t in tasks:
            if f"gold_{t}" in P and f"pred_{t}" in P:
                valid = (P[f"gold_{t}"] != -100) & (P[f"gold_{t}"].notna())
                if valid.any():
                    cis[f"{t}_macro_f1"] = bootstrap_ci(P.loc[valid, f"gold_{t}"], P.loc[valid, f"pred_{t}"])

        out[name] = {
            "pooled": res,
            "pooled_flat": flatten_metrics(res),
            "bootstrap_ci95": cis,
            "per_fold_hate_macro_f1": [float(x.get("hate_macro_f1", np.nan)) for x in per_fold]
        }
    (out_dir / "cv_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    return out
