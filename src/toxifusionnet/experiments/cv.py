"""Stratified k-fold cross-validation with POOLED out-of-fold predictions.

Why: rare classes (e.g. 81 organization, 68 society, 30 high-severity hate posts) have only a handful of examples in a
15% test split, so single-split per-class scores are very noisy. Pooling the k test folds gives every labelled post
exactly one prediction from a model that never saw it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

log = logging.getLogger("toxifusionnet")

from ..config import load_config
from ..data.grouped_splits import audit_fold_integrity, make_grouped_cv_splits, normalize_ocr_text
from ..data.schema import load_processed_csv
from ..training.metrics import evaluate_predictions, flatten_metrics
from ..training.stats import bootstrap_ci, group_bootstrap_ci
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


CANONICAL_FOLD_CACHE_SHA256 = "73d4dbb4858907f43765dea0acff36aa4ae21ddc01e1694c0fcf85791fb2d1a5"


def compute_normalized_fold_cache_sha256(cache_path: str | Path) -> str:
    """Compute the SHA-256 checksum of fold cache with in-memory CRLF line-ending normalization.

    Ensures consistent, cross-platform checksum verification across Linux (LF) and
    Windows (CRLF) environments without modifying the file on disk.
    """
    import hashlib
    raw_bytes = Path(cache_path).read_bytes()
    normalized = raw_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(normalized).hexdigest()


def validate_fold_cache(fold_assign: pd.DataFrame, expected_ids: set | None = None, folds: int = 5) -> dict[int, list]:
    """Strict pre-training validation of fold cache integrity and test partition coverage."""
    actual_folds = set(fold_assign["fold"].unique())
    if actual_folds != set(range(folds)):
        raise ValueError(f"Fold cache must contain folds 0..{folds-1}, got {actual_folds}")
    test_ids_by_fold = {}
    all_test_ids = []
    for f in range(folds):
        f_df = fold_assign[fold_assign["fold"] == f]
        tr = set(f_df[f_df["split"] == "train"]["id"])
        va = set(f_df[f_df["split"] == "val"]["id"])
        te = set(f_df[f_df["split"] == "test"]["id"])
        if len(tr.intersection(va)) > 0 or len(tr.intersection(te)) > 0 or len(va.intersection(te)) > 0:
            raise ValueError(f"Fold cache integrity violation in fold {f}: train/val/test splits overlap!")
        test_ids_by_fold[f] = sorted(list(te))
        all_test_ids.extend(te)
    if len(all_test_ids) != len(set(all_test_ids)):
        raise ValueError("Duplicate test IDs detected across folds in fold cache!")
    if expected_ids:
        missing = set(expected_ids) - set(all_test_ids)
        if missing:
            raise ValueError(f"Fold cache missing {len(missing)} dataset IDs! (examples: {list(missing)[:5]})")
    return test_ids_by_fold


def run_cv(config: str, overrides: list[str], data_path: str, out_dir: str | Path, folds: int = 5, seed: int = 42,
           include_tfidf: bool = True, grouped: bool = True, folds_cache: str | Path | None = None,
           aux_data_path: str | Path | None = None) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_processed_csv(data_path)
    if "source_dataset" not in df:
        df["source_dataset"] = "multibully"

    # Check auxiliary data path from parameter or preview config
    if not aux_data_path:
        try:
            cfg_prev = load_config(config, overrides)
            aux_data_path = cfg_prev.data.get("aux_path")
        except Exception:
            pass

    aux_df = None
    if aux_data_path and Path(aux_data_path).exists():
        raw_aux = pd.read_csv(aux_data_path)
        if "source_dataset" not in raw_aux:
            raw_aux["source_dataset"] = "aux"

        # Exclude auxiliary rows whose labels across all tasks are completely masked (zero supervision)
        from ..labels import TASKS, label_to_id, IGNORE_INDEX
        sup_mask = pd.Series(False, index=raw_aux.index)
        for t in TASKS:
            if t in raw_aux.columns:
                sup_mask = sup_mask | raw_aux[t].apply(lambda v: label_to_id(t, v) != IGNORE_INDEX)

        n_raw = len(raw_aux)
        aux_df = raw_aux[sup_mask].reset_index(drop=True)
        n_excluded = n_raw - len(aux_df)
        log.info(
            "Auxiliary dataset '%s': %d raw rows, retained %d supervised rows, excluded %d zero-supervision rows (all labels masked)",
            aux_data_path, n_raw, len(aux_df), n_excluded
        )

    if grouped:
        if folds_cache and Path(folds_cache).exists():
            cache_p = Path(folds_cache)
            sha = compute_normalized_fold_cache_sha256(cache_p)
            log.info("Using existing canonical fold cache: %s (SHA-256 [CRLF-normalized]: %s)", folds_cache, sha)
            if cache_p.name == "multibully_v2_cv_folds.csv" and sha != CANONICAL_FOLD_CACHE_SHA256:
                log.warning(
                    "Fold cache '%s' normalized SHA-256 (%s) differs from canonical (%s)",
                    folds_cache, sha, CANONICAL_FOLD_CACHE_SHA256
                )
            fold_assign = pd.read_csv(cache_p)
            validate_fold_cache(fold_assign, expected_ids=set(df["id"]), folds=folds)
            # Reconstruct fold_dfs from cached assignment
            fold_dfs = []
            for f in range(folds):
                d = df.drop(columns=["split"], errors="ignore").merge(
                    fold_assign[fold_assign["fold"] == f][["id", "split"]], on="id"
                )
                d["fold"] = f
                if aux_df is not None:
                    aux_copy = aux_df.copy()
                    aux_copy["split"] = "train"   # Auxiliary data strictly in train partition
                    aux_copy["fold"] = f
                    d = pd.concat([d, aux_copy], ignore_index=True)
                fold_dfs.append(d)
        else:
            _, fold_dfs = make_grouped_cv_splits(df, n_splits=folds, seed=seed)
            if aux_df is not None:
                for f, d in enumerate(fold_dfs):
                    aux_copy = aux_df.copy()
                    aux_copy["split"] = "train"
                    aux_copy["fold"] = f
                    fold_dfs[f] = pd.concat([d, aux_copy], ignore_index=True)
            if folds_cache:
                Path(folds_cache).parent.mkdir(parents=True, exist_ok=True)
                cached = pd.concat([d[["id", "split", "fold", "group_id"]] for d in fold_dfs])
                cached.to_csv(folds_cache, index=False)
                log.info("Saved newly generated fold cache to %s", folds_cache)
        fold_gen = enumerate(fold_dfs)
    else:
        fold_gen = make_fold_splits(df, folds, seed)

    pooled, tf_pooled = [], []
    fold_class_weights = {}
    for f, d in fold_gen:
        cfg = load_config(config, overrides + [f"seed={seed + f}"])
        res = train_or_resume(cfg, d, out_dir / f"fold{f}")
        if "class_balancing" in res:
            fold_class_weights[f"fold_{f}"] = res["class_balancing"]
        elif (out_dir / f"fold{f}" / "class_weights.json").exists():
            try:
                fold_class_weights[f"fold_{f}"] = json.loads((out_dir / f"fold{f}" / "class_weights.json").read_text())
            except Exception:
                pass
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
        cis, group_cis = {}, {}
        groups = normalize_ocr_text(P["text"]) if "text" in P else pd.Series(np.arange(len(P)).astype(str), index=P.index)
        for t in tasks:
            if f"gold_{t}" in P and f"pred_{t}" in P:
                valid = (P[f"gold_{t}"] != -100) & (P[f"gold_{t}"].notna())
                if valid.any():
                    cis[f"{t}_macro_f1"] = bootstrap_ci(P.loc[valid, f"gold_{t}"], P.loc[valid, f"pred_{t}"])
                    group_cis[f"{t}_macro_f1"] = group_bootstrap_ci(groups.loc[valid], P.loc[valid, f"gold_{t}"], P.loc[valid, f"pred_{t}"])

        out[name] = {
            "pooled": res,
            "pooled_flat": flatten_metrics(res),
            "bootstrap_ci95": cis,
            "group_bootstrap_ci95": group_cis,
            "class_weights_by_fold": fold_class_weights if name == "model" else {},
            "per_fold_hate_macro_f1": [float(x.get("hate_macro_f1", np.nan)) for x in per_fold],
            "per_fold_mean_macro_f1": [float(x.get("mean_macro_f1", np.nan)) for x in per_fold],
            "per_fold_sarcasm_macro_f1": [float(x.get("sarcasm_macro_f1", np.nan)) for x in per_fold],
            "per_fold_target_macro_f1": [float(x.get("target_macro_f1", np.nan)) for x in per_fold],
            "per_fold_severity_macro_f1": [float(x.get("severity_macro_f1", np.nan)) for x in per_fold],
            "per_fold_flat": per_fold,
        }
    (out_dir / "cv_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    return out
