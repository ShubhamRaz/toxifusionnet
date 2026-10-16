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
from ..data.grouped_splits import assign_group_ids, audit_fold_integrity, make_grouped_cv_splits, normalize_ocr_text
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
CANONICAL_COMBINED_FOLD_CACHE_SHA256 = "fd00161698dbfdc16381513b369b5d74d294dccf359f9cd937a70f33e7aa8ba7"

# Curated, leak-free fold caches produced by scripts/build_cleaned_pipeline.py
CURATED_FOLD_CACHE_SHA256 = "2f3002ab81b8b03c26228edc88f9cc04613dad2e73d1b62c4869984d6febffa8"
CURATED_COMBINED_FOLD_CACHE_SHA256 = "78d825723147f1e8a556427f50e72c916fae79245199fd457c789f19d338f26a"

ALLOWED_FOLD_CACHE_SHAS = {CANONICAL_FOLD_CACHE_SHA256, CURATED_FOLD_CACHE_SHA256}
ALLOWED_COMBINED_FOLD_CACHE_SHAS = {CANONICAL_COMBINED_FOLD_CACHE_SHA256, CURATED_COMBINED_FOLD_CACHE_SHA256}


def compute_normalized_fold_cache_sha256(cache_path: str | Path) -> str:
    """Compute the SHA-256 checksum of fold cache with in-memory CRLF line-ending normalization.

    Ensures consistent, cross-platform checksum verification across Linux (LF) and
    Windows (CRLF) environments without modifying the file on disk.
    """
    import hashlib
    raw_bytes = Path(cache_path).read_bytes()
    normalized = raw_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(normalized).hexdigest()


def validate_fold_cache_checksum(cache_path: str | Path, expected_sha: str | set[str] = ALLOWED_FOLD_CACHE_SHAS) -> str:
    """Verify that the fold cache matches the expected SHA-256 after in-memory CRLF normalization.

    Raises ValueError immediately if the checksum does not match expected_sha, preventing corrupted
    or modified fold mappings from compromising experimental reproducibility.
    """
    sha = compute_normalized_fold_cache_sha256(cache_path)
    if isinstance(expected_sha, (set, list, tuple)):
        if sha not in expected_sha:
            raise ValueError(
                f"Canonical fold cache checksum verification failed for '{cache_path}'! "
                f"Expected one of SHA-256 (CRLF-normalized): {expected_sha}, got: {sha}. "
                "The fold cache has been altered or corrupted. Automatic regeneration is disabled."
            )
    else:
        if sha != expected_sha:
            raise ValueError(
                f"Canonical fold cache checksum verification failed for '{cache_path}'! "
                f"Expected SHA-256 (CRLF-normalized): {expected_sha}, got: {sha}. "
                "The fold cache has been altered or corrupted. Automatic regeneration is disabled."
            )
    return sha


def validate_fold_cache(
    fold_assign: pd.DataFrame,
    expected_ids: set | None = None,
    folds: int = 5,
    id_col: str | list[str] | None = None,
) -> dict[int, list]:
    """Strict pre-training validation of fold cache integrity and test partition coverage."""
    required_cols = {"id", "split", "fold"}
    if not required_cols.issubset(fold_assign.columns):
        raise ValueError(f"Fold cache missing required columns: {required_cols - set(fold_assign.columns)}")

    actual_folds = set(fold_assign["fold"].unique())
    if actual_folds != set(range(folds)):
        raise ValueError(f"Fold cache must contain folds 0..{folds-1}, got {actual_folds}")

    # Determine unique identifier: use composite (source_dataset, id) only when explicitly
    # requested via id_col or when the cache itself shows within-fold id collisions.
    # NOTE: Do NOT sniff expected_ids content — that is nondeterministic for large sets.
    use_composite = False
    if "source_dataset" in fold_assign.columns:
        if id_col == ["source_dataset", "id"] or fold_assign.groupby("fold")["id"].apply(lambda s: s.duplicated().any()).any():
            use_composite = True

    def _get_key_series(df_part: pd.DataFrame) -> pd.Series:
        if use_composite:
            return df_part["source_dataset"].astype(str) + "::" + df_part["id"].astype(str)
        return df_part["id"].astype(str)

    test_ids_by_fold = {}
    all_test_keys = []
    expected_set = {str(k) for k in expected_ids} if expected_ids is not None else None

    for f in range(folds):
        f_df = fold_assign[fold_assign["fold"] == f]
        f_keys = _get_key_series(f_df)

        if f_keys.duplicated().any():
            dups = f_keys[f_keys.duplicated()].tolist()[:5]
            raise ValueError(f"Fold cache integrity violation in fold {f}: duplicate IDs within fold! Examples: {dups}")

        invalid_splits = set(f_df["split"].unique()) - {"train", "val", "test"}
        if invalid_splits:
            raise ValueError(f"Fold {f} contains invalid split values: {invalid_splits}")

        tr = set(_get_key_series(f_df[f_df["split"] == "train"]))
        va = set(_get_key_series(f_df[f_df["split"] == "val"]))
        te = set(_get_key_series(f_df[f_df["split"] == "test"]))

        if len(tr.intersection(va)) > 0 or len(tr.intersection(te)) > 0 or len(va.intersection(te)) > 0:
            raise ValueError(f"Fold cache integrity violation in fold {f}: train/val/test splits overlap!")

        if expected_set is not None:
            f_key_set = set(f_keys)
            if f_key_set != expected_set:
                missing_f = expected_set - f_key_set
                unexpected_f = f_key_set - expected_set
                err_parts = []
                if missing_f:
                    err_parts.append(f"missing {len(missing_f)} IDs (e.g. {list(missing_f)[:5]})")
                if unexpected_f:
                    err_parts.append(f"unexpected {len(unexpected_f)} IDs (e.g. {list(unexpected_f)[:5]})")
                raise ValueError(f"Fold {f} ID mismatch: {', '.join(err_parts)}")

        test_ids_by_fold[f] = sorted(list(_get_key_series(f_df[f_df["split"] == "test"])))
        all_test_keys.extend(te)

    # Verify exactly one outer-test assignment per sample
    if len(all_test_keys) != len(set(all_test_keys)):
        from collections import Counter
        counts = Counter(all_test_keys)
        dups = [item for item, c in counts.items() if c > 1][:5]
        raise ValueError(f"Duplicate test IDs detected across folds in fold cache! Examples: {dups}")

    if expected_set is not None:
        missing = expected_set - set(all_test_keys)
        if missing:
            raise ValueError(f"Fold cache missing {len(missing)} dataset IDs! (examples: {list(missing)[:5]})")
        unexpected = set(all_test_keys) - expected_set
        if unexpected:
            raise ValueError(f"Fold cache contains {len(unexpected)} unexpected IDs not in dataset! (examples: {list(unexpected)[:5]})")

    if "source_dataset" in fold_assign.columns:
        if fold_assign.duplicated(subset=["fold", "source_dataset", "id"]).any():
            raise ValueError("Duplicate (source_dataset, id) detected within a fold in fold cache!")

    return test_ids_by_fold


def _evaluate_cv_slice(P_slice: pd.DataFrame, tasks: list[str]) -> dict[str, Any]:
    """Compute complete pooled and per-fold metrics for a dataset slice."""
    res = evaluate_predictions(P_slice, tasks)
    flat = flatten_metrics(res)
    per_fold = [flatten_metrics(evaluate_predictions(g, tasks)) for _, g in P_slice.groupby("fold")]

    cis, group_cis = {}, {}
    groups = normalize_ocr_text(P_slice["text"]) if "text" in P_slice else pd.Series(np.arange(len(P_slice)).astype(str), index=P_slice.index)
    for t in tasks:
        if f"gold_{t}" in P_slice and f"pred_{t}" in P_slice:
            valid = (P_slice[f"gold_{t}"] != -100) & (P_slice[f"gold_{t}"].notna())
            if valid.any():
                cis[f"{t}_macro_f1"] = bootstrap_ci(P_slice.loc[valid, f"gold_{t}"], P_slice.loc[valid, f"pred_{t}"])
                group_cis[f"{t}_macro_f1"] = group_bootstrap_ci(groups.loc[valid], P_slice.loc[valid, f"gold_{t}"], P_slice.loc[valid, f"pred_{t}"])

    fold_stats = {}
    for metric_key in ["mean_macro_f1"] + [f"{t}_macro_f1" for t in tasks]:
        vals = [float(x.get(metric_key, np.nan)) for x in per_fold if metric_key in x and not np.isnan(x.get(metric_key, np.nan))]
        if vals:
            fold_stats[metric_key] = {
                "values": vals,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

    return {
        "pooled": res,
        "pooled_flat": flat,
        "bootstrap_ci95": cis,
        "group_bootstrap_ci95": group_cis,
        "fold_stats": fold_stats,
        "per_fold": per_fold,
    }


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
            if cache_p.name == "multibully_v2_cv_folds.csv":
                validate_fold_cache_checksum(cache_p, ALLOWED_FOLD_CACHE_SHAS)
            elif cache_p.name == "combined_multibully_hot_cv_folds.csv":
                validate_fold_cache_checksum(cache_p, ALLOWED_COMBINED_FOLD_CACHE_SHAS)
            fold_assign = pd.read_csv(cache_p)
            use_comp = not df["id"].is_unique and "source_dataset" in df.columns
            expected_keys = (
                set(df["source_dataset"].astype(str) + "::" + df["id"].astype(str))
                if use_comp
                else set(df["id"].astype(str))
            )
            id_col = ["source_dataset", "id"] if use_comp else "id"
            validate_fold_cache(fold_assign, expected_ids=expected_keys, folds=folds, id_col=id_col)
            # Reconstruct fold_dfs from cached assignment while strictly preserving group_id and all metadata
            fold_dfs = []
            if use_comp:
                on_cols = ["source_dataset", "id"]
                merge_cols = ["source_dataset", "id", "split"]
            else:
                on_cols = ["id"]
                merge_cols = ["id", "split"]

            if "group_id" in fold_assign.columns and "group_id" not in merge_cols:
                merge_cols.append("group_id")

            # Avoid suffix collisions by dropping cached assignment columns (except join keys) from df prior to merge
            cols_to_drop_from_df = [c for c in ["split", "group_id"] if c in df.columns and c not in on_cols]

            for f in range(folds):
                f_assign = fold_assign[fold_assign["fold"] == f][merge_cols]
                base_df = df.drop(columns=cols_to_drop_from_df, errors="ignore")
                # Left merge with indicator keeps every original row visible so we can
                # detect two independent failure modes:
                #   (a) validate="one_to_one"  → raises MergeError if either side has
                #       duplicate keys on on_cols (catches row multiplication immediately)
                #   (b) _merge != "both"       → catches rows silently dropped by an
                #       inner join when the fold cache is missing an assignment
                d = base_df.merge(
                    f_assign, on=on_cols, how="left",
                    indicator=True, validate="one_to_one",
                )
                unmatched = d[d["_merge"] != "both"]
                if not unmatched.empty:
                    examples = unmatched[on_cols].head(5).to_dict("records")
                    raise ValueError(
                        f"Fold {f} cache reconstruction: {len(unmatched)} of {len(df)} "
                        f"original rows had no matching fold assignment. "
                        f"Example unmatched keys: {examples}"
                    )
                d = d.drop(columns=["_merge"])
                if len(d) != len(df):
                    raise ValueError(
                        f"Fold {f} cache reconstruction row count mismatch: got {len(d)}, expected {len(df)}"
                    )

                if "group_id" not in d.columns:
                    d = assign_group_ids(d)
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
                cache_cols = ["id", "split", "fold", "group_id"]
                if "source_dataset" in fold_dfs[0].columns:
                    cache_cols.insert(1, "source_dataset")
                cached = pd.concat([d[[c for c in cache_cols if c in d.columns]] for d in fold_dfs], ignore_index=True)
                cached.to_csv(folds_cache, index=False)
                log.info("Saved newly generated fold cache to %s", folds_cache)
        fold_gen = enumerate(fold_dfs)
    else:
        fold_gen = make_fold_splits(df, folds, seed)
        fold_dfs = [d for _, d in fold_gen]
        fold_gen = enumerate(fold_dfs)

    # Pre-training integrity audit: stop execution if validation fails
    if grouped:
        use_comp = not df["id"].is_unique and "source_dataset" in df.columns
        id_col = ["source_dataset", "id"] if use_comp else "id"
        audit = audit_fold_integrity(fold_dfs, len(df), id_col=id_col)
        if not audit["passed"]:
            raise ValueError(f"Pre-training fold integrity audit failed! Fold details: {audit}")
        log.info("Pre-training fold integrity audit passed across all %d folds.", folds)

    pooled, tf_pooled = [], []
    fold_class_weights = {}
    fold_training_meta = {}
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
        fold_training_meta[f"fold_{f}"] = {
            k: res[k] for k in (
                "best_epoch", "best_step", "best_val_score", "total_steps",
                "successful_updates", "attempted_batches", "skipped_updates",
                "steps_budget", "eval_checkpoint", "eval_checkpoint_type"
            ) if k in res
        }
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

        # Slices for overall and source-specific metrics
        slices = {"overall": P}
        if "source_dataset" in P.columns and P["source_dataset"].nunique() > 1:
            for src in sorted(P["source_dataset"].dropna().unique()):
                sub = P[P["source_dataset"] == src].reset_index(drop=True)
                if len(sub) > 0:
                    slices[str(src)] = sub

        slice_metrics = {sname: _evaluate_cv_slice(sdf, tasks) for sname, sdf in slices.items()}
        overall_m = slice_metrics["overall"]

        out[name] = {
            "pooled": overall_m["pooled"],
            "pooled_flat": overall_m["pooled_flat"],
            "bootstrap_ci95": overall_m["bootstrap_ci95"],
            "group_bootstrap_ci95": overall_m["group_bootstrap_ci95"],
            "fold_stats": overall_m["fold_stats"],
            "class_weights_by_fold": fold_class_weights if name == "model" else {},
            "training_meta_by_fold": fold_training_meta if name == "model" else {},
            "per_fold_hate_macro_f1": [float(x.get("hate_macro_f1", np.nan)) for x in overall_m["per_fold"]],
            "per_fold_mean_macro_f1": [float(x.get("mean_macro_f1", np.nan)) for x in overall_m["per_fold"]],
            "per_fold_sarcasm_macro_f1": [float(x.get("sarcasm_macro_f1", np.nan)) for x in overall_m["per_fold"]],
            "per_fold_target_macro_f1": [float(x.get("target_macro_f1", np.nan)) for x in overall_m["per_fold"]],
            "per_fold_severity_macro_f1": [float(x.get("severity_macro_f1", np.nan)) for x in overall_m["per_fold"]],
            "per_fold_flat": overall_m["per_fold"],
            "sources": {k: v for k, v in slice_metrics.items() if k != "overall"},
            "slices": slice_metrics,
        }
    (out_dir / "cv_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    return out
