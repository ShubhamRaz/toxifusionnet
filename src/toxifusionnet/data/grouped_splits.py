"""Reproducible group-aware multi-task cross-validation splitting for MultiBully v2.

Guarantees:
1. Zero group leakage: Identical normalized OCR texts are grouped; all samples in a group
   remain in the same partition (train, val, or test).
2. Multi-task priority stratification: Stratifies jointly on (severity, target, hate, sarcasm)
   with explicit priority for ultra-rare classes (high severity, society, organization).
3. Strict outer-test isolation: Outer test fold is held out strictly for confirmatory evaluation;
   checkpoint selection and early stopping use a dedicated inner validation split carved only
   from the training pool.
4. Data-dependent isolation: Class weights and preprocessing fit only on the inner train split.
"""
from __future__ import annotations

import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

log = logging.getLogger("toxifusionnet")


def normalize_ocr_text(texts: pd.Series) -> pd.Series:
    """Normalize OCR text for strict near-duplicate grouping."""
    return texts.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def assign_group_ids(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Assign integer group IDs to identical normalized texts."""
    df = df.copy()
    norm = normalize_ocr_text(df[text_col])
    df["group_id"] = norm.factorize()[0]
    return df


def _sample_stratum(row) -> str:
    h = str(row.get("hate", "none"))
    s = str(row.get("sarcasm", "none"))
    t = str(row.get("target")) if pd.notna(row.get("target")) else "none"
    v = str(row.get("severity")) if pd.notna(row.get("severity")) else "none"

    if v == "high":
        return f"SEV_HIGH_{s}"
    if t in ("society", "organization"):
        return f"TGT_{t}_{s}"
    if v == "low":
        return f"SEV_LOW_{s}"
    if h == "hate":
        if t == "none":
            return f"HATE_individual_{s}"
        return f"HATE_{t}_{s}"
    return f"NONHATE_{s}"


# Priority order: rare classes must dominate the group-level stratum assignment
_STRATUM_PRIORITY = {
    "SEV_HIGH": 1,
    "TGT_society": 2,
    "TGT_organization": 3,
    "SEV_LOW": 4,
    "HATE_community": 5,
    "HATE_individual": 6,
    "NONHATE": 7,
}


def _get_priority(st: str) -> int:
    for prefix, prio in _STRATUM_PRIORITY.items():
        if st.startswith(prefix):
            return prio
    return 99


def make_grouped_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Build group-aware 5-fold cross-validation datasets.

    Args:
        df: Input processed dataset (must contain 'id', 'text', and task labels).
        n_splits: Number of outer cross-validation folds (default: 5).
        seed: Random seed for reproducible splitting.
        val_frac: Fraction of training groups reserved for inner validation (default: 0.15).

    Returns:
        group_df: Group-level summary containing group_id, stratum, and outer_fold.
        fold_dfs: List of 5 DataFrames, one per fold, with a 'split' column ('train', 'val', 'test').
    """
    df = assign_group_ids(df)
    df["sample_stratum"] = df.apply(_sample_stratum, axis=1)

    # For each group, pick the highest priority stratum among its constituent samples
    group_strata: Dict[int, str] = {}
    for gid, group_rows in df.groupby("group_id"):
        sorted_strata = sorted(group_rows["sample_stratum"].unique(), key=_get_priority)
        group_strata[gid] = sorted_strata[0]

    group_df = pd.DataFrame({
        "group_id": list(group_strata.keys()),
        "stratum": list(group_strata.values())
    })

    # Stratified K-Fold on the unique groups
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    group_df["outer_fold"] = -1
    for fold, (_, test_idx) in enumerate(skf.split(group_df, group_df["stratum"])):
        group_df.iloc[test_idx, group_df.columns.get_loc("outer_fold")] = fold

    # Clean old split column if it exists in df
    clean_df = df.drop(columns=["split"], errors="ignore")

    fold_dfs: List[pd.DataFrame] = []
    for f in range(n_splits):
        f_group_df = group_df.copy()
        test_mask = f_group_df["outer_fold"] == f
        rest_idx = f_group_df[~test_mask].index

        # Stratified split of remaining groups into inner train and inner val
        tr_groups, val_groups = train_test_split(
            rest_idx,
            test_size=val_frac,
            random_state=seed + f,
            stratify=f_group_df.loc[rest_idx, "stratum"]
        )

        f_group_df["split"] = "train"
        f_group_df.loc[val_groups, "split"] = "val"
        f_group_df.loc[test_mask, "split"] = "test"

        d = clean_df.merge(f_group_df[["group_id", "split"]], on="group_id")
        d["fold"] = f
        fold_dfs.append(d)

    return group_df, fold_dfs


def audit_fold_integrity(fold_dfs: List[pd.DataFrame], original_len: int) -> Dict[str, Any]:
    """Audit fold datasets for zero leakage, full coverage, and rare class presence."""
    checks: Dict[str, Any] = {"passed": True, "folds": []}
    all_test_ids: List[Any] = []

    for f, d in enumerate(fold_dfs):
        tr_g = set(d[d["split"] == "train"]["group_id"])
        va_g = set(d[d["split"] == "val"]["group_id"])
        te_g = set(d[d["split"] == "test"]["group_id"])

        leakage_tr_te = len(tr_g & te_g)
        leakage_va_te = len(va_g & te_g)
        leakage_tr_va = len(tr_g & va_g)

        if leakage_tr_te > 0 or leakage_va_te > 0 or leakage_tr_va > 0:
            checks["passed"] = False

        test_ids = list(d[d["split"] == "test"]["id"])
        all_test_ids.extend(test_ids)

        fold_stat: Dict[str, Any] = {
            "fold": f,
            "leakage_tr_te": leakage_tr_te,
            "leakage_va_te": leakage_va_te,
            "leakage_tr_va": leakage_tr_va,
            "counts": {}
        }

        for part in ["train", "val", "test"]:
            sub = d[d["split"] == part]
            high_sev = int((sub["severity"] == "high").sum()) if "severity" in sub else 0
            soc = int((sub["target"] == "society").sum()) if "target" in sub else 0
            org = int((sub["target"] == "organization").sum()) if "target" in sub else 0
            hate = int((sub["hate"] == "hate").sum()) if "hate" in sub else 0
            sarc = int((sub["sarcasm"] == "sarcastic").sum()) if "sarcasm" in sub else 0

            # Val zero-support check
            if part == "val" and high_sev == 0:
                checks["passed"] = False

            fold_stat["counts"][part] = {
                "total": len(sub),
                "hate": hate,
                "sarcasm": sarc,
                "high_severity": high_sev,
                "society": soc,
                "organization": org,
                "rare_warning": high_sev < 5 if part == "test" else False
            }

        checks["folds"].append(fold_stat)

    checks["total_test_evaluated"] = len(all_test_ids)
    checks["unique_test_evaluated"] = len(set(all_test_ids))
    if len(all_test_ids) != original_len or len(set(all_test_ids)) != original_len:
        checks["passed"] = False

    return checks
