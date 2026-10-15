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


def _normalize_label_val(val: Any) -> str:
    """Normalize label values, mapping NaNs, None, and whitespace strings consistently to 'none'."""
    if pd.isna(val):
        return "none"
    s = str(val).strip()
    return s if s and s.lower() != "nan" else "none"


def _sample_stratum(row) -> tuple[str, str]:
    """Return (fine_stratum, base_stratum) for a sample with normalized labels."""
    h = _normalize_label_val(row.get("hate"))
    s = _normalize_label_val(row.get("sarcasm"))
    t = _normalize_label_val(row.get("target"))
    v = _normalize_label_val(row.get("severity"))

    if v == "high":
        return f"SEV_HIGH_{s}", "SEV_HIGH"
    if t in ("society", "organization"):
        return f"TGT_{t}_{s}", f"TGT_{t}"
    if v == "low":
        return f"SEV_LOW_{s}", "SEV_LOW"
    if h == "hate":
        if t in ("community", "individual"):
            return f"HATE_{t}_{s}", f"HATE_{t}"
        return f"HATE_individual_{s}", "HATE_individual"
    return f"NONHATE_{s}", "NONHATE"


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


def aggregate_rare_group_strata(
    group_fine: dict[int, str],
    group_base: dict[int, str],
    min_groups: int = 10,
    n_splits: int = 5,
) -> dict[int, str]:
    """Deterministically aggregate strata with fewer than min_groups members at the group level.

    Guarantees that:
    1. Every stratum passed to outer StratifiedKFold has at least n_splits members.
    2. Every stratum in the inner training pool has at least 2 members for train_test_split.
    """
    from collections import Counter
    counts = Counter(group_fine.values())
    aggregated: dict[int, str] = {}

    for gid, st in group_fine.items():
        if counts[st] < min_groups:
            base = group_base[gid]
            # Sibling strata sharing the same base prefix with sufficient support
            siblings = [s for s, c in counts.items() if s.startswith(base) and c >= min_groups]
            if siblings:
                aggregated[gid] = max(siblings, key=lambda s: counts[s])
            else:
                aggregated[gid] = base
        else:
            aggregated[gid] = st

    # Secondary check: if any aggregated stratum still has < min_groups
    counts2 = Counter(aggregated.values())
    final: dict[int, str] = {}
    for gid, st in aggregated.items():
        if counts2[st] < min_groups:
            final[gid] = "RARE_STRATUM"
        else:
            final[gid] = st

    # Final guard: ensure no stratum has fewer members than n_splits
    counts3 = Counter(final.values())
    if "RARE_STRATUM" in counts3 and counts3["RARE_STRATUM"] < n_splits:
        majority = max(counts3.keys(), key=lambda k: counts3[k] if k != "RARE_STRATUM" else -1)
        for gid, st in list(final.items()):
            if st == "RARE_STRATUM":
                final[gid] = majority

    return final


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
    strata_res = df.apply(_sample_stratum, axis=1)
    df["sample_stratum"] = [x[0] for x in strata_res]
    df["base_stratum"] = [x[1] for x in strata_res]

    # For each group, pick the highest priority stratum among its constituent samples
    group_fine: Dict[int, str] = {}
    group_base: Dict[int, str] = {}
    for gid, group_rows in df.groupby("group_id"):
        sorted_pairs = sorted(
            zip(group_rows["sample_stratum"], group_rows["base_stratum"]),
            key=lambda pair: _get_priority(pair[0])
        )
        group_fine[gid], group_base[gid] = sorted_pairs[0]

    min_g = max(2 * n_splits, 10)
    final_strata = aggregate_rare_group_strata(group_fine, group_base, min_groups=min_g, n_splits=n_splits)

    group_df = pd.DataFrame({
        "group_id": list(final_strata.keys()),
        "stratum": list(final_strata.values())
    })

    min_groups_needed = n_splits * 2
    if len(group_df) < min_groups_needed:
        raise ValueError(
            f"Cannot construct valid {n_splits}-fold splits with inner validation: "
            f"dataset contains only {len(group_df)} unique groups, but at least "
            f"{min_groups_needed} groups are required to populate train, val, and test partitions in every fold."
        )

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
        strat_series = f_group_df.loc[rest_idx, "stratum"]
        strat_counts = strat_series.value_counts()
        can_stratify = (strat_counts >= 2).all() and len(strat_counts) > 1

        tr_groups, val_groups = train_test_split(
            rest_idx,
            test_size=val_frac,
            random_state=seed + f,
            stratify=strat_series if can_stratify else None
        )
        if len(tr_groups) == 0 or len(val_groups) == 0:
            raise ValueError(
                f"Fold {f}: inner split resulted in an empty partition (train={len(tr_groups)}, val={len(val_groups)}). "
                f"Dataset with {len(group_df)} groups is too small for val_frac={val_frac}."
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
            if len(sub) == 0:
                checks["passed"] = False
            high_sev = int((sub["severity"] == "high").sum()) if "severity" in sub else 0
            soc = int((sub["target"] == "society").sum()) if "target" in sub else 0
            org = int((sub["target"] == "organization").sum()) if "target" in sub else 0
            hate = int((sub["hate"] == "hate").sum()) if "hate" in sub else 0
            sarc = int((sub["sarcasm"] == "sarcastic").sum()) if "sarcasm" in sub else 0

            # Val zero-support check (only when dataset contains high severity instances)
            total_high_sev = int((d["severity"] == "high").sum()) if "severity" in d else 0
            if part == "val" and high_sev == 0 and total_high_sev > 0:
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
