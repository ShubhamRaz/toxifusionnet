"""Per-task dataset balancing and fold-specific class weighting utilities.

Provides square-root inverse-frequency class weighting, clipping, normalization,
and metadata tracking across multi-task heads (hate, sarcasm, target, severity).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..labels import IGNORE_INDEX, LABEL_SPACES, label_to_id

log = logging.getLogger("toxifusionnet")


def compute_task_class_weights(
    df: pd.DataFrame,
    task: str,
    mode: str = "sqrt_balanced",
    max_weight: float = 4.0,
    source_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute fold-specific class weights for a single prediction task.

    Formula for 'sqrt_balanced':
        weight_c = sqrt(N / (K_active * count_c))
        normalized to mean(active) = 1.0
        capped at max_weight (initially 4.0)

    Args:
        df: Training partition dataframe.
        task: Target task name ('hate', 'sarcasm', 'target', 'severity').
        mode: Weighting mode ('none', 'sqrt_balanced', 'linear_balanced').
        max_weight: Upper cap for normalized class weights.
        source_filter: Optional source_dataset string (e.g. 'multibully')
            to restrict weight derivation strictly to the primary benchmark.

    Returns:
        Dictionary containing counts, raw weights, normalized weights,
        capped weights, and distribution statistics.
    """
    mode = (mode or "none").lower()
    space = LABEL_SPACES.get(task, [])
    k = len(space)

    meta: Dict[str, Any] = {
        "task": task,
        "mode": mode,
        "label_space": list(space),
        "num_classes": k,
        "source_filter": source_filter,
        "max_weight": float(max_weight),
        "counts": {c: 0 for c in space},
        "total_supervised": 0,
        "raw_weights": [1.0] * k if k else [],
        "normalized_weights": [1.0] * k if k else [],
        "weights": None,
    }

    if mode == "none" or task not in df or k == 0:
        return meta

    # Apply optional source filter (e.g. derive weights exclusively from MultiBully)
    sub_df = df
    if source_filter is not None and "source_dataset" in df.columns:
        sub_df = df[df["source_dataset"] == source_filter]

    # Map labels to integer IDs, excluding missing and IGNORE_INDEX (-100)
    raw_labels = sub_df[task].tolist()
    mapped_ids = np.array([label_to_id(task, v) for v in raw_labels], dtype=int)
    valid_mask = (mapped_ids != IGNORE_INDEX) & (mapped_ids >= 0) & (mapped_ids < k)
    valid_ids = mapped_ids[valid_mask]

    n = len(valid_ids)
    meta["total_supervised"] = int(n)

    if n == 0:
        log.warning("Task '%s' has 0 supervised samples in training partition; returning unweighted", task)
        meta["weights"] = [1.0] * k
        return meta

    counts = np.bincount(valid_ids, minlength=k).astype(float)
    meta["counts"] = {space[i]: int(counts[i]) for i in range(k)}

    active = counts > 0
    k_active = int(active.sum())

    if k_active == 0:
        meta["weights"] = [1.0] * k
        return meta

    raw_weights = np.zeros(k, dtype=float)

    if mode in ("sqrt_balanced", "sqrt", "sqrt_inverse"):
        # weight_c = sqrt(N / (K_active * count_c))
        raw_weights[active] = np.sqrt(n / (k_active * counts[active]))
    elif mode in ("linear_balanced", "balanced", "linear"):
        # weight_c = N / (K_active * count_c)
        raw_weights[active] = n / (k_active * counts[active])
    else:
        raise ValueError(f"Unknown class_weighting mode: '{mode}'. Supported: 'none', 'sqrt_balanced', 'linear_balanced'")

    # Normalize weights so the mean over active classes is exactly 1.0
    active_mean = float(raw_weights[active].mean())
    norm_weights = np.zeros(k, dtype=float)
    if active_mean > 0:
        norm_weights[active] = raw_weights[active] / active_mean

    # Cap extreme weights at configurable threshold (default 4.0)
    capped_weights = np.minimum(norm_weights, float(max_weight))

    # Zero-support classes explicitly receive 0.0 weight to prevent division by zero / spurious loss
    for c in range(k):
        if counts[c] == 0:
            capped_weights[c] = 0.0

    # Strict validation: finite, non-negative
    if not (np.isfinite(capped_weights).all() and (capped_weights >= 0.0).all()):
        raise ValueError(f"Computed invalid non-finite or negative class weights for task {task}: {capped_weights}")

    meta["raw_weights"] = [float(x) for x in raw_weights]
    meta["normalized_weights"] = [float(x) for x in norm_weights]
    meta["weights"] = [float(x) for x in capped_weights]
    meta["mean_active_weight"] = float(capped_weights[active].mean()) if active.any() else 0.0

    return meta


def compute_fold_class_weights(
    df: pd.DataFrame,
    tasks: List[str],
    mode: str = "sqrt_balanced",
    max_weight: float = 4.0,
    source_filter: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute fold-specific class weights across all model tasks.

    Args:
        df: Training dataframe for the current fold.
        tasks: List of model tasks to balance.
        mode: Weighting mode.
        max_weight: Maximum allowed weight cap.
        source_filter: Optional source_dataset string filter.

    Returns:
        Dictionary mapping task names to their balancing metadata and weights.
    """
    res = {}
    for task in tasks:
        res[task] = compute_task_class_weights(
            df, task, mode=mode, max_weight=max_weight, source_filter=source_filter
        )
    return res
