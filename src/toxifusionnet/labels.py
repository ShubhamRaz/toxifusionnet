"""Label spaces, taken from the project flow chart (Objective box)."""
from __future__ import annotations

IGNORE_INDEX = -100  # label value meaning "no supervision for this task on this sample"

SARCASM_LABELS = ["not_sarcastic", "sarcastic"]
HATE_LABELS = ["non_hate", "hate"]
# Target taxonomy is configurable (config key ``labels.target``). Default = the flow chart's five classes;
# MultiBully annotates a different taxonomy (see MULTIBULLY_TARGET_LABELS), which configs/multibully.yaml selects.
DEFAULT_TARGET_LABELS = ("individual", "caste", "religion", "gender", "demography")
MULTIBULLY_TARGET_LABELS = ("individual", "organization", "community", "society")
TARGET_LABELS = list(DEFAULT_TARGET_LABELS)  # mutated IN PLACE by set_target_labels so every importer sees the change
SEVERITY_LABELS = ["low", "moderate", "high"]

TASKS = ("sarcasm", "hate", "target", "severity")
LABEL_SPACES = {
    "sarcasm": SARCASM_LABELS,
    "hate": HATE_LABELS,
    "target": TARGET_LABELS,
    "severity": SEVERITY_LABELS,
}
# Tasks that only make sense when the post is hate speech (flow chart: "If Hate Speech is Detected").
HATE_CONDITIONAL_TASKS = ("target", "severity")


def set_target_labels(classes) -> None:
    classes = [str(c).strip().lower() for c in classes]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError(f"target classes must be >=2 and unique, got {classes}")
    TARGET_LABELS[:] = classes


def apply_label_config(labels_cfg=None) -> None:
    """Apply the ``labels`` section of a config / schema (resets to the default taxonomy when absent)."""
    target = (labels_cfg or {}).get("target") if labels_cfg else None
    set_target_labels(target if target else DEFAULT_TARGET_LABELS)


def num_classes(task: str) -> int:
    return len(LABEL_SPACES[task])


def label_to_id(task: str, value) -> int:
    """Map a canonical string label (or an int already in range) to an id; IGNORE_INDEX if missing."""
    if value is None:
        return IGNORE_INDEX
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return IGNORE_INDEX
    except Exception:  # pragma: no cover
        pass
    space = LABEL_SPACES[task]
    if isinstance(value, str):
        v = value.strip().lower()
        if v in space:
            return space.index(v)
        if v.lstrip("-").isdigit():
            value = int(v)
        else:
            return IGNORE_INDEX
    iv = int(value)
    return iv if 0 <= iv < len(space) else IGNORE_INDEX


def id_to_label(task: str, idx: int) -> str:
    return LABEL_SPACES[task][idx]
