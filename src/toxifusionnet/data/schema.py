"""Schema-driven conversion of an arbitrary raw dataset (e.g. Kaggle *Multi-Bully*) to the canonical format.

Canonical columns (labels are human-readable strings, empty/NaN = "no supervision"):
    id, text, sarcasm, hate, target, severity, sarcasm_rationale, hate_rationale, narrative, label_source, split
The raw file layout is described in a YAML schema (see ``configs/schema_multibully.yaml``).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..labels import HATE_LABELS, LABEL_SPACES, SARCASM_LABELS, SEVERITY_LABELS, TARGET_LABELS, apply_label_config  # noqa: F401

log = logging.getLogger("toxifusionnet")

POS_VALUES = {"1", "true", "yes", "y", "sarcastic", "sarcasm", "hate", "hateful", "hatespeech", "bully", "bullying",
              "cyberbullying", "other_cyberbullying", "abusive", "toxic"}
NEG_VALUES = {"0", "false", "no", "n", "not_sarcastic", "non_sarcastic", "non_hate", "not_hate", "none", "normal",
              "neutral", "not_bully", "not_bullying", "not_cyberbullying", "non_bully", "clean", "non_toxic"}
TARGET_SYNONYMS = {
    "individual": "individual", "person": "individual", "personal": "individual", "personal_attack": "individual",
    "caste": "caste", "casteist": "caste", "religion": "religion", "religious": "religion", "gender": "gender",
    "sexist": "gender", "sexism": "gender", "sex": "gender", "demography": "demography", "demographic": "demography",
    "ethnicity": "demography", "race": "demography", "nationality": "demography", "region": "demography",
    "origin": "demography", "age": "demography",
    "organization": "organization", "organisation": "organization", "community": "community", "society": "society",
}
SEVERITY_SYNONYMS = {"low": "low", "mild": "low", "0": "low", "moderate": "moderate", "medium": "moderate", "1": "moderate",
                     "high": "high", "severe": "high", "2": "high"}
CANONICAL_COLUMNS = ["id", "text", "sarcasm", "hate", "target", "severity", "sarcasm_rationale", "hate_rationale",
                     "narrative", "label_source"]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".csv", ".txt"):
        kw = dict(keep_default_na=False, na_values=[""])   # "None"/"NA" are legitimate label/text values here
        try:
            return pd.read_csv(path, **kw)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1", **kw)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])
    if suf == ".json":
        return pd.read_json(path)
    if suf == ".jsonl":
        return pd.read_json(path, lines=True)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def meta_path(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix(".meta.json")


def write_meta(csv_path: str | Path, schema: dict, extra: dict | None = None) -> Path:
    """Sidecar next to a processed CSV: the label taxonomy and provenance travel with the data."""
    meta = {"labels": dict(schema.get("labels") or {}), "schema_file": schema.get("_schema_file"),
            "split": schema.get("split"), **(extra or {})}
    p = meta_path(csv_path)
    p.write_text(json.dumps(meta, indent=2))
    return p


def load_processed_csv(path: str | Path) -> pd.DataFrame:
    """Read a canonical CSV and activate the label taxonomy recorded by ``prepare`` (so target classes are right)."""
    df = pd.read_csv(path)
    mp = meta_path(path)
    apply_label_config(json.loads(mp.read_text()).get("labels") if mp.exists() else None)
    return df


def load_schema(path: str | Path) -> dict:
    sch = yaml.safe_load(Path(path).read_text()) or {}
    sch["_schema_file"] = str(path)
    return sch


def _norm(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(v).strip().lower().replace(" ", "_")


def _to_binary(v, pos: set, neg: set, space: list[str]) -> str | float:
    s = _norm(v)
    if s is None:
        return np.nan
    if s in pos:
        return space[1]
    if s in neg:
        return space[0]
    return np.nan


def _map_value(v, mapping: dict, default_syn: dict | None = None):
    s = _norm(v)
    if s is None:
        return np.nan
    if s in mapping:
        return np.nan if mapping[s] is None else mapping[s]
    if default_syn and s in default_syn:
        return default_syn[s]
    return np.nan


def _lower_keys(d: dict | None) -> dict:
    return {str(k).strip().lower().replace(" ", "_"): v for k, v in (d or {}).items()}


def _canon(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_column(df: pd.DataFrame, name):
    """Find ``name`` in df.columns ignoring case, spaces, hyphens and underscores (``Img-Text`` == ``img_text``)."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    if name in df.columns:
        return name
    lookup = {}
    for c in df.columns:
        lookup.setdefault(_canon(c), c)
    return lookup.get(_canon(name))


def _validate_target(values: pd.Series) -> pd.Series:
    """Targets outside the active taxonomy cannot be trained on -> NaN (and reported)."""
    space = set(LABEL_SPACES["target"])
    bad = values.dropna()[~values.dropna().isin(space)]
    if len(bad):
        log.warning("%d target labels outside the active taxonomy %s were dropped (examples: %s)",
                    len(bad), sorted(space), sorted(set(map(str, bad)))[:5])
    return values.where(values.isna() | values.isin(space), np.nan)


def build_canonical(df: pd.DataFrame, schema: dict, weak_fill: bool = True) -> pd.DataFrame:
    """Apply a schema to a raw dataframe and return the canonical dataframe."""
    apply_label_config(schema.get("labels"))
    text_col = resolve_column(df, schema.get("text_col", "text"))
    if text_col is None:
        raise KeyError(f"text column '{schema.get('text_col')}' not found; available: {list(df.columns)}")
    cols = {k: resolve_column(df, v) for k, v in (schema.get("columns", {}) or {}).items() if v}
    derive = schema.get("derive", {}) or {}
    vmaps = {t: _lower_keys(m) for t, m in (schema.get("value_maps", {}) or {}).items()}

    df = df.copy()
    rules = schema.get("exclude") or []
    for exc in ([rules] if isinstance(rules, dict) else rules):
        rc = resolve_column(df, exc.get("column"))
        if not rc:
            continue
        flagged = df[rc].notna() & (df[rc].astype(str).str.strip() != "")
        log.info("exclude: %d rows have a non-empty '%s'", int(flagged.sum()), rc)
        if exc.get("when_not_empty", True):
            df = df[~flagged]

    out = pd.DataFrame({"text": df[text_col].astype(str).str.strip()}, index=df.index)
    id_col = resolve_column(df, schema.get("id_col"))
    out["id"] = df[id_col].astype(str) if id_col else np.arange(len(df)).astype(str)
    source: dict[str, str] = {}

    def col_or_derived(task: str):
        if cols.get(task):
            source[task] = "gold"
            return df[cols[task]]
        d = derive.get(task) or {}
        dc = resolve_column(df, d.get("from"))
        if dc:
            source[task] = "gold-derived"
            return df[dc]
        return None

    raw = col_or_derived("hate")
    if raw is not None:
        d = derive.get("hate") or {}
        non_hate = {_norm(x) for x in d.get("non_hate_values", [])}
        hate_vals = {_norm(x) for x in d.get("hate_values", [])}
        ignored_vals = {_norm(x) for x in d.get("ignored_values", []) or d.get("ignore_values", [])}
        if non_hate or hate_vals:
            out_hate = []
            for v in raw:
                nv = _norm(v)
                if nv is None:
                    out_hate.append(np.nan)
                elif nv in ignored_vals:
                    out_hate.append(np.nan)
                elif nv in non_hate:
                    out_hate.append(HATE_LABELS[0])
                elif hate_vals and nv in hate_vals:
                    out_hate.append(HATE_LABELS[1])
                elif not hate_vals:
                    # Explicit check: do NOT map offensive/needs_review/uncertain to hate
                    if nv in ("offensive", "2", "2.0", "needs_review", "uncertain"):
                        out_hate.append(np.nan)
                    elif nv in POS_VALUES or nv in TARGET_SYNONYMS or nv in ("bully", "hate", "cyberbullying"):
                        out_hate.append(HATE_LABELS[1])
                    else:
                        raise ValueError(f"Unknown class '{v}' encountered in hate derivation. Cannot silently map to hate.")
                else:
                    raise ValueError(f"Unknown class '{v}' encountered in hate derivation. Expected hate_values {hate_vals} or non_hate_values {non_hate}.")
            out["hate"] = out_hate
        else:
            m = vmaps.get("hate", {})
            out_hate = []
            for v in raw:
                nv = _norm(v)
                if nv is None:
                    out_hate.append(np.nan)
                elif nv in m:
                    out_hate.append(m[nv])
                elif nv in ("offensive", "2", "2.0", "needs_review", "uncertain"):
                    out_hate.append(np.nan)
                elif nv in POS_VALUES:
                    out_hate.append(HATE_LABELS[1])
                elif nv in NEG_VALUES:
                    out_hate.append(HATE_LABELS[0])
                else:
                    raise ValueError(f"Unknown class '{v}' encountered in hate column. Cannot silently map to hate.")
            out["hate"] = out_hate
    else:
        out["hate"] = np.nan

    raw = col_or_derived("target")
    if raw is not None:
        mapping = _lower_keys((derive.get("target") or {}).get("map")) or vmaps.get("target", {})
        out["target"] = _validate_target(pd.Series([_map_value(v, mapping, TARGET_SYNONYMS) for v in raw], index=out.index, dtype=object))
    else:
        out["target"] = np.nan

    raw = col_or_derived("sarcasm")
    if raw is not None:
        m = vmaps.get("sarcasm", {})
        out["sarcasm"] = [m[_norm(v)] if _norm(v) in m else _to_binary(v, POS_VALUES, NEG_VALUES, SARCASM_LABELS) for v in raw]
    else:
        out["sarcasm"] = np.nan
    raw = col_or_derived("severity")
    if raw is not None:
        out["severity"] = [_map_value(v, vmaps.get("severity", {}), SEVERITY_SYNONYMS) for v in raw]
    else:
        out["severity"] = np.nan

    for key in ("sarcasm_rationale", "hate_rationale"):
        out[key] = df[cols[key]] if cols.get(key) else None
    out["narrative"] = df[cols["narrative"]] if cols.get("narrative") else np.nan

    # target / severity are undefined for non-hate posts (flow chart: classified only if hate is detected)
    out.loc[out["hate"] == HATE_LABELS[0], ["target", "severity"]] = np.nan

    out["label_source"] = ""
    if weak_fill:
        from .weak_labels import apply_weak_labels
        want = [t for t in ("sarcasm", "severity") if (derive.get(t) or {}).get("weak") and t not in source]
        if want:
            out = apply_weak_labels(out, want)
            for t in want:
                source[t] = "weak"
    out["label_source"] = json.dumps(source)
    out = out[out["text"].str.len() > 0]
    task_cols = [t for t in ("hate", "sarcasm", "target", "severity") if t in out.columns]
    if task_cols:
        n0 = len(out)
        out = out[out[task_cols].notna().any(axis=1)]
        if len(out) < n0:
            log.info("dropped %d rows without any task supervision", n0 - len(out))
    return out[CANONICAL_COLUMNS]


def guess_schema(df: pd.DataFrame) -> dict:
    """Heuristically propose a schema for an unknown dataframe (used by ``toxifusion inspect``)."""
    obj_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
    text_col = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean()) if obj_cols else df.columns[0]
    low = {c: c.lower() for c in df.columns}
    find = lambda *keys: next((c for c, l in low.items() if any(k in l for k in keys) and c != text_col), None)  # noqa: E731
    schema = {"text_col": text_col, "id_col": find("id"), "columns": {
        "sarcasm": find("sarcas", "irony"), "hate": find("hate"), "target": find("target"),
        "severity": find("sever", "intens", "level"), "sarcasm_rationale": find("sarcasm_rat"),
        "hate_rationale": find("hate_rat", "rationale"), "narrative": find("narrat", "fake", "misinfo")},
        "derive": {}, "value_maps": {}}
    label_col = find("label", "class", "bully", "toxic", "tag", "type", "category")
    if label_col and not schema["columns"]["hate"]:
        vals = sorted(map(str, df[label_col].dropna().unique()))[:30]
        schema["derive"]["hate"] = {"from": label_col, "non_hate_values": [v for v in vals if _norm(v) in NEG_VALUES] or ["<EDIT_ME>"]}
        schema["derive"]["target"] = {"from": label_col, "map": {v: "<individual|caste|religion|gender|demography>" for v in vals}}
    if not schema["columns"]["sarcasm"]:
        schema["derive"]["sarcasm"] = {"weak": True}
    if not schema["columns"]["severity"]:
        schema["derive"]["severity"] = {"weak": True}
    return schema
