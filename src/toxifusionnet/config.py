"""YAML-backed configuration with dotted command-line overrides (e.g. ``model.use_fusion=false``)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


class Config(dict):
    """A dict with attribute access (recursively)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return v

    def __setattr__(self, k, v):
        self[k] = v

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))


def _parse_value(s: str) -> Any:
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like a.b.c=value, got: {item}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def _load_yaml_chain(path: Path, default_path: Path | None, _seen=()) -> dict:
    """Load ``path``; if it has ``extends: other.yaml`` that file is loaded first (recursively), otherwise
    ``default.yaml`` from the same folder is the base."""
    path = Path(path)
    if path.resolve() in _seen:
        raise ValueError(f"circular 'extends' at {path}")
    own = yaml.safe_load(path.read_text()) or {}
    ext = own.pop("extends", None)
    if ext:
        base = _load_yaml_chain(path.parent / ext, default_path, _seen + (path.resolve(),))
    elif default_path and default_path.exists() and default_path.resolve() != path.resolve():
        base = yaml.safe_load(default_path.read_text()) or {}
    else:
        base = {}
    return _deep_update(base, own)


def load_config(path: str | Path, overrides: list[str] | None = None, default_path: str | Path | None = None) -> Config:
    """Merge ``default.yaml`` <- (``extends`` chain) <- ``path`` <- dotted command-line overrides."""
    path = Path(path)
    default_path = Path(default_path) if default_path else path.parent / "default.yaml"
    cfg = _load_yaml_chain(path, default_path)
    apply_overrides(cfg, overrides)
    return Config.wrap(copy.deepcopy(cfg))
