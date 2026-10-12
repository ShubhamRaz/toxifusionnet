"""Ablation / baseline-model runner: trains every variant (x seeds), aggregates mean +- std, runs paired-bootstrap tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..training.stats import paired_bootstrap
from ..training.trainer import train_or_resume

VARIANTS: dict[str, list[str]] = {
    "full": [],
    "hate_only_transformer": ["model.tasks=[hate]", "model.use_sarcasm_branch=false"],   # standard transformer classifier baseline
    "multitask_no_sarcasm": ["model.use_sarcasm_branch=false"],                            # hate+target+severity, no sarcasm
    "sarcasm_aux_no_fusion": ["model.use_fusion=false"],                                   # sarcasm as auxiliary task only
    "no_cue_features": ["model.use_cue_features=false"],
    "no_rationale_supervision": ["rationale.enabled=false"],
    "hate_rationale_only": ["rationale.use_sarcasm_rationale=false"],
    "uncertainty_loss_weighting": ["loss.weighting=uncertainty"],
    "no_normalization": ["preprocess.normalize=false"],
    "no_transliteration": ["preprocess.transliterate_devanagari=false"],
}
KEY_METRICS = ["hate_macro_f1", "hate_FNR_sarcastic", "hate_F1_sarcastic", "sarcasm_macro_f1", "target_macro_f1",
               "severity_macro_f1", "mean_macro_f1"]


def collect_finished_runs(out_dir: str | Path) -> pd.DataFrame:
    """One row per finished (variant, seed) run directory found under ``out_dir`` -- the summary never covers a subset."""
    out_dir = Path(out_dir)
    rows = []
    for vdir in sorted(p for p in out_dir.iterdir() if p.is_dir() and not p.name.startswith("cv")):
        for sdir in sorted(vdir.glob("seed*")):
            if (sdir / "metrics.json").exists():
                rows.append({"variant": vdir.name, "seed": int(sdir.name[4:]), **json.loads((sdir / "metrics.json").read_text())["test_flat"]})
    order = {v: i for i, v in enumerate(VARIANTS)}
    return pd.DataFrame(rows).sort_values(by=["variant", "seed"], key=lambda c: c.map(order) if c.name == "variant" else c).reset_index(drop=True)


def run_ablations(config: str, base_overrides: list[str], out_dir: str | Path, variants: list[str] | None = None,
                  seeds=(42,)) -> pd.DataFrame:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    base_cfg = load_config(config, base_overrides)
    wanted = variants or list(VARIANTS)
    if not base_cfg.get("rationale", {}).get("enabled", False):   # rationale variants would be identical to "full"
        skipped = [v for v in wanted if v in ("no_rationale_supervision", "hate_rationale_only")]
        if skipped:
            print(f"skipping {skipped}: rationale supervision is disabled in the base config (no rationale annotations)")
        wanted = [v for v in wanted if v not in skipped]
    for name in wanted:
        for seed in seeds:
            cfg = load_config(config, base_overrides + VARIANTS[name] + [f"seed={seed}"])
            df = pd.read_csv(cfg.data.processed_path)
            res = train_or_resume(cfg, df, out_dir / name / f"seed{seed}")
            rows.append({"variant": name, "seed": seed, **res["test_flat"]})
    raw = collect_finished_runs(out_dir)
    raw.to_csv(out_dir / "ablation_raw.csv", index=False)
    cols = [c for c in KEY_METRICS if c in raw]
    agg = raw.groupby("variant", sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = raw.groupby("variant", sort=False).size()
    agg.to_csv(out_dir / "ablation_summary.csv")
    # paired bootstrap: full vs every other variant (first seed), whole test set and sarcastic-hate subset
    tests = {}
    s0 = seeds[0]
    full_p = out_dir / "full" / f"seed{s0}" / "preds_test.csv"
    if full_p.exists():
        full = pd.read_csv(full_p)
        warned = False
        for name in raw["variant"].unique():
            if name == "full":
                continue
            p = out_dir / name / f"seed{s0}" / "preds_test.csv"
            if not p.exists():
                continue
            other = pd.read_csv(p)
            m = (full.gold_hate == 1) & (full.gold_sarcasm == 1)
            if int(m.sum()) < 30 and not warned:
                warned = True
                print(f"WARNING: only {int(m.sum())} sarcastic-hate test posts -> bootstrap on this subset is underpowered")
            tests[name] = {
                "full_vs_variant_all": paired_bootstrap(full.gold_hate, other.pred_hate, full.pred_hate, seed=s0),
                "full_vs_variant_sarcastic_hate": paired_bootstrap(full.gold_hate, other.pred_hate, full.pred_hate, seed=s0, mask=m,
                                                                   metric=lambda yt, yp: float((yp[yt == 1] == 1).mean()) if (yt == 1).any() else 0.0)}
    (out_dir / "significance.json").write_text(json.dumps(tests, indent=2))
    return agg
