"""Collect every result artefact into one markdown file (results/RESULTS.md) + CSVs ready for the paper tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..training.metrics import flatten_metrics


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.copy()
    cols = [df.index.name or "system"] + list(df.columns)
    rows = []
    for idx, r in df.iterrows():
        rows.append([str(idx)] + [floatfmt.format(v) if isinstance(v, (float, int)) and not pd.isna(v) else ("" if pd.isna(v) else str(v)) for v in r.tolist()])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def _cv_section(cv_dir: Path) -> list[str]:
    from ..labels import LABEL_SPACES
    from ..training.metrics import evaluate_predictions
    from ..training.stats import paired_bootstrap
    cv = json.loads((cv_dir / "cv_metrics.json").read_text())
    L = [f"## 9. Cross-validation, pooled out-of-fold predictions (`{cv_dir}`)\n",
         "Every labelled post is predicted exactly once by a model that never saw it; this is the reliable source for per-class results.\n"]
    flat = pd.DataFrame({k: v["pooled_flat"] for k, v in cv.items()}).T
    keep = [c for c in ("mean_macro_f1", "hate_macro_f1", "hate_auroc", "sarcasm_macro_f1", "sarcasm_auroc", "target_macro_f1",
                        "target_bal_acc", "severity_macro_f1", "severity_bal_acc", "hate_FNR_sarcastic", "hate_FNR_non_sarcastic") if c in flat]
    L += [_md(flat[keep]), "\n"]
    if "model" in cv:
        for t in ("target", "severity"):
            if t in cv["model"]["pooled"]:
                pc = cv["model"]["pooled"][t]["per_class"]; cls = cv["model"]["pooled"][t]["classes"]
                rows = {cls[int(k)]: {"support": v["support"], "precision": v["precision"], "recall": v["recall"], "f1": v["f1"]} for k, v in pc.items()}
                L += [f"**{t}: per-class results (model, pooled)**\n", _md(pd.DataFrame(rows).T), "\n"]
        L += ["**Hate: per-fold macro-F1 (model)**: " + ", ".join(f"{x:.3f}" for x in cv["model"]["per_fold_hate_macro_f1"]) + "\n"]
    if (cv_dir / "pooled_model.csv").exists() and (cv_dir / "pooled_tfidf_lr.csv").exists():
        a = pd.read_csv(cv_dir / "pooled_tfidf_lr.csv"); b = pd.read_csv(cv_dir / "pooled_model.csv")
        m = a.merge(b, on="id", suffixes=("_a", "_b"))
        sh = (m.gold_hate_a == 1) & (m.gold_sarcasm_a == 1)
        rec = lambda yt, yp: float((yp[yt == 1] == 1).mean()) if (yt == 1).any() else 0.0  # noqa: E731
        r1 = paired_bootstrap(m.gold_hate_a, m.pred_hate_a, m.pred_hate_b, n_boot=1000)
        r2 = paired_bootstrap(m.gold_hate_a, m.pred_hate_a, m.pred_hate_b, n_boot=1000, mask=sh, metric=rec)
        L += ["**Model vs TF-IDF + LR (paired bootstrap on pooled predictions; positive = model better)**\n",
              _md(pd.DataFrame({"hate macro-F1 (all posts)": {"delta": r1["delta_mean"], "CI_lo": r1["ci95"][0], "CI_hi": r1["ci95"][1], "p": r1["p_value"], "n": r1["n"]},
                                "recall on sarcastic hate": {"delta": r2["delta_mean"], "CI_lo": r2["ci95"][0], "CI_hi": r2["ci95"][1], "p": r2["p_value"], "n": r2["n"]}}).T), "\n"]
    sib = []
    for d in sorted(cv_dir.parent.glob("cv_*")):
        if (d / "pooled_model.csv").exists() and (cv_dir / "pooled_model.csv").exists():
            a, b = pd.read_csv(d / "pooled_model.csv"), pd.read_csv(cv_dir / "pooled_model.csv")
            m = a.merge(b, on="id", suffixes=("_a", "_b"))
            sh = (m.gold_hate_a == 1) & (m.gold_sarcasm_a == 1)
            rec = lambda yt, yp: float((yp[yt == 1] == 1).mean()) if (yt == 1).any() else 0.0  # noqa: E731
            r1 = paired_bootstrap(m.gold_hate_a, m.pred_hate_a, m.pred_hate_b, n_boot=1000)
            r2 = paired_bootstrap(m.gold_hate_a, m.pred_hate_a, m.pred_hate_b, n_boot=1000, mask=sh, metric=rec)
            sib.append((d.name[3:], flatten_metrics(evaluate_predictions(a, [t for t in ("sarcasm", "hate", "target", "severity") if f"pred_{t}" in a])), r1, r2))
    if sib:
        rows = {}
        for name, fl, r1, r2 in sib:
            rows[name] = {"hate macro-F1 of variant": fl.get("hate_macro_f1"), "delta all (full - variant)": r1["delta_mean"], "CI_lo": r1["ci95"][0],
                          "CI_hi": r1["ci95"][1], "p": r1["p_value"], "delta recall sarc. hate": r2["delta_mean"], "CI2_lo": r2["ci95"][0],
                          "CI2_hi": r2["ci95"][1], "p2": r2["p_value"], "n_sarc_hate": r2["n"]}
        L += ["**Full model vs. variants (pooled cross-validation, paired bootstrap; positive delta = full model better)**\n", _md(pd.DataFrame(rows).T), "\n"]
    return L


def write_results_md(out: str | Path, run_dir: str | Path | None = None, ablation_dir: str | Path | None = None,
                     baselines_csv: str | Path | None = None, data_report: str | Path | None = None,
                     cv_dir: str | Path | None = None) -> Path:
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    L = ["# ToxiFusionNet -- results\n", "_Generated by `toxifusion report`. Every number below comes from the run artefacts listed in the section headers._\n"]
    if data_report and Path(data_report).exists():
        L += ["## 1. Data\n", Path(data_report).read_text().replace("# Dataset report", "").strip(), "\n"]
    if baselines_csv and Path(baselines_csv).exists():
        b = pd.read_csv(baselines_csv, index_col=0)
        L += [f"## 2. Baselines (`{baselines_csv}`)\n", _md(b), "\n"]
    if ablation_dir:
        ad = Path(ablation_dir)
        if (ad / "ablation_summary.csv").exists():
            s = pd.read_csv(ad / "ablation_summary.csv", index_col=0)
            L += [f"## 3. Models and ablations (mean/std over seeds) (`{ad}/ablation_summary.csv`)\n", _md(s), "\n"]
        if (ad / "significance.json").exists():
            sig = json.loads((ad / "significance.json").read_text())
            rows = {}
            for name, v in sig.items():
                a, c = v["full_vs_variant_all"], v["full_vs_variant_sarcastic_hate"]
                rows[name] = {"d_macroF1_all": a["delta_mean"], "CI_lo": a["ci95"][0], "CI_hi": a["ci95"][1], "p_all": a["p_value"],
                              "d_recall_sarc_hate": c["delta_mean"], "CI2_lo": c["ci95"][0], "CI2_hi": c["ci95"][1], "p_sarc": c["p_value"],
                              "n_sarc_hate": c["n"], "underpowered(n<30)": "YES" if c["n"] < 30 else "no"}
            L += ["## 4. Paired bootstrap, full model vs. each variant (positive delta = full model better)\n", _md(pd.DataFrame(rows).T), "\n"]
    if run_dir:
        rd = Path(run_dir)
        if (rd / "robustness.csv").exists():
            L += ["## 5. Robustness (full model, first seed)\n", _md(pd.read_csv(rd / "robustness.csv").set_index("perturbation")), "\n"]
        if (rd / "explainability.json").exists():
            L += ["## 6. Explainability\n", "```json\n" + (rd / "explainability.json").read_text() + "\n```\n"]
        if (rd / "benchmark.csv").exists():
            L += ["## 7. Latency / size (CPU)\n", _md(pd.read_csv(rd / "benchmark.csv").set_index("model"), "{:.2f}"), "\n"]
        if (rd / "metrics.json").exists():
            m = json.loads((rd / "metrics.json").read_text())
            ss = m["test"].get("sarcasm_subset")
            if ss:
                L += ["## 8. Hate detection by sarcasm (test, full model, first seed)\n", _md(pd.DataFrame(ss).T), "\n"]
            L += [f"label sources: `{m.get('label_source')}`\n"]
    if cv_dir and (Path(cv_dir) / "cv_metrics.json").exists():
        L += _cv_section(Path(cv_dir))
    out.write_text("\n".join(L))
    return out
