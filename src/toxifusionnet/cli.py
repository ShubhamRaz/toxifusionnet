"""Command line interface:  ``toxifusion <command> [options]``  (or ``python -m toxifusionnet.cli ...``)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd


def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def _load_cfg(args):
    from .config import load_config
    return load_config(args.config, args.override)


def cmd_synth(args):
    from .data.splits import make_splits
    from .data.synthetic import make_multibully_layout, make_synthetic
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.layout == "multibully":   # raw file in the real MultiBully layout -> run `prepare` on it
        df = make_multibully_layout(args.n, args.seed)
    else:
        df = make_splits(make_synthetic(args.n, args.seed))
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} SYNTHETIC rows ({args.layout} layout) to {args.out} (pipeline testing only)")


def cmd_inspect(args):
    from .data.schema import guess_schema, read_table
    import yaml
    df = read_table(args.raw)
    print(f"shape: {df.shape}\ncolumns/dtypes:\n{df.dtypes.to_string()}\n\nhead:\n{df.head(5).to_string()[:2000]}\n")
    for c in df.columns:
        if df[c].nunique() <= 25:
            print(f"--- value counts for '{c}':\n{df[c].value_counts(dropna=False).to_string()}\n")
    sch = guess_schema(df)
    sch["file"] = str(args.raw)
    print("=== suggested schema draft (review and save as configs/schema_<name>.yaml) ===")
    print(yaml.safe_dump(sch, sort_keys=False, allow_unicode=True))
    from .preprocessing.language_id import analyze
    sample = df[sch["text_col"]].astype(str).sample(min(500, len(df)), random_state=0)
    cmi = [analyze(t).cmi for t in sample]
    print(f"mean code-mixing index on 500 sampled posts: {sum(cmi)/len(cmi):.1f}  (0 = monolingual; Hinglish corpora are typically >10)")


def cmd_prepare(args):
    from .data.schema import build_canonical, load_schema, read_table, write_meta
    from .data.splits import make_splits
    sch = load_schema(args.schema)
    raw = args.raw or sch.get("file")
    df = build_canonical(read_table(raw), sch, weak_fill=not args.no_weak)
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    sp = sch.get("split", {}) or {}
    val = args.val_size if args.val_size is not None else float(sp.get("val_size", 0.1))
    test = args.test_size if args.test_size is not None else float(sp.get("test_size", 0.1))
    df = make_splits(df, val, test, args.seed if args.seed is not None else int(sp.get("seed", 42)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    write_meta(args.out, sch, {"raw_file": str(raw), "n_rows": int(len(df))})
    print(f"wrote {len(df)} rows -> {args.out}\nlabel_source: {df.label_source.iloc[0]}")
    for t in ("sarcasm", "hate", "target", "severity"):
        print(f"\n{t}:\n{df[t].value_counts(dropna=False).to_string()}")


def cmd_train(args):
    from .training.trainer import train_run
    cfg = _load_cfg(args)
    df = pd.read_csv(cfg.data.processed_path)
    run_dir = Path(args.run_dir or Path(cfg.output_dir) / cfg.experiment_name)
    res = train_run(cfg, df, run_dir)
    print(json.dumps(res["test_flat"], indent=2))
    print(f"run saved in {run_dir}")


def _pipeline(args):
    from .inference.pipeline import ToxiFusionPipeline
    return ToxiFusionPipeline.from_run(args.run_dir, device=args.device)


def cmd_predict(args):
    pipe = _pipeline(args)
    texts = args.text or [l.rstrip("\n") for l in open(args.file, encoding="utf-8") if l.strip()]
    for r in pipe.predict(texts, explain=not args.no_explain):
        print(json.dumps(r, indent=2, ensure_ascii=False))


def cmd_baselines(args):
    from .baselines.classical import run_baselines
    from .data.schema import load_processed_csv
    df = load_processed_csv(args.data)
    out = run_baselines(df)
    tab = pd.DataFrame(out["metrics"]).T
    print(tab.to_string())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out)
    for name, pf in out["preds"].items():
        pf.to_csv(Path(args.out).with_name(f"preds_{name}.csv"), index=False)


def cmd_ablate(args):
    from .experiments.ablation import VARIANTS, run_ablations
    if args.aggregate_only:   # no training: just rebuild the summary from the finished run directories
        import pandas as pd
        from .experiments.ablation import KEY_METRICS, collect_finished_runs
        raw = collect_finished_runs(args.out_dir); raw.to_csv(Path(args.out_dir) / "ablation_raw.csv", index=False)
        cols = [c for c in KEY_METRICS if c in raw]
        agg = raw.groupby("variant", sort=False)[cols].agg(["mean", "std"]); agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg["n_seeds"] = raw.groupby("variant", sort=False).size(); agg.to_csv(Path(args.out_dir) / "ablation_summary.csv")
        print(agg.round(3).to_string()); return
    agg = run_ablations(args.config, args.override, args.out_dir, args.variants or None, tuple(args.seeds))
    print(agg.round(4).to_string())


def cmd_robust(args):
    from .robustness.evaluate import evaluate_robustness
    pipe = _pipeline(args)
    df = pd.read_csv(args.data)
    df = df[df.split == "test"].reset_index(drop=True)
    res = evaluate_robustness(pipe, df)
    print(res.round(4).to_string(index=False))
    res.to_csv(Path(args.run_dir) / "robustness.csv", index=False)


def cmd_explain(args):
    from .explain.evaluate import evaluate_rationales, export_human_eval_sheet
    pipe = _pipeline(args)
    df = pd.read_csv(args.data)
    te = df[df.split == "test"].reset_index(drop=True)
    res = evaluate_rationales(pipe, te, n=args.n)
    print(json.dumps(res, indent=2, default=float))
    (Path(args.run_dir) / "explainability.json").write_text(json.dumps(res, indent=2, default=float))
    sheet = export_human_eval_sheet(pipe, te, str(Path(args.run_dir) / "human_eval_sheet.csv"), n=args.sheet_n)
    print("human evaluation sheet:", sheet)
    if args.lime:
        from .explain.lime_text import lime_explain
        print(lime_explain(pipe, args.lime))


def cmd_bench(args):
    from .inference.benchmark import benchmark
    pipe = _pipeline(args)
    rows = benchmark(pipe, ["Wah wah, kya baat hai, great job lol 🙄 yaar tum log to genius ho"] , quantize=not args.no_quant)
    t = pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv(Path(args.run_dir) / "benchmark.csv", index=False)


def cmd_compare(args):
    if getattr(args, "out_dir", None) or getattr(args, "full", False):
        from .experiments.compare import compare_predictions
        name_a = getattr(args, "name_a", "Model_A")
        name_b = getattr(args, "name_b", "Model_B")
        out = compare_predictions(args.preds_a, args.preds_b, name_a=name_a, name_b=name_b, out_dir=getattr(args, "out_dir", None))
        for t, data in out["task_comparisons"].items():
            print(f"{t.upper()}: delta={data['delta_macro_f1']:+.4f}, 95% CI={data['delta_ci95']}, p={data['p_value_bootstrap']:.4f}")
    else:
        from .training.stats import paired_bootstrap
        a, b = pd.read_csv(args.preds_a), pd.read_csv(args.preds_b)
        m = (a.gold_hate == 1) & (a.get("gold_sarcasm", 0) == 1) if args.sarcastic_only else None
        print(json.dumps(paired_bootstrap(a.gold_hate, a.pred_hate, b.pred_hate, mask=m), indent=2))


def cmd_figures(args):
    from .experiments.plots import plot_run
    for p in plot_run(args.run_dir):
        print(p)


def cmd_serve(args):
    import os
    import uvicorn
    os.environ["TOXIFUSION_RUN_DIR"] = args.run_dir
    uvicorn.run("toxifusionnet.app.api:app", host=args.host, port=args.port)


def cmd_describe(args):
    from .analysis.describe import describe_dataset, to_markdown
    from .data.schema import load_processed_csv
    df = load_processed_csv(args.data)
    d = describe_dataset(df)
    md = to_markdown(d)
    print(md)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md); out.with_suffix(".json").write_text(json.dumps(d, indent=2, default=str))


def cmd_report(args):
    from .experiments.report import write_results_md
    print("wrote", write_results_md(args.out, args.run_dir, args.ablation_dir, args.baselines, args.data_report, args.cv_dir))


def cmd_cv(args):
    from .experiments.cv import run_cv
    out = run_cv(args.config, args.override, args.data, args.out_dir, args.folds, args.seed,
                  grouped=args.grouped, folds_cache=args.folds_cache, aux_data_path=getattr(args, "aux_data", None))
    for name, v in out.items():
        print(name, {k: round(x, 3) for k, x in v["pooled_flat"].items() if x == x})


def build_parser():
    p = argparse.ArgumentParser(prog="toxifusion", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_cfg(sp):
        sp.add_argument("--config", default="configs/default.yaml")
        sp.add_argument("--override", "-o", nargs="*", default=[], help="dotted overrides, e.g. model.use_fusion=false")

    s = sub.add_parser("synth", help="generate SYNTHETIC smoke-test data"); s.add_argument("--n", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--out", default="data/processed/synthetic.csv")
    s.add_argument("--layout", choices=["canonical", "multibully"], default="canonical"); s.set_defaults(fn=cmd_synth)
    s = sub.add_parser("inspect", help="inspect a raw dataset and propose a schema"); s.add_argument("--raw", required=True); s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("prepare", help="raw dataset + schema -> canonical processed CSV with splits")
    s.add_argument("--schema", default="configs/schema_multibully.yaml"); s.add_argument("--raw", default=None)
    s.add_argument("--out", default="data/processed/dataset.csv"); s.add_argument("--val-size", type=float, default=None)
    s.add_argument("--test-size", type=float, default=None); s.add_argument("--seed", type=int, default=None)
    s.add_argument("--no-weak", action="store_true", help="do not create weak sarcasm/severity labels"); s.set_defaults(fn=cmd_prepare)
    s = sub.add_parser("train", help="train ToxiFusionNet"); add_cfg(s); s.add_argument("--run-dir", default=None); s.set_defaults(fn=cmd_train)
    def add_run(sp):
        sp.add_argument("--run-dir", required=True); sp.add_argument("--device", default="cpu")

    s = sub.add_parser("predict", help="run the full flow-chart pipeline on text"); add_run(s)
    s.add_argument("--text", nargs="*"); s.add_argument("--file"); s.add_argument("--no-explain", action="store_true"); s.set_defaults(fn=cmd_predict)
    s = sub.add_parser("baselines", help="keyword / sentiment / TF-IDF baselines"); s.add_argument("--data", default="data/processed/dataset.csv")
    s.add_argument("--out", default="runs/baselines/baselines.csv"); s.set_defaults(fn=cmd_baselines)
    s = sub.add_parser("ablate", help="train all ablation variants (and standard-transformer baseline)"); add_cfg(s)
    s.add_argument("--out-dir", default="runs/ablation"); s.add_argument("--variants", nargs="*"); s.add_argument("--seeds", nargs="*", type=int, default=[42])
    s.add_argument("--aggregate-only", action="store_true", help="rebuild the summary from finished runs, no training"); s.set_defaults(fn=cmd_ablate)
    s = sub.add_parser("robust", help="robustness to spelling/transliteration/code-switch/fake-narrative perturbations"); add_run(s)
    s.add_argument("--data", default="data/processed/dataset.csv"); s.set_defaults(fn=cmd_robust)
    s = sub.add_parser("explain", help="rationale plausibility/faithfulness + human-evaluation sheet"); add_run(s)
    s.add_argument("--data", default="data/processed/dataset.csv"); s.add_argument("--n", type=int, default=200)
    s.add_argument("--sheet-n", type=int, default=100); s.add_argument("--lime", default=None, help="also print a LIME explanation for this text"); s.set_defaults(fn=cmd_explain)
    s = sub.add_parser("bench", help="latency / size benchmark (fp32 vs int8)"); add_run(s); s.add_argument("--no-quant", action="store_true"); s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("compare", help="paired bootstrap between two preds_test.csv files")
    s.add_argument("preds_a"); s.add_argument("preds_b"); s.add_argument("--sarcastic-only", action="store_true")
    s.add_argument("--out-dir", default=None, help="Directory to save JSON and Markdown comparison reports")
    s.add_argument("--name-a", default="Model_A", help="Name for Model A")
    s.add_argument("--name-b", default="Model_B", help="Name for Model B")
    s.add_argument("--full", action="store_true", help="Run full 4-task paired group-aware comparison")
    s.set_defaults(fn=cmd_compare)
    s = sub.add_parser("figures", help="plots for a run"); s.add_argument("--run-dir", required=True); s.set_defaults(fn=cmd_figures)
    s = sub.add_parser("describe", help="label distributions, sarcastic-hate counts per split, language mix")
    s.add_argument("--data", default="data/processed/dataset.csv"); s.add_argument("--out", default="results/data_report.md"); s.set_defaults(fn=cmd_describe)
    s = sub.add_parser("report", help="assemble all results into results/RESULTS.md")
    s.add_argument("--out", default="results/RESULTS.md"); s.add_argument("--run-dir"); s.add_argument("--ablation-dir")
    s.add_argument("--baselines"); s.add_argument("--cv-dir"); s.add_argument("--data-report", default="results/data_report.md"); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("cv", help="stratified k-fold cross-validation with pooled out-of-fold predictions"); add_cfg(s)
    s.add_argument("--data", default="data/processed/multibully.csv"); s.add_argument("--out-dir", default="runs/cv")
    s.add_argument("--folds", type=int, default=5); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--folds-cache", default="data/processed/multibully_v2_cv_folds.csv", help="path to persistent canonical fold mapping")
    s.add_argument("--aux-data", default=None, help="path to auxiliary training dataset (e.g. data/processed/hot_v2.csv) injected strictly into train partitions")
    s.add_argument("--no-grouped", dest="grouped", action="store_false", default=True, help="disable group-aware text splitting")
    s.set_defaults(fn=cmd_cv)
    s = sub.add_parser("serve", help="REST API"); add_run(s); s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=8000); s.set_defaults(fn=cmd_serve)
    return p


def main(argv=None):
    _setup_logging()
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
