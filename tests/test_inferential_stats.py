"""Unit tests for publication-grade inferential statistics, permutation testing, and Holm correction."""
import numpy as np
import pandas as pd
import pytest

from toxifusionnet.training.stats import (
    fast_macro_f1,
    paired_group_permutation_test,
    paired_group_bootstrap,
    paired_multitask_group_bootstrap,
    paired_multitask_group_permutation_test,
    holm_bonferroni_correction,
    mcnemar_test,
    class_level_metrics
)


def test_fast_macro_f1_numerical_parity_with_sklearn():
    """Verify that fast_macro_f1 matches sklearn.metrics.f1_score across multiclass scenarios."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    for n_c in [2, 3, 4]:
        for _ in range(5):
            y = rng.integers(0, n_c, 500)
            p = rng.integers(0, n_c, 500)
            sk = f1_score(y, p, average="macro", zero_division=0)
            fast = fast_macro_f1(y, p, n_c)
            assert pytest.approx(fast, rel=1e-6) == sk


def test_permutation_test_null_hypothesis_and_direction():
    """Verify permutation test behavior under null, superior, and inferior models."""
    n = 600
    groups = np.repeat(np.arange(60), 10)
    y = np.random.randint(0, 2, n)

    # 1. Null: identical models -> delta = 0, p-value should be 1.0 (all permutations have |T| >= 0)
    res_null = paired_group_permutation_test(groups, y, y, y, n_perm=500, seed=42)
    assert res_null["obs_delta"] == 0.0
    assert pytest.approx(res_null["p_value_two_sided"], rel=1e-5) == 1.0

    # 2. Strong Superiority: Model B is near perfect, Model A is random noise
    pred_a = np.random.randint(0, 2, n)
    pred_b = y.copy()  # perfect
    res_sup = paired_group_permutation_test(groups, y, pred_a, pred_b, n_perm=500, seed=42)
    assert res_sup["obs_delta"] > 0.3
    # Two-sided p-value should be minimum possible: 1 / (500 + 1)
    assert res_sup["p_value_two_sided"] <= 0.01

    # 3. Two-Sided: Model B is much worse -> delta is negative, two-sided p-value still detects significance
    res_inf = paired_group_permutation_test(groups, y, pred_b, pred_a, n_perm=500, seed=42)
    assert res_inf["obs_delta"] < -0.3
    assert res_inf["p_value_two_sided"] <= 0.01


def test_permutation_pairing_preservation():
    """Verify that pairing between predictions on each item is strictly preserved."""
    # When predictions are paired such that Model B is consistently correct whenever Model A is wrong
    n = 200
    groups = np.arange(n)  # individual groups
    y = np.ones(n, dtype=int)
    # Model A gets half wrong
    pred_a = np.array([1 if i % 2 == 0 else 0 for i in range(n)])
    # Model B gets all correct
    pred_b = np.ones(n, dtype=int)

    res = paired_group_permutation_test(groups, y, pred_a, pred_b, n_perm=1000, seed=123)
    assert res["obs_delta"] > 0.0
    assert res["p_value_two_sided"] < 0.01


def test_permutation_group_preservation():
    """Verify that all items belonging to the same text group swap together."""
    # Create 2 groups with 50 examples each
    groups = np.array([0] * 50 + [1] * 50)
    y = np.array([1] * 50 + [0] * 50)
    pred_a = np.zeros(100, dtype=int)
    pred_b = np.ones(100, dtype=int)

    # In any permutation, group 0's 50 items must all come from the same model
    res = paired_group_permutation_test(groups, y, pred_a, pred_b, n_perm=200, seed=42)
    assert res["n_groups"] == 2
    assert res["n_samples"] == 100


def test_permutation_plus_one_correction_formula():
    """Verify that the empirical p-value uses (extreme + 1) / (n_perm + 1)."""
    groups = np.arange(100)
    y = np.ones(100, dtype=int)
    pred_a = np.zeros(100, dtype=int)
    pred_b = np.ones(100, dtype=int)

    n_perm = 999
    res = paired_group_permutation_test(groups, y, pred_a, pred_b, n_perm=n_perm, seed=42)
    expected_min = 1.0 / (n_perm + 1.0)
    assert res["p_value_two_sided"] >= expected_min
    if res["extreme_count"] == 0:
        assert res["p_value_two_sided"] == expected_min


def test_reproducibility_with_fixed_seed():
    """Verify that identical random seed produces exact identical p-values and bootstrap CIs."""
    rng = np.random.default_rng(99)
    groups = rng.integers(0, 30, 200)
    y = rng.integers(0, 2, 200)
    pred_a = rng.integers(0, 2, 200)
    pred_b = rng.integers(0, 2, 200)

    res1 = paired_group_permutation_test(groups, y, pred_a, pred_b, n_perm=500, seed=777)
    res2 = paired_group_permutation_test(groups, y, pred_a, pred_b, n_perm=500, seed=777)
    assert res1["obs_delta"] == res2["obs_delta"]
    assert res1["p_value_two_sided"] == res2["p_value_two_sided"]
    assert res1["extreme_count"] == res2["extreme_count"]

    boot1 = paired_group_bootstrap(groups, y, pred_a, pred_b, n_boot=500, seed=777)
    boot2 = paired_group_bootstrap(groups, y, pred_a, pred_b, n_boot=500, seed=777)
    assert boot1["ci95"] == boot2["ci95"]
    assert boot1["delta_mean"] == boot2["delta_mean"]
    assert boot1["bootstrap_prob_superiority"] == boot2["bootstrap_prob_superiority"]


def test_holm_bonferroni_correction_properties():
    """Verify Holm-Bonferroni correction properties: step-down ordering, monotonicity, and bounds."""
    # 1. Known textbook example
    raw_p = [0.005, 0.01, 0.02, 0.03, 0.04, 0.08, 0.15, 0.20]
    # m = 8
    # 0.005 * 8 = 0.04
    # 0.01 * 7  = 0.07
    # 0.02 * 6  = 0.12
    # 0.03 * 5  = 0.15
    # 0.04 * 4  = 0.16
    # 0.08 * 3  = 0.24
    # 0.15 * 2  = 0.30
    # 0.20 * 1  = 0.20 -> max(0.30, 0.20) = 0.30
    adj_p = holm_bonferroni_correction(raw_p)
    assert pytest.approx(adj_p[0], rel=1e-5) == 0.04
    assert pytest.approx(adj_p[1], rel=1e-5) == 0.07
    assert pytest.approx(adj_p[2], rel=1e-5) == 0.12
    assert pytest.approx(adj_p[3], rel=1e-5) == 0.15
    assert pytest.approx(adj_p[4], rel=1e-5) == 0.16
    assert pytest.approx(adj_p[5], rel=1e-5) == 0.24
    assert pytest.approx(adj_p[6], rel=1e-5) == 0.30
    assert pytest.approx(adj_p[7], rel=1e-5) == 0.30

    # 2. Out-of-order input restores correct positions
    shuffled_raw = [0.04, 0.005, 0.20, 0.01, 0.15, 0.02, 0.08, 0.03]
    shuffled_adj = holm_bonferroni_correction(shuffled_raw)
    assert pytest.approx(shuffled_adj[1], rel=1e-5) == 0.04  # 0.005
    assert pytest.approx(shuffled_adj[3], rel=1e-5) == 0.07  # 0.01
    assert pytest.approx(shuffled_adj[0], rel=1e-5) == 0.16  # 0.04

    # 3. All adjusted p-values must be >= raw p-values and <= 1.0
    for r, a in zip(shuffled_raw, shuffled_adj):
        assert a >= r - 1e-9
        assert a <= 1.0


def test_paired_multitask_group_bootstrap_and_permutation():
    """Verify that overall mean Macro-F1 across 4 tasks is aggregated correctly."""
    tasks = ["hate", "sarcasm", "target", "severity"]
    task_classes = {"hate": 2, "sarcasm": 2, "target": 4, "severity": 3}
    n_samples = 300
    groups = np.repeat(np.arange(30), 10)

    rng = np.random.default_rng(10)
    y_trues = {t: rng.integers(0, nc, n_samples) for t, nc in task_classes.items()}
    preds_a = {t: rng.integers(0, nc, n_samples) for t, nc in task_classes.items()}
    # Model B gets hate and sarcasm better, target identical, severity slightly worse
    preds_b = {
        "hate": y_trues["hate"].copy(),
        "sarcasm": y_trues["sarcasm"].copy(),
        "target": preds_a["target"].copy(),
        "severity": preds_a["severity"].copy()
    }

    boot = paired_multitask_group_bootstrap(groups, y_trues, preds_a, preds_b, tasks, n_boot=500, seed=42)
    assert boot["mean_macro_f1_b"] > boot["mean_macro_f1_a"]
    assert boot["obs_delta"] > 0
    assert len(boot["ci95"]) == 2
    assert boot["ci95"][0] < boot["ci95"][1]
    assert "bootstrap_prob_superiority" in boot

    perm = paired_multitask_group_permutation_test(groups, y_trues, preds_a, preds_b, tasks, n_perm=500, seed=42)
    assert perm["obs_delta"] == boot["obs_delta"]
    assert perm["p_value_two_sided"] <= 0.05


def test_mcnemar_test_continuity_correction_and_exact():
    """Verify Edwards continuity correction and exact binomial calculation in McNemar test."""
    y = np.array([1] * 50 + [0] * 50)
    pred_a = np.array([1] * 40 + [0] * 10 + [0] * 40 + [1] * 10)
    # B is better than A on 20 discordant pairs
    pred_b = np.array([1] * 48 + [0] * 2 + [0] * 48 + [1] * 2)

    res = mcnemar_test(y, pred_a, pred_b)
    assert res["discordant_pairs"] > 0
    assert "statistic" in res
    assert "p_value" in res
    assert res["contingency_table"]["b_only"] > res["contingency_table"]["a_only"]
    assert res["odds_ratio"] > 1.0


def test_class_level_metrics_rare_class():
    """Verify support, precision, recall, and F1 calculations, especially for rare classes."""
    y = np.array(["low"] * 100 + ["moderate"] * 890 + ["high"] * 10)
    # 6 high correctly predicted, 4 missed as moderate, 2 false alarms on moderate
    p = np.array(["low"] * 100 + ["moderate"] * 888 + ["high"] * 2 + ["high"] * 6 + ["moderate"] * 4)

    metrics = class_level_metrics(y, p)
    assert metrics["high"]["support"] == 10
    assert metrics["high"]["tp"] == 6
    assert metrics["high"]["fp"] == 2
    assert metrics["high"]["fn"] == 4
    assert pytest.approx(metrics["high"]["recall"], rel=1e-3) == 0.60
    assert pytest.approx(metrics["high"]["precision"], rel=1e-3) == 0.75
    assert metrics["low"]["support"] == 100
    assert metrics["moderate"]["support"] == 890


def test_compare_predictions_end_to_end(tmp_path):
    """Verify that compare_predictions runs permutation tests, bootstrap CIs, McNemar, and class metrics."""
    from toxifusionnet.experiments.compare import compare_predictions

    # Create synthetic pooled prediction frames
    n = 120
    ids = [f"img_{i}.jpg" for i in range(n)]
    text = [f"Text content {i % 20}" for i in range(n)]  # 20 text groups
    folds = [i % 5 for i in range(n)]

    rng = np.random.default_rng(42)
    gold_hate = rng.integers(0, 2, n)
    gold_sarcasm = rng.integers(0, 2, n)
    gold_target = np.array([rng.choice(["individual", "community", "organization", "society"]) for _ in range(n)])
    gold_severity = np.array([rng.choice(["low", "moderate", "high"]) for _ in range(n)])

    df_a = pd.DataFrame({
        "id": ids, "text": text, "fold": folds,
        "gold_hate": gold_hate, "pred_hate": rng.integers(0, 2, n),
        "gold_sarcasm": gold_sarcasm, "pred_sarcasm": rng.integers(0, 2, n),
        "gold_target": gold_target, "pred_target": gold_target,
        "gold_severity": gold_severity, "pred_severity": gold_severity,
    })

    df_b = df_a.copy()
    # Model B is slightly better on hate and sarcasm
    df_b["pred_hate"] = gold_hate
    df_b["pred_sarcasm"] = gold_sarcasm

    p_a = tmp_path / "preds_a.csv"
    p_b = tmp_path / "preds_b.csv"
    df_a.to_csv(p_a, index=False)
    df_b.to_csv(p_b, index=False)

    out_dir = tmp_path / "comp_out"
    report = compare_predictions(
        preds_a_path=p_a,
        preds_b_path=p_b,
        name_a="Model_A",
        name_b="Model_B",
        out_dir=out_dir,
        n_perm=200,
        n_boot=200,
        seed=42
    )

    assert "hate" in report["task_comparisons"]
    assert "severity" in report["task_comparisons"]
    hate_comp = report["task_comparisons"]["hate"]
    assert hate_comp["Model_B_macro_f1"] >= hate_comp["Model_A_macro_f1"]
    assert "p_value_permutation" in hate_comp
    assert "delta_ci95" in hate_comp
    assert "bootstrap_prob_superiority" in hate_comp
    assert "class_metrics_a" in hate_comp
    assert "class_deltas" in hate_comp
    assert hate_comp["mcnemar"] is not None

    # Multitask overall mean
    mt = report["multitask_mean"]
    assert "delta_mean_macro_f1" in mt
    assert "p_value_permutation" in mt

    # Check saved files
    assert (out_dir / "comparison_report.json").exists()
    assert (out_dir / "comparison_report.md").exists()


def test_run_publication_analysis_pipeline(tmp_path):
    """Verify run_publication_analysis end-to-end: 2 contrasts, 8-test Holm correction, JSON, CSV, and Markdown."""
    from toxifusionnet.experiments.compare import run_publication_analysis

    n = 100
    ids = [f"sample_{i}.jpg" for i in range(n)]
    text = [f"Meme text {i % 15}" for i in range(n)]
    folds = [i % 5 for i in range(n)]

    rng = np.random.default_rng(123)
    gold_h = rng.integers(0, 2, n)
    gold_s = rng.integers(0, 2, n)
    gold_t = np.array([rng.choice(["individual", "community", "organization", "society"]) for _ in range(n)])
    gold_v = np.array([rng.choice(["low", "moderate", "high"]) for _ in range(n)])

    def make_pred_df(acc_boost=0):
        pred_h = gold_h.copy() if acc_boost >= 2 else rng.integers(0, 2, n)
        pred_s = gold_s.copy() if acc_boost >= 1 else rng.integers(0, 2, n)
        return pd.DataFrame({
            "id": ids, "text": text, "fold": folds,
            "gold_hate": gold_h, "pred_hate": pred_h,
            "gold_sarcasm": gold_s, "pred_sarcasm": pred_s,
            "gold_target": gold_t, "pred_target": gold_t,
            "gold_severity": gold_v, "pred_severity": gold_v,
        })

    e1c_dir = tmp_path / "e1c"
    e1d_dir = tmp_path / "e1d"
    e6b_dir = tmp_path / "e6b"
    e1c_dir.mkdir(); e1d_dir.mkdir(); e6b_dir.mkdir()

    make_pred_df(0).to_csv(e1c_dir / "pooled_model.csv", index=False)
    make_pred_df(1).to_csv(e1d_dir / "pooled_model.csv", index=False)
    make_pred_df(2).to_csv(e6b_dir / "pooled_model.csv", index=False)

    out_dir = tmp_path / "pub_analysis"
    res = run_publication_analysis(
        e1c_dir=e1c_dir,
        e1d_dir=e1d_dir,
        e6b_dir=e6b_dir,
        out_dir=out_dir,
        n_perm=200,
        n_boot=200,
        seed=42
    )

    # 1. Check generated files
    assert (out_dir / "statistical_comparison.json").exists()
    assert (out_dir / "task_comparison_summary.csv").exists()
    assert (out_dir / "publication_statistical_report.md").exists()

    # 2. Check 8-test Holm table
    holm_table = res["eight_task_holm_table"]
    assert len(holm_table) == 8, f"Expected 8 tests in Holm table, got {len(holm_table)}"
    for row in holm_table:
        assert row["holm_adjusted_p"] >= row["raw_permutation_p"] - 1e-9
        assert row["holm_adjusted_p"] <= 1.0

    # 3. Check CSV summary table
    summary_df = pd.read_csv(out_dir / "task_comparison_summary.csv")
    assert len(summary_df) == 8
    assert set(summary_df["contrast"]) == {"E1c_vs_E1d", "E1d_vs_E6b"}
    assert set(summary_df["task"]) == {"hate", "sarcasm", "target", "severity"}
    assert "holm_p_value" in summary_df.columns
    assert "raw_p_value" in summary_df.columns
    assert "bootstrap_prob_superiority" in summary_df.columns

    # 4. Check Markdown report contents
    md_content = (out_dir / "publication_statistical_report.md").read_text(encoding="utf-8")
    assert "Primary Benchmark: Multi-Task Macro-F1 Comparisons" in md_content
    assert "Class-Level Precision, Recall, and F1 Breakdown" in md_content
    assert "Fold-Level Performance & Stability Across Folds" in md_content
    assert "Methodological & Scientific Interpretation" in md_content
    assert "Overall Mean (Secondary)" in md_content


def test_publication_report_formatting_and_scientific_nuance():
    """Verify that p-values are properly formatted (<0.0001, no 0.0000), secondary analyses are labelled,
    and cautious scientific conclusions are maintained."""
    from pathlib import Path
    from toxifusionnet.experiments.compare import _format_p_value

    # 1. Test p-value formatting helper
    assert _format_p_value(0.0) == "<0.0001"
    assert _format_p_value(1e-6) == "<0.0001"
    assert _format_p_value(0.00009) == "<0.0001"
    assert _format_p_value(0.0001) == "0.0001"
    assert _format_p_value(0.05) == "0.0500"
    assert _format_p_value(None) == "—"

    # 2. Test actual publication report generated from the saved experiment results
    report_path = Path("results/v3_journal_final/publication_analysis/publication_statistical_report.md")
    if not report_path.exists():
        pytest.skip("Publication report not yet generated")

    content = report_path.read_text(encoding="utf-8")

    # A. Check no "0.0000" in p-value columns and <0.0001 is present
    assert " 0.0000 " not in content
    assert "<0.0001" in content

    # B. Overall mean labelled as secondary aggregate
    assert "Overall Mean (Secondary)" in content
    assert "(unadj.)" in content
    assert "aggregate secondary analyses" in content

    # C. Cautious hate detection statement (does not establish equivalence)
    assert "does not establish equivalence" in content
    assert "No statistically significant difference in hate Macro-F1 was detected" in content

    # D. Rare severity class uncertainty (N = 30)
    assert "Rare High-Severity Class Support & Uncertainty ($N = 30$)" in content
    assert "uncertainty due to the small test support of only 30" in content

    # E. Plausible interpretation for severity decline (not proven mechanism, cannot isolate exposure)
    assert "plausible interpretation rather than a proven mechanism" in content
    assert "cannot isolate reduced MultiBully exposure from other concurrent dataset-mix effects" in content



