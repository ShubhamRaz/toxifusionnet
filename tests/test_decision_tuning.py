"""Unit tests for leakage-safe decision tuning and diagnostic analysis.

Tests:
1. Validation-only threshold selection (no test-set peeking)
2. Tie-breaking toward default 0.50
3. Direct NaN and IGNORE_INDEX handling in tune_binary_threshold
4. Test-set isolation guarantee
5. Insufficient rare-class support guardrail (< min_support)
6. Task-specific guardrail rationale (Target taxonomy vs Severity classes)
7. Missing-label masking (IGNORE_INDEX = -100 and NaN)
8. Cross-validation fold evaluation pipeline on mock data (val_support distinct from test_support)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from toxifusionnet.analysis.decision_tuning import (
    DEFAULT_MIN_SUPPORT,
    check_support_sufficiency,
    evaluate_binary_task_tuning_cv,
    evaluate_multiclass_decision_guardrail,
    tune_binary_threshold,
)
from toxifusionnet.analysis.diagnostic import (
    _format_diagnostic_markdown_report,
    compute_binary_threshold_fractions,
    compute_task_diagnostic,
)
from toxifusionnet.labels import IGNORE_INDEX


def test_tune_binary_threshold_validation_only():
    """Verify that tune_binary_threshold selects the threshold maximizing Macro-F1 on validation data."""
    np.random.seed(42)
    n_pos = 100
    n_neg = 200

    pos_scores = np.random.normal(loc=0.42, scale=0.08, size=n_pos).clip(0.01, 0.99)
    neg_scores = np.random.normal(loc=0.25, scale=0.08, size=n_neg).clip(0.01, 0.99)

    probs = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

    result = tune_binary_threshold(
        val_probs=probs,
        val_labels=labels,
        metric="macro_f1",
        candidate_thresholds=np.linspace(0.1, 0.9, 81),
    )

    assert "best_threshold" in result
    assert "best_val_score" in result
    assert "default_val_score" in result
    assert 0.30 <= result["best_threshold"] <= 0.45
    assert result["best_val_score"] >= result["default_val_score"]


def test_tune_binary_threshold_tie_breaking():
    """Verify that when multiple thresholds achieve the exact same score, tie-breaking selects closest to 0.50."""
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])

    candidates = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    result = tune_binary_threshold(
        val_probs=probs,
        val_labels=labels,
        metric="macro_f1",
        candidate_thresholds=candidates,
    )

    # Closest to 0.50 should be 0.50
    assert result["best_threshold"] == pytest.approx(0.50)


def test_tune_binary_threshold_nan_and_ignore_index_direct():
    """Verify that NaN labels and IGNORE_INDEX are masked before conversion and do not cause exceptions or bias."""
    clean_probs = np.array([0.15, 0.25, 0.45, 0.65, 0.85, 0.90])
    clean_labels = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    # Run on clean data
    clean_result = tune_binary_threshold(
        val_probs=clean_probs,
        val_labels=clean_labels,
        metric="macro_f1",
    )

    # Inject NaNs, IGNORE_INDEX, negative values, and NaN probabilities
    noisy_probs = np.array([0.15, np.nan, 0.25, 0.99, 0.45, 0.65, 0.01, 0.85, 0.90, np.nan])
    noisy_labels = np.array([0.0, 1.0, 0.0, np.nan, 1.0, 1.0, IGNORE_INDEX, 1.0, 1.0, -100])

    # Must NOT raise ValueError: cannot convert float NaN to integer
    noisy_result = tune_binary_threshold(
        val_probs=noisy_probs,
        val_labels=noisy_labels,
        metric="macro_f1",
    )

    # Valid support must equal clean length
    assert noisy_result["val_support"] == len(clean_labels)
    # The selected threshold and best score must be identical to the clean version
    assert noisy_result["best_threshold"] == pytest.approx(clean_result["best_threshold"])
    assert noisy_result["best_val_score"] == pytest.approx(clean_result["best_val_score"])
    assert noisy_result["val_positives"] == 4
    assert noisy_result["val_negatives"] == 2


def test_test_set_isolation():
    """Verify mathematical and algorithmic isolation of the outer test set from threshold selection."""
    np.random.seed(42)
    # Validation data: optimal threshold is around 0.65
    val_pos = np.random.normal(loc=0.75, scale=0.05, size=50).clip(0.01, 0.99)
    val_neg = np.random.normal(loc=0.55, scale=0.05, size=50).clip(0.01, 0.99)
    val_probs = np.concatenate([val_pos, val_neg])
    val_labels = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])

    # Test data: distribution is arbitrarily shifted such that 0.20 would be optimal on test
    test_pos = np.random.normal(loc=0.35, scale=0.05, size=50).clip(0.01, 0.99)
    test_neg = np.random.normal(loc=0.10, scale=0.05, size=50).clip(0.01, 0.99)
    test_probs = np.concatenate([test_pos, test_neg])
    test_labels = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])

    # Tuning strictly on validation
    val_tuning = tune_binary_threshold(
        val_probs=val_probs,
        val_labels=val_labels,
        metric="macro_f1",
        candidate_thresholds=np.linspace(0.1, 0.9, 81),
    )
    chosen_tau = val_tuning["best_threshold"]

    # Verify that the chosen threshold reflects the validation distribution (~0.60-0.70), NOT the test distribution (~0.20-0.30)
    assert chosen_tau >= 0.55
    assert chosen_tau != pytest.approx(0.20, abs=0.10)

    # Evaluate chosen_tau on test data (clean out-of-fold application)
    test_preds = (test_probs >= chosen_tau).astype(int)
    assert len(test_preds) == len(test_labels)


def test_check_support_sufficiency_rare_class_guardrail():
    """Verify that rare classes (< min_support) trigger the safety guardrail."""
    # Case 1: Insufficient support (simulating High Severity with N=3)
    labels_insufficient = np.array([0] * 50 + [1] * 100 + [2] * 3)
    supp_info = check_support_sufficiency(labels_insufficient, min_support=20)

    assert not supp_info["is_sufficient"]
    assert 2 in supp_info["insufficient_classes"]
    assert supp_info["insufficient_details"][2] == 3
    assert supp_info["class_counts"][2] == 3

    # Case 2: Sufficient support across all classes
    labels_sufficient = np.array([0] * 30 + [1] * 40 + [2] * 25)
    supp_info_ok = check_support_sufficiency(labels_sufficient, min_support=20)

    assert supp_info_ok["is_sufficient"]
    assert len(supp_info_ok["insufficient_classes"]) == 0
    assert min(supp_info_ok["class_counts"].values()) == 25


def test_missing_label_masking():
    """Verify that IGNORE_INDEX (-100) and NaN labels are properly masked out in diagnostics and tuning."""
    df = pd.DataFrame({
        "gold_hate": [1, 0, IGNORE_INDEX, 1, np.nan, 0],
        "pred_hate": [1, 0, 1, 0, 0, 0],
        "p_hate": [0.85, 0.15, 0.90, 0.40, 0.10, 0.20],
    })

    # Diagnostic check
    diag = compute_task_diagnostic(df, "hate", class_names=["non-hate", "hate"])
    assert diag["n_total"] == 6
    assert diag["n_masked"] == 2
    assert diag["n_eval"] == 4
    # Evaluated indices are 0, 1, 3, 5: golds are [1, 0, 1, 0], preds are [1, 0, 0, 0]
    # TP=1, TN=2, FP=0, FN=1 -> Accuracy = 3/4 = 0.75
    assert diag["accuracy"] == pytest.approx(0.75)

    # Binary score distribution check inside diagnostic
    assert "score_distribution" in diag
    score_dist = diag["score_distribution"]
    assert score_dist["overall_quantiles"]["p50"] > 0
    assert score_dist["brier_score"] >= 0


def test_evaluate_multiclass_guardrail_task_specific_explanations(tmp_path):
    """Verify task-specific guardrail rationales: target taxonomy vs severity classes."""
    exp_dir = tmp_path / "mock_exp"
    exp_dir.mkdir()

    # Create mock 2-fold cross validation structure (fold0, fold1)
    for f in range(2):
        fold_dir = exp_dir / f"fold{f}"
        fold_dir.mkdir()

        # Validation fold with rare target and severity classes (length = 84)
        val_df = pd.DataFrame({
            "gold_severity": [0] * 30 + [1] * 50 + [2] * 4,
            "pred_severity": [0] * 30 + [1] * 50 + [1] * 4,
            "gold_target": [0] * 45 + [1] * 5 + [2] * 30 + [3] * 4,
            "pred_target": [0] * 45 + [0] * 5 + [2] * 30 + [0] * 4,
        })
        val_df.to_csv(fold_dir / "preds_val.csv", index=False)

        # Test fold (length = 84)
        test_df = pd.DataFrame({
            "gold_severity": [0] * 30 + [1] * 50 + [2] * 4,
            "pred_severity": [0] * 30 + [1] * 50 + [1] * 4,
            "gold_target": [0] * 45 + [1] * 5 + [2] * 30 + [3] * 4,
            "pred_target": [0] * 45 + [0] * 5 + [2] * 30 + [0] * 4,
        })
        test_df.to_csv(fold_dir / "preds_test.csv", index=False)

    pooled_test = pd.concat([
        pd.read_csv(exp_dir / "fold0" / "preds_test.csv"),
        pd.read_csv(exp_dir / "fold1" / "preds_test.csv"),
    ], ignore_index=True)
    pooled_test.to_csv(exp_dir / "pooled_model.csv", index=False)

    # 1. Target guardrail
    tgt_guard = evaluate_multiclass_decision_guardrail(
        exp_dir=exp_dir, task="target", n_folds=2, min_support=20
    )
    assert tgt_guard["action_taken"] == "RETAIN_DEFAULT_ARGMAX"
    assert tgt_guard["has_insufficient_support"]
    tgt_rationale = tgt_guard["scientific_rationale"].lower()
    # Must preserve MultiBully taxonomy
    assert "individual" in tgt_rationale
    assert "organization" in tgt_rationale
    assert "community" in tgt_rationale
    assert "society" in tgt_rationale
    # Must NOT mention high severity
    assert "severity" not in tgt_rationale
    assert "high" not in tgt_rationale

    # 2. Severity guardrail
    sev_guard = evaluate_multiclass_decision_guardrail(
        exp_dir=exp_dir, task="severity", n_folds=2, min_support=20
    )
    assert sev_guard["action_taken"] == "RETAIN_DEFAULT_ARGMAX"
    assert sev_guard["has_insufficient_support"]
    sev_rationale = sev_guard["scientific_rationale"].lower()
    assert "severity" in sev_rationale
    assert "high" in sev_rationale
    assert "low" in sev_rationale


def test_cv_decision_tuning_val_support_distinct_from_test_support(tmp_path):
    """Verify that val_support is accurately stored and distinct from test_support across folds."""
    exp_dir = tmp_path / "mock_cv_exp"
    exp_dir.mkdir()

    n_val = 45
    n_test = 75

    for f in range(2):
        fold_dir = exp_dir / f"fold{f}"
        fold_dir.mkdir()

        # Val fold
        np.random.seed(100 + f)
        val_golds = np.random.choice([0, 1], size=n_val)
        val_probs = np.where(val_golds == 1, np.random.uniform(0.4, 0.8, size=n_val), np.random.uniform(0.1, 0.5, size=n_val))
        val_df = pd.DataFrame({
            "gold_sarcasm": val_golds,
            "pred_sarcasm": (val_probs >= 0.5).astype(int),
            "p_sarcasm": val_probs,
            "text": [f"val_text_{i}" for i in range(n_val)],
        })
        val_df.to_csv(fold_dir / "preds_val.csv", index=False)

        # Test fold
        test_golds = np.random.choice([0, 1], size=n_test)
        test_probs = np.where(test_golds == 1, np.random.uniform(0.4, 0.8, size=n_test), np.random.uniform(0.1, 0.5, size=n_test))
        test_df = pd.DataFrame({
            "gold_sarcasm": test_golds,
            "pred_sarcasm": (test_probs >= 0.5).astype(int),
            "p_sarcasm": test_probs,
            "text": [f"test_text_{f}_{i}" for i in range(n_test)],
        })
        test_df.to_csv(fold_dir / "preds_test.csv", index=False)

    tuning_out = evaluate_binary_task_tuning_cv(
        exp_dir=exp_dir,
        task="sarcasm",
        n_folds=2,
        metric="macro_f1",
        n_perm=20,
        n_boot=20,
    )

    assert tuning_out["task"] == "sarcasm"
    assert tuning_out["n_folds"] == 2
    assert len(tuning_out["fold_results"]) == 2

    for fr in tuning_out["fold_results"]:
        assert fr["val_support"] == n_val
        assert fr["test_support"] == n_test
        assert fr["val_support"] != fr["test_support"]


def test_compute_binary_threshold_fractions_synthetic():
    """Verify compute_binary_threshold_fractions calculation logic and edge cases."""
    # 1. Edge case: empty input
    res_empty = compute_binary_threshold_fractions([])
    assert res_empty["pos_total"] == 0
    assert res_empty["below_default"]["numerator"] == 0
    assert res_empty["below_default"]["denominator"] == 0
    assert res_empty["below_default"]["fraction"] == 0.0
    assert res_empty["below_default"]["percentage"] == 0.0
    assert res_empty["in_interval"]["numerator"] == 0
    assert res_empty["in_interval"]["denominator"] == 0
    assert res_empty["in_interval"]["fraction"] == 0.0
    assert res_empty["in_interval"]["percentage"] == 0.0

    # 2. Known distribution:
    # 10 values: [0.10, 0.20, 0.35, 0.36, 0.40, 0.45, 0.49, 0.50, 0.60, 0.80]
    # Total = 10
    # p < 0.50: 7 values -> 7/10 = 0.70 (70.0%)
    # 0.36 <= p < 0.50: 4 values [0.36, 0.40, 0.45, 0.49] -> 4/10 = 0.40 (40.0%)
    scores = [0.10, 0.20, 0.35, 0.36, 0.40, 0.45, 0.49, 0.50, 0.60, 0.80]
    res = compute_binary_threshold_fractions(scores, lower_threshold=0.36, default_threshold=0.50)
    assert res["pos_total"] == 10
    assert res["below_default"]["numerator"] == 7
    assert res["below_default"]["denominator"] == 10
    assert res["below_default"]["fraction"] == 0.7
    assert res["below_default"]["percentage"] == 70.0

    assert res["in_interval"]["numerator"] == 4
    assert res["in_interval"]["denominator"] == 10
    assert res["in_interval"]["fraction"] == 0.4
    assert res["in_interval"]["percentage"] == 40.0


def test_sarcasm_threshold_fractions_exact_calculation():
    """Verify exact calculation of true sarcastic predictions below 0.50 and in [0.36, 0.50)."""
    e1d_csv = "results/v3_journal_final/e1d_text_weighted/pooled_model.csv"
    df = pd.read_csv(e1d_csv)
    diag = compute_task_diagnostic(df, "sarcasm", class_names=["non-sarcastic", "sarcastic"])

    assert "score_distribution" in diag
    sd = diag["score_distribution"]
    assert "threshold_fractions" in sd
    tf = sd["threshold_fractions"]

    # True positive support is 2,129
    assert tf["pos_total"] == 2129

    b50 = tf["below_default"]
    assert b50["numerator"] == 1306
    assert b50["denominator"] == 2129
    assert b50["fraction"] == pytest.approx(1306 / 2129, abs=1e-4)
    assert b50["percentage"] == pytest.approx(61.34, abs=0.01)
    # Confirm that > 50% have p < 0.50
    assert b50["fraction"] > 0.50

    inter = tf["in_interval"]
    assert inter["numerator"] == 765
    assert inter["denominator"] == 2129
    assert inter["fraction"] == pytest.approx(765 / 2129, abs=1e-4)
    assert inter["percentage"] == pytest.approx(35.93, abs=0.01)
    # Confirm that [0.36, 0.50) is < 50% (35.93%), disproving the old claim that > 50% were in [0.36, 0.50)
    assert inter["fraction"] < 0.50


def test_regenerated_diagnostic_report_markdown_and_dynamic_prose():
    """Verify that regenerated markdown report removes unsupported claim and dynamically includes exact fractions."""
    report_md_path = "results/v3_journal_final/diagnostic_analysis/diagnostic_report.md"
    with open(report_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # 1. Unsupported claim must NOT be present
    assert "more than half of true sarcastic posts receive predicted probabilities in the range `[0.36, 0.50)`" not in md_text
    assert "more than half of true sarcastic posts receive predicted probabilities in the range `[0.36, 0.50)" not in md_text

    # 2. Correct dynamic numbers and percentages must be present
    assert "1,306 of 2,129 true sarcastic examples (61.34%, fraction = 0.6134) have $p_{\\text{sarcasm}} < 0.50$" in md_text
    assert "765 of 2,129 true sarcastic examples (35.93%, fraction = 0.3593)" in md_text

    # 3. Dynamic formatting check with mock report data
    mock_report = {
        "experiments": {
            "E1c_unweighted": {"tasks": {"hate": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}, "sarcasm": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}, "target": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}, "severity": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}}, "overall_mean_macro_f1": 0.5, "rare_class_diagnostics": {}, "end_to_end_conditional": {}},
            "E1d_weighted": {
                "tasks": {
                    "hate": {"n_eval": 10, "macro_f1": 0.6, "accuracy": 0.6, "balanced_accuracy": 0.6, "score_distribution": {"overall_mean": 0.5, "overall_std": 0.1, "pos_quantiles": {"p25": 0.4, "p50": 0.6, "p75": 0.8}, "neg_quantiles": {"p25": 0.2, "p50": 0.3, "p75": 0.4}, "mean_separation": 0.3, "auroc": 0.8, "brier_score": 0.1}},
                    "sarcasm": {
                        "n_eval": 100, "macro_f1": 0.55, "accuracy": 0.55, "balanced_accuracy": 0.55,
                        "score_distribution": {
                            "overall_mean": 0.45, "overall_std": 0.1,
                            "pos_quantiles": {"p25": 0.38, "p50": 0.48, "p75": 0.58},
                            "neg_quantiles": {"p25": 0.28, "p50": 0.38, "p75": 0.48},
                            "mean_separation": 0.1, "auroc": 0.65, "brier_score": 0.2,
                            "threshold_fractions": {
                                "pos_total": 50,
                                "below_default": {"numerator": 31, "denominator": 50, "fraction": 0.62, "percentage": 62.0},
                                "in_interval": {"lower_threshold": 0.36, "upper_threshold": 0.50, "numerator": 18, "denominator": 50, "fraction": 0.36, "percentage": 36.0},
                            }
                        }
                    },
                    "target": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5},
                    "severity": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5},
                },
                "overall_mean_macro_f1": 0.55,
                "rare_class_diagnostics": {},
                "end_to_end_conditional": {},
            },
            "E6b_combined_gold": {"tasks": {"hate": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}, "sarcasm": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}, "target": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}, "severity": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}}, "overall_mean_macro_f1": 0.5, "rare_class_diagnostics": {}, "end_to_end_conditional": {}},
            "TFIDF_LogReg": {"tasks": {"hate": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}, "sarcasm": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}, "target": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}, "severity": {"n_eval": 10, "macro_f1": 0.5, "accuracy": 0.5, "balanced_accuracy": 0.5}}, "overall_mean_macro_f1": 0.5},
        }
    }
    formatted = _format_diagnostic_markdown_report(mock_report)
    assert "31 of 50 true sarcastic examples (62.00%, fraction = 0.6200) have $p_{\\text{sarcasm}} < 0.50$" in formatted
    assert "18 of 50 true sarcastic examples (36.00%, fraction = 0.3600)" in formatted

