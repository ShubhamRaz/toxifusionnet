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
from toxifusionnet.analysis.diagnostic import compute_task_diagnostic
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
