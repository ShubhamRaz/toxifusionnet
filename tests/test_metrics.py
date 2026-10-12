import numpy as np
import pandas as pd

from toxifusionnet.labels import IGNORE_INDEX
from toxifusionnet.training.metrics import end_to_end_metrics, evaluate_predictions, sarcasm_subset_metrics
from toxifusionnet.training.stats import mcnemar, paired_bootstrap


def frame():
    return pd.DataFrame({
        "gold_hate": [1, 1, 1, 1, 0, 0, 0, 0], "pred_hate": [1, 0, 0, 1, 0, 0, 1, 0],
        "gold_sarcasm": [1, 1, 0, 0, 1, 0, 0, 1], "pred_sarcasm": [1, 1, 0, 0, 1, 0, 0, 1],
        "gold_target": [0, 1, 2, 3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
        "pred_target": [0, 1, 0, 3, 0, 0, 2, 0], "gold_severity": [0] * 4 + [IGNORE_INDEX] * 4, "pred_severity": [0] * 8,
        "p_hate": [.9, .2, .4, .8, .1, .2, .7, .3], "p_sarcasm": [.9, .9, .1, .1, .9, .1, .1, .9]})


def test_sarcasm_subset_fnr():
    s = sarcasm_subset_metrics(frame())
    assert s["sarcastic"]["false_negative_rate"] == 0.5 and s["non_sarcastic"]["false_negative_rate"] == 0.5
    assert s["non_sarcastic"]["false_positive_rate"] == 0.5


def test_evaluate_predictions_has_all_sections():
    r = evaluate_predictions(frame())
    assert {"sarcasm", "hate", "target", "severity", "sarcasm_subset", "end_to_end_target"} <= set(r)
    assert r["sarcasm"]["macro_f1"] == 1.0 and 0 < r["hate"]["macro_f1"] < 1
    assert "auroc" in r["hate"] and r["target"]["n"] == 4  # target scored only where gold exists


def test_end_to_end_penalises_missed_hate():
    m = end_to_end_metrics(frame(), "target")
    assert m["classes"][-1] == "none" and m["accuracy"] < 1.0


def test_paired_bootstrap_and_mcnemar_detect_better_system():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = np.where(rng.random(300) < .95, y, 1 - y); bad = np.where(rng.random(300) < .7, y, 1 - y)
    r = paired_bootstrap(y, bad, good, n_boot=300)
    assert r["delta_mean"] > 0.1 and r["p_value"] < 0.05 and r["ci95"][0] > 0
    assert mcnemar(y, bad, good)["p_value"] < 0.01
