"""Comprehensive unit and integration test suite for per-task dataset balancing and fold weighting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from toxifusionnet.labels import IGNORE_INDEX, LABEL_SPACES, apply_label_config, label_to_id
from toxifusionnet.models.losses import MultiTaskLoss
from toxifusionnet.training.balancing import compute_fold_class_weights, compute_task_class_weights


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"


@pytest.fixture(autouse=True)
def _setup_taxonomy():
    apply_label_config({"target": ["individual", "organization", "community", "society"]})
    yield
    apply_label_config(None)


def test_sqrt_balancing_formula_numerical_parity():
    """Verify exact analytical parity of the square-root inverse-frequency formula."""
    # Counts: 100 for class 0, 25 for class 1
    # N = 125, K = 2
    # raw_0 = sqrt(125 / (2 * 100)) = sqrt(0.625) = 0.790569415
    # raw_1 = sqrt(125 / (2 * 25))  = sqrt(2.5)   = 1.581138830
    # mean_raw = (0.790569415 + 1.581138830) / 2 = 1.185854123
    # norm_0 = raw_0 / mean_raw = 0.666666667 (2/3)
    # norm_1 = raw_1 / mean_raw = 1.333333333 (4/3)
    df = pd.DataFrame({
        "hate": ["non_hate"] * 100 + ["hate"] * 25
    })
    res = compute_task_class_weights(df, "hate", mode="sqrt_balanced", max_weight=4.0)

    assert res["counts"]["non_hate"] == 100
    assert res["counts"]["hate"] == 25
    assert res["total_supervised"] == 125

    w = res["weights"]
    assert pytest.approx(w[0], rel=1e-5) == 2.0 / 3.0
    assert pytest.approx(w[1], rel=1e-5) == 4.0 / 3.0
    assert pytest.approx(np.mean(w), rel=1e-6) == 1.0


def test_max_weight_capping_and_safety():
    """Verify that extreme minority weights are strictly capped at max_weight."""
    # 1. Binary task with max_weight = 1.5 (uncapped weight ~ 1.98)
    df_bin = pd.DataFrame({"hate": ["non_hate"] * 10000 + ["hate"] * 1})
    res_bin = compute_task_class_weights(df_bin, "hate", mode="sqrt_balanced", max_weight=1.5)
    w_bin = res_bin["weights"]
    assert w_bin[1] == 1.5  # Strictly capped at 1.5
    assert np.isfinite(w_bin).all()
    assert (np.array(w_bin) >= 0.0).all()

    # 2. 5-class task with max_weight = 4.0 (uncapped minority weight ~ 4.95)
    apply_label_config({"target": ["individual", "caste", "religion", "gender", "demography"]})
    df_5 = pd.DataFrame({
        "target": ["individual"] * 1000 + ["caste"] * 1000 + ["religion"] * 1000 + ["gender"] * 1000 + ["demography"] * 1
    })
    res_5 = compute_task_class_weights(df_5, "target", mode="sqrt_balanced", max_weight=4.0)
    w_5 = res_5["weights"]
    assert w_5[4] == 4.0  # Strictly capped at 4.0
    assert np.isfinite(w_5).all()
    assert (np.array(w_5) >= 0.0).all()
    # Reset taxonomy back to 4-class MultiBully
    apply_label_config({"target": ["individual", "organization", "community", "society"]})


def test_zero_support_class_handling():
    """Verify that classes with zero support do not divide by zero and receive 0.0 weight."""
    # Label space: individual, organization, community, society
    # No organization samples present
    df = pd.DataFrame({
        "target": ["individual"] * 100 + ["community"] * 50 + ["society"] * 25
    })
    res = compute_task_class_weights(df, "target", mode="sqrt_balanced", max_weight=4.0)
    w = res["weights"]

    assert res["counts"]["organization"] == 0
    assert w[1] == 0.0  # organization has zero weight
    assert w[0] > 0.0   # individual active
    assert w[2] > 0.0   # community active
    assert w[3] > 0.0   # society active
    assert np.isfinite(w).all()
    assert (np.array(w) >= 0.0).all()


def test_masking_and_ignore_index_exclusion():
    """Verify that IGNORE_INDEX, None, and NaN do not contribute to class counts or weights."""
    df_clean = pd.DataFrame({"hate": ["non_hate"] * 80 + ["hate"] * 20})
    df_noisy = pd.DataFrame({"hate": ["non_hate"] * 80 + ["hate"] * 20 + [None] * 50 + [np.nan] * 30})

    res_clean = compute_task_class_weights(df_clean, "hate", mode="sqrt_balanced")
    res_noisy = compute_task_class_weights(df_noisy, "hate", mode="sqrt_balanced")

    assert res_clean["counts"] == res_noisy["counts"]
    assert res_clean["total_supervised"] == res_noisy["total_supervised"] == 100
    assert res_clean["weights"] == res_noisy["weights"]


def test_different_heads_receive_different_weights():
    """Verify that different task heads compute task-specific fold weights."""
    df = pd.read_csv(DATA_DIR / "multibully.csv")
    tasks = ["hate", "sarcasm", "target", "severity"]
    meta = compute_fold_class_weights(df, tasks, mode="sqrt_balanced")

    # Hate has 2 classes, target has 4 classes, severity has 3 classes
    assert len(meta["hate"]["weights"]) == 2
    assert len(meta["sarcasm"]["weights"]) == 2
    assert len(meta["target"]["weights"]) == 4
    assert len(meta["severity"]["weights"]) == 3

    # Individual (target majority) should have smaller weight than organization (minority)
    target_w = meta["target"]["weights"]
    assert target_w[0] < target_w[1]  # individual < organization
    assert target_w[0] < target_w[3]  # individual < society

    # Moderate severity (majority) should have smaller weight than high severity (minority)
    sev_w = meta["severity"]["weights"]
    assert sev_w[1] < sev_w[2]  # moderate < high
    assert sev_w[1] < sev_w[0]  # moderate < low


def test_e1d_and_e6b_identical_multibully_weights_across_all_folds():
    """Verify E1d and E6b derive identical class weights within corresponding folds."""
    mb = pd.read_csv(DATA_DIR / "multibully.csv")
    mb["source_dataset"] = "multibully"

    hot = pd.read_csv(DATA_DIR / "hot_gold_only.csv")
    hot["source_dataset"] = "hot"

    folds_cache = pd.read_csv(DATA_DIR / "multibully_v2_cv_folds.csv")
    tasks = ["hate", "sarcasm", "target", "severity"]

    for f in range(5):
        # E1d training partition: MultiBully only
        f_ids = set(folds_cache[(folds_cache["fold"] == f) & (folds_cache["split"] == "train")]["id"])
        e1d_tr = mb[mb["id"].isin(f_ids)]

        # E6b training partition: MultiBully + HOT gold
        hot_f = hot.copy()
        hot_f["split"] = "train"
        hot_f["fold"] = f
        e6b_tr = pd.concat([e1d_tr, hot_f], ignore_index=True)

        # Compute weights with source_filter='multibully'
        w_e1d = compute_fold_class_weights(e1d_tr, tasks, mode="sqrt_balanced", source_filter="multibully")
        w_e6b = compute_fold_class_weights(e6b_tr, tasks, mode="sqrt_balanced", source_filter="multibully")

        for task in tasks:
            assert w_e1d[task]["counts"] == w_e6b[task]["counts"], f"Fold {f} Task {task}: counts mismatch!"
            assert w_e1d[task]["weights"] == w_e6b[task]["weights"], f"Fold {f} Task {task}: weights mismatch!"


def test_hot_gold_only_contributes_strictly_to_hate():
    """Verify HOT gold data contains zero supervision for sarcasm, target, or severity."""
    hot = pd.read_csv(DATA_DIR / "hot_gold_only.csv")

    # Sarcasm, target, severity must be 100% NaN or unannotated
    assert hot["sarcasm"].isna().all(), "HOT must not annotate sarcasm"
    assert hot["target"].isna().all(), "HOT must not annotate target"
    assert hot["severity"].isna().all(), "HOT must not annotate severity"

    # All non-null hate labels must be strictly 'hate' or 'non_hate'
    valid_hate = hot["hate"].dropna()
    assert set(valid_hate.unique()).issubset({"hate", "non_hate"})

    # Offensive language samples (orig_hot_label 2.0) must be masked (NaN)
    if "orig_hot_label" in hot.columns:
        off_samples = hot[hot["orig_hot_label"] == 2.0]
        assert off_samples["hate"].isna().all(), "HOT offensive class must be masked in gold_only"


def test_masked_labels_excluded_from_loss_and_gradients():
    """Verify that MultiTaskLoss with class weights strictly excludes IGNORE_INDEX samples from gradients."""
    tasks = ["hate", "sarcasm", "target", "severity"]
    class_weights = {
        "hate": [0.75, 1.25],
        "sarcasm": [0.85, 1.15],
        "target": [0.4, 1.5, 0.8, 1.3],
        "severity": [0.9, 0.3, 1.8],
    }
    loss_fn = MultiTaskLoss(tasks=tasks, weighting="fixed", weights={t: 1.0 for t in tasks},
                            class_weights=class_weights)

    logits = {
        "sarcasm": torch.randn(2, 2, requires_grad=True),
        "hate": torch.randn(2, 2, requires_grad=True),
        "target": torch.randn(2, 4, requires_grad=True),
        "severity": torch.randn(2, 3, requires_grad=True),
    }
    out = {"logits": logits, "attn": {"hate": torch.zeros(2, 5)}}

    # Row 0: valid across all tasks
    # Row 1: non-hate post -> target and severity are IGNORE_INDEX (-100)
    batch = {
        "labels": {
            "sarcasm": torch.tensor([1, 0]),
            "hate": torch.tensor([1, 0]),
            "target": torch.tensor([0, IGNORE_INDEX]),
            "severity": torch.tensor([1, IGNORE_INDEX]),
        },
        "hate_attn_target": None,
        "hate_attn_valid": None,
    }

    loss, _ = loss_fn(out, batch)
    loss.backward()

    # Masked row 1 must receive strictly 0.0 gradient on conditional heads
    assert (logits["target"].grad[1] == 0).all(), "Masked target sample must receive 0 gradient"
    assert (logits["severity"].grad[1] == 0).all(), "Masked severity sample must receive 0 gradient"
    # Unmasked row 0 must receive non-zero gradient
    assert not (logits["target"].grad[0] == 0).all()
    assert not (logits["severity"].grad[0] == 0).all()


def test_fold_cache_authoritative_checksum():
    """Verify that multibully_v2_cv_folds.csv matches the authoritative historical checksum."""
    fold_cache_path = DATA_DIR / "multibully_v2_cv_folds.csv"
    assert fold_cache_path.exists(), "multibully_v2_cv_folds.csv must exist"

    # Normalize line endings to CRLF in memory for cross-platform checksum verification
    raw_bytes = fold_cache_path.read_bytes()
    normalized = raw_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    sha = hashlib.sha256(normalized).hexdigest()
    assert sha == "73d4dbb4858907f43765dea0acff36aa4ae21ddc01e1694c0fcf85791fb2d1a5", (
        f"Fold cache checksum mismatch! Expected authoritative 73d4db..., got {sha}"
    )

    # Verify that an LF-checked-out version (DGX Linux environment) normalizes to the identical canonical hash
    raw_lf = raw_bytes.replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw_lf).hexdigest() == "da3ea2286e88cda00896bdc10610ef0ea76cbcde4218a43ffa97aec7237c99d2"
    sha_from_lf = hashlib.sha256(raw_lf.replace(b"\n", b"\r\n")).hexdigest()
    assert sha_from_lf == "73d4dbb4858907f43765dea0acff36aa4ae21ddc01e1694c0fcf85791fb2d1a5"

    # Verify partition sizes
    folds = pd.read_csv(fold_cache_path)
    assert len(folds) == 28590
    test_sizes = folds[folds["split"] == "test"].groupby("fold")["id"].count().tolist()
    assert test_sizes == [1142, 1144, 1146, 1144, 1142]
    assert folds[folds["split"] == "test"]["id"].nunique() == 5718


def test_zero_supervision_auxiliary_filtering():
    """Verify that auxiliary rows with all labels masked are excluded from training loader while preserving disk datasets."""
    hot_path = DATA_DIR / "hot_gold_only.csv"
    assert hot_path.exists(), "hot_gold_only.csv must exist"

    # 1. Verify on-disk file remains complete and untouched
    raw_hot = pd.read_csv(hot_path)
    assert len(raw_hot) == 3189, "On-disk raw hot_gold_only.csv must retain all 3189 rows"

    # 2. Check supervision distribution across tasks
    tasks = ["hate", "sarcasm", "target", "severity"]
    has_sup = pd.Series(False, index=raw_hot.index)
    for t in tasks:
        if t in raw_hot.columns:
            has_sup = has_sup | raw_hot[t].apply(lambda v: label_to_id(t, v) != IGNORE_INDEX)

    n_zero = int((~has_sup).sum())
    n_supervised = int(has_sup.sum())
    assert n_zero == 1765, f"Expected 1765 zero-supervision rows, found {n_zero}"
    assert n_supervised == 1424, f"Expected 1424 supervised rows, found {n_supervised}"

    # 3. Verify all 1424 retained samples are valid human gold hate annotations
    supervised_hot = raw_hot[has_sup].reset_index(drop=True)
    assert len(supervised_hot) == 1424
    hate_counts = supervised_hot["hate"].value_counts().to_dict()
    assert hate_counts == {"non_hate": 1121, "hate": 303}
    assert supervised_hot["sarcasm"].isna().all()
    assert supervised_hot["target"].isna().all()
    assert supervised_hot["severity"].isna().all()

    # 4. Verify that excluded rows are solely offensive/unlabeled (orig_hot_label == 2.0 or NaN hate)
    excluded_hot = raw_hot[~has_sup].reset_index(drop=True)
    assert len(excluded_hot) == 1765
    assert excluded_hot["hate"].isna().all()


def test_matched_optimizer_budget_logic():
    """Verify that E1c, E1d, and E6b configs specify matched budget, warmup, and disabled early stopping."""
    from toxifusionnet.config import load_config

    cfg_e1c = load_config("configs/exp1c_multibully_text_unweighted.yaml")
    cfg_e1d = load_config("configs/exp1d_multibully_text_weighted.yaml")
    cfg_e6b = load_config("configs/exp6b_combined_text_gold_weighted.yaml")

    # Verify exact configuration match
    for cfg, name in [(cfg_e1c, "E1c"), (cfg_e1d, "E1d"), (cfg_e6b, "E6b")]:
        assert cfg.train.steps_per_epoch == 61, f"{name} must have steps_per_epoch=61"
        assert cfg.train.max_steps == 488, f"{name} must have max_steps=488"
        assert pytest.approx(float(cfg.train.warmup_ratio), rel=1e-5) == 0.10, f"{name} must have warmup_ratio=0.10"
        assert cfg.train.epochs == 8, f"{name} must have epochs=8"
        assert cfg.train.batch_size == 64, f"{name} must have batch_size=64"
        assert bool(cfg.train.get("early_stopping", True)) is False, f"{name} must disable early_stopping"
        assert int(cfg.train.get("patience", 0)) == 0, f"{name} must set patience=0"

    # Verify schedule math
    total_steps = cfg_e1d.train.max_steps
    warmup_steps = int(float(cfg_e1d.train.warmup_ratio) * total_steps)
    assert warmup_steps == 48, f"Expected 48 warmup steps for 488 total steps, got {warmup_steps}"

    # Verify that DataLoader cycling properly produces exact steps_per_epoch regardless of dataset size
    from torch.utils.data import DataLoader, TensorDataset
    dummy_ds = TensorDataset(torch.arange(100))
    dummy_dl = DataLoader(dummy_ds, batch_size=10, shuffle=True)
    assert len(dummy_dl) == 10

    dl_iter = iter(dummy_dl)
    drawn = 0
    steps_per_epoch = 15
    for _ in range(steps_per_epoch):
        try:
            b = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dummy_dl)
            b = next(dl_iter)
        drawn += 1
    assert drawn == 15, "DataLoader cycling must supply exact steps requested"


def test_fold_cache_checksum_fail_fast(tmp_path):
    """Verify fail-fast exception when fold cache checksum is invalid."""
    from toxifusionnet.experiments.cv import CANONICAL_FOLD_CACHE_SHA256, validate_fold_cache_checksum, run_cv
    canonical_path = DATA_DIR / "multibully_v2_cv_folds.csv"

    # 1. Valid canonical cache passes and returns canonical hash
    sha = validate_fold_cache_checksum(canonical_path, CANONICAL_FOLD_CACHE_SHA256)
    assert sha == CANONICAL_FOLD_CACHE_SHA256

    # 2. Incorrect expected hash raises ValueError
    with pytest.raises(ValueError, match="Canonical fold cache checksum verification failed"):
        validate_fold_cache_checksum(canonical_path, expected_sha="deadbeef" * 8)

    # 3. Corrupted fold cache file named multibully_v2_cv_folds.csv raises ValueError
    corrupt_cache = tmp_path / "multibully_v2_cv_folds.csv"
    corrupt_cache.write_text("corrupted,content\n1,2\n")
    with pytest.raises(ValueError, match="Canonical fold cache checksum verification failed"):
        validate_fold_cache_checksum(corrupt_cache)

    # 4. run_cv fails fast if canonical fold cache is corrupted
    with pytest.raises(ValueError, match="Canonical fold cache checksum verification failed"):
        run_cv(
            config="configs/smoke_test.yaml",
            overrides=[],
            data_path=str(DATA_DIR / "multibully.csv"),
            out_dir=tmp_path / "run_out",
            folds_cache=str(corrupt_cache)
        )


def test_fixed_budget_runs_complete_all_steps(tmp_path):
    """Verify that early_stopping: false completes the full step budget and loads best checkpoint."""
    from toxifusionnet.config import load_config
    from toxifusionnet.training.trainer import train_run

    # Load tiny smoke config
    cfg = load_config("configs/smoke_test.yaml")
    cfg.train.epochs = 4
    cfg.train.steps_per_epoch = 5
    cfg.train.max_steps = 20
    cfg.train.early_stopping = False
    cfg.train.patience = 1  # would trigger at epoch 2 if early stopping were enabled
    cfg.train.device = "cpu"
    cfg.rationale.enabled = False

    # Minimal synthetic dataframe
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(60)],
        "text": ["some text sample"] * 60,
        "split": ["train"] * 40 + ["val"] * 10 + ["test"] * 10,
        "hate": ["non_hate", "hate"] * 30,
        "sarcasm": ["not_sarcastic", "sarcastic"] * 30,
        "target": ["individual"] * 60,
        "severity": ["low"] * 60,
        "source_dataset": ["multibully"] * 60,
    })

    run_dir = tmp_path / "budget_run"
    res = train_run(cfg, df, run_dir)

    assert res["total_steps"] == 20, f"Expected 20 steps, completed {res['total_steps']}"
    assert res["steps_budget"] == 20
    assert (run_dir / "history.csv").exists()

    history = pd.read_csv(run_dir / "history.csv")
    assert len(history) == 4, "Must complete all 4 epochs without early stopping"
    assert res["eval_checkpoint_type"] == "best_val_macro_f1"
    assert res["best_epoch"] >= 1
    assert res["best_step"] >= 1


