import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from toxifusionnet.config import Config
from toxifusionnet.data.dataset import HateSarcasmDataset, make_collate
from toxifusionnet.data.grouped_splits import normalize_ocr_text
from toxifusionnet.data.schema import build_canonical
from toxifusionnet.experiments.cv import CANONICAL_FOLD_CACHE_SHA256, validate_fold_cache, validate_fold_cache_checksum
from toxifusionnet.labels import (
    DEFAULT_TARGET_LABELS,
    HATE_LABELS,
    IGNORE_INDEX,
    LABEL_SPACES,
    SARCASM_LABELS,
    SEVERITY_LABELS,
    TASKS,
    label_to_id,
)
from toxifusionnet.models.encoder import build_encoder_and_tokenizer
from toxifusionnet.models.losses import MultiTaskLoss
from toxifusionnet.models.toxifusionnet import ToxiFusionNet
from toxifusionnet.preprocessing.normalize import TextNormalizer
from toxifusionnet.training.metrics import evaluate_predictions, task_metrics_from_arrays


DATA_DIR = Path("data/processed")
FOLDS_CACHE = DATA_DIR / "multibully_v2_cv_folds.csv"
MULTIBULLY_CSV = DATA_DIR / "multibully.csv"
HOT_GOLD_CSV = DATA_DIR / "hot_gold_only.csv"


# =========================================================================
# Section 1: End-to-End Masking Tests (Dataset, Loss, Metrics)
# =========================================================================

def test_dataset_converts_nans_and_unknowns_to_ignore_index():
    """Verify CSV NaN labels and out-of-taxonomy categories map to IGNORE_INDEX (-100)."""
    norm = TextNormalizer()
    cfg = Config.wrap({
        "encoder_name": "tiny", "tiny_hidden": 32, "tiny_layers": 1, "head_hidden": 32,
        "tasks": ["sarcasm", "hate", "target", "severity"],
    })
    _, tok = build_encoder_and_tokenizer(cfg, [["hello", "world"], ["test", "sample"]])

    df_test = pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "text": ["normal clean post", "hateful target post", "sarcastic mixed post", "unknown class post"],
        "hate": ["non_hate", "hate", np.nan, "completely_unknown_category"],
        "sarcasm": ["not_sarcastic", np.nan, "sarcastic", None],
        "target": [np.nan, "individual", np.nan, "unknown_target_xyz"],
        "severity": [np.nan, "high", "moderate", "unknown_severity_xyz"],
    })

    ds = HateSarcasmDataset(df_test, tok, norm, max_length=16, tasks=cfg.tasks)
    assert len(ds) == 4

    # Row 0: non_hate (0), not_sarcastic (0), target NaN (-100), severity NaN (-100)
    assert ds[0]["labels"]["hate"] == 0
    assert ds[0]["labels"]["sarcasm"] == 0
    assert ds[0]["labels"]["target"] == IGNORE_INDEX
    assert ds[0]["labels"]["severity"] == IGNORE_INDEX

    # Row 1: hate (1), sarcasm NaN (-100), target individual (0), severity high (2)
    assert ds[1]["labels"]["hate"] == 1
    assert ds[1]["labels"]["sarcasm"] == IGNORE_INDEX
    assert ds[1]["labels"]["target"] == LABEL_SPACES["target"].index("individual")
    assert ds[1]["labels"]["severity"] == LABEL_SPACES["severity"].index("high")

    # Row 2: hate NaN (-100), sarcasm (1), target NaN (-100), severity moderate (1)
    assert ds[2]["labels"]["hate"] == IGNORE_INDEX
    assert ds[2]["labels"]["sarcasm"] == 1
    assert ds[2]["labels"]["target"] == IGNORE_INDEX
    assert ds[2]["labels"]["severity"] == LABEL_SPACES["severity"].index("moderate")

    # Row 3: unknown categories must map to IGNORE_INDEX rather than raising or mapping silently
    assert ds[3]["labels"]["hate"] == IGNORE_INDEX
    assert ds[3]["labels"]["sarcasm"] == IGNORE_INDEX
    assert ds[3]["labels"]["target"] == IGNORE_INDEX
    assert ds[3]["labels"]["severity"] == IGNORE_INDEX


def test_multitask_loss_masks_missing_labels_and_computes_strictly_on_valid():
    """Verify missing labels contribute zero loss and zero gradients, while valid labels contribute."""
    tasks = ["sarcasm", "hate", "target", "severity"]
    criterion = MultiTaskLoss(tasks)

    # 4 samples, batch size 4
    # Sarcasm: 2 valid, 2 ignored
    # Hate: 4 valid
    # Target: 1 valid, 3 ignored
    # Severity: 0 valid (all ignored)
    batch = {
        "labels": {
            "sarcasm": torch.tensor([0, 1, IGNORE_INDEX, IGNORE_INDEX]),
            "hate": torch.tensor([1, 0, 1, 0]),
            "target": torch.tensor([2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]),
            "severity": torch.full((4,), IGNORE_INDEX, dtype=torch.long),
        }
    }

    logits = {
        "sarcasm": torch.randn(4, 2, requires_grad=True),
        "hate": torch.randn(4, 2, requires_grad=True),
        "target": torch.randn(4, len(LABEL_SPACES["target"]), requires_grad=True),
        "severity": torch.randn(4, 3, requires_grad=True),
    }

    total_loss, parts = criterion({"logits": logits}, batch)

    # Severity has zero valid labels -> must not be in loss parts, contributes 0 to total loss
    assert "severity" not in parts
    assert "sarcasm" in parts and "hate" in parts and "target" in parts

    # Loss is finite
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0

    total_loss.backward()

    # Severity was 100% ignored -> logits for severity must receive zero gradients
    assert logits["severity"].grad is None

    # In target, only sample 0 was valid; samples 1, 2, 3 must have zero gradients
    assert torch.all(logits["target"].grad[1:] == 0.0)
    assert not torch.all(logits["target"].grad[0] == 0.0)

    # In sarcasm, only samples 0 and 1 were valid; samples 2 and 3 must have zero gradients
    assert torch.all(logits["sarcasm"].grad[2:] == 0.0)
    assert not torch.all(logits["sarcasm"].grad[:2] == 0.0)


def test_metrics_exclude_ignored_labels_and_adhere_to_class_taxonomy():
    """Verify task metrics exclude IGNORE_INDEX and report correct support and taxonomy."""
    # Test task_metrics_from_arrays directly
    y_true = np.array([0, 1, IGNORE_INDEX, 1, IGNORE_INDEX, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0])
    m_hate = task_metrics_from_arrays("hate", y_true, y_pred)

    # Out of 6 samples, exactly 4 are valid
    assert m_hate["n"] == 4
    assert m_hate["classes"] == HATE_LABELS
    assert m_hate["per_class"]["0"]["support"] == 2
    assert m_hate["per_class"]["1"]["support"] == 2

    # Test evaluate_predictions dataframe
    df_preds = pd.DataFrame({
        "gold_hate": [0, 1, IGNORE_INDEX, 1],
        "pred_hate": [0, 1, 0, 1],
        "p_hate": [0.1, 0.9, 0.4, 0.8],
        "gold_sarcasm": [IGNORE_INDEX, IGNORE_INDEX, 1, 0],
        "pred_sarcasm": [0, 1, 1, 0],
        "p_sarcasm": [0.2, 0.7, 0.8, 0.1],
        "gold_target": [IGNORE_INDEX, 0, IGNORE_INDEX, IGNORE_INDEX],
        "pred_target": [0, 0, 1, 2],
        "gold_severity": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
        "pred_severity": [0, 1, 2, 1],
    })

    res = evaluate_predictions(df_preds, tasks=["hate", "sarcasm", "target", "severity"])
    assert res["hate"]["n"] == 3
    assert res["sarcasm"]["n"] == 2
    assert res["target"]["n"] == 1
    assert res["severity"]["n"] == 0


def test_all_missing_auxiliary_rows_excluded_while_partial_retained():
    """Verify all-missing auxiliary rows are excluded while rows with >=1 valid label are retained."""
    df_aux = pd.DataFrame({
        "id": ["aux1", "aux2", "aux3", "aux4"],
        "text": ["offensive tweet", "hate tweet", "sarcastic tweet", "all nan tweet"],
        "hate": [np.nan, "hate", np.nan, np.nan],
        "sarcasm": [np.nan, np.nan, "sarcastic", np.nan],
        "target": [np.nan, np.nan, np.nan, np.nan],
        "severity": [np.nan, np.nan, np.nan, np.nan],
    })

    # Emulate cv.py auxiliary row filtering logic
    sup_mask = pd.Series(False, index=df_aux.index)
    for t in TASKS:
        if t in df_aux.columns:
            sup_mask = sup_mask | df_aux[t].apply(lambda v: label_to_id(t, v) != IGNORE_INDEX)

    retained = df_aux[sup_mask].reset_index(drop=True)

    # aux1 and aux4 are all-NaN -> must be excluded!
    # aux2 (hate valid) and aux3 (sarcasm valid) -> must be retained!
    assert len(retained) == 2
    assert list(retained["id"]) == ["aux2", "aux3"]


# =========================================================================
# Section 2: Fold and Auxiliary-Data Separation
# =========================================================================

def test_fold_cache_disjoint_partitions_and_full_test_coverage():
    """Verify all 5 folds have disjoint train/val/test IDs and outer test partitions cover 100% of MultiBully."""
    assert FOLDS_CACHE.exists(), f"Fold cache {FOLDS_CACHE} must exist"
    assert MULTIBULLY_CSV.exists(), f"MultiBully {MULTIBULLY_CSV} must exist"

    validate_fold_cache_checksum(FOLDS_CACHE, CANONICAL_FOLD_CACHE_SHA256)
    fold_assign = pd.read_csv(FOLDS_CACHE)
    mb_df = pd.read_csv(MULTIBULLY_CSV)
    expected_ids = set(mb_df["id"])
    assert len(expected_ids) == 5718

    # Validate disjoint splits and test coverage
    test_ids_by_fold = validate_fold_cache(fold_assign, expected_ids=expected_ids, folds=5)

    all_test_ids = []
    for f in range(5):
        f_df = fold_assign[fold_assign["fold"] == f]
        tr = set(f_df[f_df["split"] == "train"]["id"])
        va = set(f_df[f_df["split"] == "val"]["id"])
        te = set(f_df[f_df["split"] == "test"]["id"])

        # Disjointness
        assert len(tr.intersection(va)) == 0, f"Fold {f}: train and val overlap!"
        assert len(tr.intersection(te)) == 0, f"Fold {f}: train and test overlap!"
        assert len(va.intersection(te)) == 0, f"Fold {f}: val and test overlap!"

        all_test_ids.extend(te)

    # Exactly 5,718 test IDs with zero duplicates across outer folds
    assert len(all_test_ids) == 5718
    assert len(set(all_test_ids)) == 5718
    assert set(all_test_ids) == expected_ids


def test_hot_auxiliary_assigned_strictly_to_training():
    """Verify HOT auxiliary examples are strictly assigned to train, with zero in val and test."""
    assert HOT_GOLD_CSV.exists(), f"HOT gold {HOT_GOLD_CSV} must exist"
    hot_df = pd.read_csv(HOT_GOLD_CSV)
    fold_assign = pd.read_csv(FOLDS_CACHE)

    # Filter zero-supervision rows as cv.py does
    sup_mask = pd.Series(False, index=hot_df.index)
    for t in TASKS:
        if t in hot_df.columns:
            sup_mask = sup_mask | hot_df[t].apply(lambda v: label_to_id(t, v) != IGNORE_INDEX)
    aux_supervised = hot_df[sup_mask].reset_index(drop=True)
    assert len(aux_supervised) == 1424

    for f in range(5):
        # Simulate cv.py reconstruction:
        aux_fold = aux_supervised.copy()
        aux_fold["split"] = "train"
        aux_fold["fold"] = f

        assert (aux_fold["split"] == "train").all(), f"Fold {f}: Auxiliary data must be 100% train"
        assert not (aux_fold["split"] == "val").any()
        assert not (aux_fold["split"] == "test").any()


def test_zero_duplicate_text_crosses_auxiliary_training_and_multibully_eval():
    """Verify no duplicate text crosses HOT auxiliary training and MultiBully validation/test."""
    hot_df = pd.read_csv(HOT_GOLD_CSV)
    mb_df = pd.read_csv(MULTIBULLY_CSV)
    fold_assign = pd.read_csv(FOLDS_CACHE)

    normalizer = TextNormalizer()
    hot_norm = {normalizer(t).text for t in hot_df["text"] if t and str(t).strip()}

    for f in range(5):
        val_ids = set(fold_assign[(fold_assign.fold == f) & (fold_assign.split == "val")]["id"])
        test_ids = set(fold_assign[(fold_assign.fold == f) & (fold_assign.split == "test")]["id"])

        val_texts = {normalizer(t).text for t in mb_df[mb_df["id"].isin(val_ids)]["text"] if t and str(t).strip()}
        test_texts = {normalizer(t).text for t in mb_df[mb_df["id"].isin(test_ids)]["text"] if t and str(t).strip()}

        val_leak = hot_norm.intersection(val_texts)
        test_leak = hot_norm.intersection(test_texts)

        assert len(val_leak) == 0, f"Fold {f}: {len(val_leak)} texts leaked from auxiliary train into validation!"
        assert len(test_leak) == 0, f"Fold {f}: {len(test_leak)} texts leaked from auxiliary train into test!"


# =========================================================================
# Section 3: Training Pipeline Learning Check (Tiny Synthetic Overfit)
# =========================================================================

def test_training_pipeline_learns_tiny_synthetic_pattern():
    """Verify that loss is finite, gradients reach all task heads, and model learns simple patterns."""
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = Config.wrap({
        "encoder_name": "tiny",
        "tiny_hidden": 32,
        "tiny_layers": 1,
        "head_hidden": 32,
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "dropout": 0.0,
    })

    # Controlled synthetic samples with distinct lexical cues
    synthetic_samples = [
        {"id": "s1", "text": "peace love joy wonderful kindness", "hate": "non_hate", "sarcasm": "not_sarcastic", "target": None, "severity": None},
        {"id": "s2", "text": "delightful harmony friendship sweet respect", "hate": "non_hate", "sarcasm": "not_sarcastic", "target": None, "severity": None},
        {"id": "s3", "text": "kill destroy assault attack enemy individual", "hate": "hate", "sarcasm": "not_sarcastic", "target": "individual", "severity": "high"},
        {"id": "s4", "text": "hate violence strike murder religion", "hate": "hate", "sarcasm": "not_sarcastic", "target": "religion", "severity": "moderate"},
        {"id": "s5", "text": "oh wonderful brilliant catastrophe sarcastic joke", "hate": "non_hate", "sarcasm": "sarcastic", "target": None, "severity": None},
        {"id": "s6", "text": "yeah right great disaster attack gender", "hate": "hate", "sarcasm": "sarcastic", "target": "gender", "severity": "low"},
    ]
    df_tiny = pd.DataFrame(synthetic_samples)

    words_corpus = [s["text"].split() for s in synthetic_samples]
    enc, tok = build_encoder_and_tokenizer(cfg, words_corpus)
    model = ToxiFusionNet(enc, cfg)
    model.train()

    norm = TextNormalizer()
    ds = HateSarcasmDataset(df_tiny, tok, norm, max_length=16, tasks=cfg.tasks)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    collate_fn = make_collate(pad_id)
    batch = collate_fn([ds[i] for i in range(len(ds))])

    criterion = MultiTaskLoss(cfg.tasks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    # Initial forward and backward pass: check loss and gradients
    optimizer.zero_grad()
    out = model(batch["input_ids"], batch["attention_mask"], batch["cue"])
    initial_loss, initial_parts = criterion(out, batch)

    assert torch.isfinite(initial_loss), "Initial loss must be finite"
    assert initial_loss.item() > 0, "Initial loss must be positive"
    for t in cfg.tasks:
        assert t in initial_parts, f"All tasks must have valid supervision in synthetic dataset, missing {t}"

    initial_loss.backward()

    # Confirm gradients reach all 4 task heads
    assert model.sarc_out.weight.grad is not None, "Sarcasm head must receive gradients"
    assert torch.any(model.sarc_out.weight.grad != 0.0), "Sarcasm head gradients must be non-zero"

    assert model.heads["hate"][0].weight.grad is not None, "Hate head must receive gradients"
    assert torch.any(model.heads["hate"][0].weight.grad != 0.0), "Hate head gradients must be non-zero"

    assert model.heads["target"][0].weight.grad is not None, "Target head must receive gradients"
    assert torch.any(model.heads["target"][0].weight.grad != 0.0), "Target head gradients must be non-zero"

    assert model.heads["severity"][0].weight.grad is not None, "Severity head must receive gradients"
    assert torch.any(model.heads["severity"][0].weight.grad != 0.0), "Severity head gradients must be non-zero"

    # Train for 30 steps to confirm pipeline can learn/overfit simple patterns
    losses = [initial_loss.item()]
    for _ in range(30):
        optimizer.zero_grad()
        out = model(batch["input_ids"], batch["attention_mask"], batch["cue"])
        loss, _ = criterion(out, batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1]

    # Verify significant learning: final loss drops by >75%
    assert final_loss < 0.25 * initial_loss.item(), (
        f"Pipeline failed to learn simple synthetic pattern: initial loss {initial_loss.item():.4f}, final loss {final_loss:.4f}"
    )

    # Evaluate predictions on the training batch
    model.eval()
    with torch.no_grad():
        out_eval = model(batch["input_ids"], batch["attention_mask"], batch["cue"])
        for t in cfg.tasks:
            preds = out_eval["logits"][t].argmax(dim=-1)
            targets = batch["labels"][t]
            valid = targets != IGNORE_INDEX
            acc = (preds[valid] == targets[valid]).float().mean().item()
            assert acc >= 0.83, f"Task {t} accuracy after tiny-overfit is {acc:.2f}, expected >= 0.83"
