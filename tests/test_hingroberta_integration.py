"""Tests for L3Cube HingRoBERTa integration into ToxiFusionNet.

Covers:
1. Encoder selection and configuration parsing.
2. Hugging Face model and tokenizer initialization.
3. Dynamic hidden-dimension compatibility with all 4 task heads.
4. Multi-task forward passes for sarcasm, hate, target, and severity.
5. Checkpoint save/reload and prediction consistency (parity).
6. Error handling on invalid/missing model configurations.
7. Preservation of canonical fold cache verification and leak-free split guarantees.
8. Output directory and experiment isolation.
9. End-to-end real-model smoke test (forward + backward + optimizer step).
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
import torch
import yaml
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel

from toxifusionnet.config import Config, load_config
from toxifusionnet.experiments.cv import (
    ALLOWED_COMBINED_FOLD_CACHE_SHAS,
    ALLOWED_FOLD_CACHE_SHAS,
    compute_normalized_fold_cache_sha256,
    validate_fold_cache_checksum,
)
from toxifusionnet.models.encoder import build_encoder_and_tokenizer, rebuild_encoder_from_dir
from toxifusionnet.models.losses import MultiTaskLoss
from toxifusionnet.models.toxifusionnet import ToxiFusionNet


HING_ROBERTA_NAME = "l3cube-pune/hing-roberta"


def test_hingroberta_config_parsing():
    """Verify configs/exp1d_hingroberta_cleaned.yaml parses correctly and preserves all baseline hyperparameters."""
    cfg_path = Path("configs/exp1d_hingroberta_cleaned.yaml")
    assert cfg_path.exists(), f"Configuration file missing: {cfg_path}"

    cfg = load_config(cfg_path)
    assert cfg.experiment_name == "exp1d_hingroberta_cleaned"
    assert cfg.model.encoder_name == HING_ROBERTA_NAME
    assert cfg.data.processed_path == "data/cleaned/multibully_cleaned.csv"
    assert cfg.data.folds_cache == "data/cleaned/multibully_v2_cv_folds.csv"

    # Tasks and Heads
    assert set(cfg.model.tasks) == {"sarcasm", "hate", "target", "severity"}
    assert cfg.model.use_sarcasm_branch is True
    assert cfg.model.use_fusion is True
    assert cfg.model.use_cue_features is True
    assert cfg.model.use_clip is False  # Text-only baseline

    # Loss and class balancing parity with E1d
    assert cfg.loss.class_weighting == "sqrt_balanced"
    assert float(cfg.loss.max_class_weight) == 4.0
    assert cfg.loss.class_weight_source == "multibully"

    # Training schedule parity
    assert int(cfg.train.epochs) == 8
    assert int(cfg.train.batch_size) == 64
    assert int(cfg.train.max_steps) == 488
    assert float(cfg.train.encoder_lr) == 1.0e-5
    assert float(cfg.train.head_lr) == 5.0e-4


def test_dynamic_hidden_dimension_compatibility():
    """Verify ToxiFusionNet dynamically inspects encoder.config.hidden_size and adapts all 4 heads."""
    for dim in [64, 128, 768]:
        dummy_cfg = BertConfig(
            vocab_size=100,
            hidden_size=dim,
            num_hidden_layers=1,
            num_attention_heads=2 if dim % 2 == 0 else 1,
            intermediate_size=dim * 2,
        )
        dummy_enc = BertModel(dummy_cfg)

        model_cfg = {
            "encoder_name": f"synthetic-{dim}",
            "tasks": ["sarcasm", "hate", "target", "severity"],
            "dropout": 0.1,
            "head_hidden": 64,
            "use_sarcasm_branch": True,
            "use_fusion": True,
            "use_cue_features": True,
            "cue_hidden": 16,
            "use_clip": False,
        }
        model = ToxiFusionNet(dummy_enc, model_cfg)
        assert model.hate_pool.proj.in_features == dim
        assert model.fusion.gate.in_features == 2 * dim + 1
        assert model.heads["hate"][0].in_features == dim
        assert model.heads["target"][0].in_features == dim
        assert model.heads["severity"][0].in_features == dim

        # Verify forward pass with dynamic dim
        inp = torch.randint(0, 100, (2, 8))
        mask = torch.ones((2, 8), dtype=torch.long)
        cue = torch.zeros(2, 16)
        out = model(inp, mask, cue=cue)
        assert out["logits"]["sarcasm"].shape == (2, 2)
        assert out["logits"]["hate"].shape == (2, 2)
        assert out["logits"]["target"].shape == (2, 5)
        assert out["logits"]["severity"].shape == (2, 3)


def test_invalid_encoder_configuration_handling():
    """Verify that an invalid or non-existent encoder name fails explicitly without falling back silently."""
    invalid_cfg = Config({"encoder_name": "non_existent_hf_org/totally_fake_model_xyz_9999"})
    with pytest.raises(Exception):
        build_encoder_and_tokenizer(invalid_cfg)


def test_canonical_fold_cache_preservation_with_hingroberta():
    """Verify that fold cache verification accepts the curated cleaned folds and detects corruption."""
    mb_cache = Path("data/cleaned/multibully_v2_cv_folds.csv")
    comb_cache = Path("data/cleaned/combined_multibully_hot_cv_folds.csv")

    assert mb_cache.exists()
    assert comb_cache.exists()

    sha_mb = validate_fold_cache_checksum(mb_cache, ALLOWED_FOLD_CACHE_SHAS)
    sha_comb = validate_fold_cache_checksum(comb_cache, ALLOWED_COMBINED_FOLD_CACHE_SHAS)

    assert sha_mb in ALLOWED_FOLD_CACHE_SHAS
    assert sha_comb in ALLOWED_COMBINED_FOLD_CACHE_SHAS

    # Verify that a tampered file is rejected
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
        tmp.write("id,split,fold,group_id\n1,train,0,0\n")
        tmp_path = Path(tmp.name)
    try:
        with pytest.raises(ValueError, match="Canonical fold cache checksum verification failed"):
            validate_fold_cache_checksum(tmp_path, ALLOWED_FOLD_CACHE_SHAS)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_experiment_output_isolation():
    """Verify experiment output locations are strictly isolated in run_supercomputer.sh."""
    script_text = Path("run_supercomputer.sh").read_text(encoding="utf-8")
    assert "cv_e1d_hingroberta_cleaned)" in script_text
    assert "results/v3_journal_final/e1d_hingroberta_cleaned" in script_text
    assert "configs/exp1d_hingroberta_cleaned.yaml" in script_text
    # Ensure distinct directory from baseline
    assert "results/v3_journal_final/e1d_cleaned" in script_text


def test_hingroberta_checkpoint_save_reload_parity(tmp_path: Path):
    """Verify encoder saving and reloading via rebuild_encoder_from_dir reproduces identical outputs."""
    tok = AutoTokenizer.from_pretrained(HING_ROBERTA_NAME)
    enc = AutoModel.from_pretrained(HING_ROBERTA_NAME)

    cfg = {
        "encoder_name": HING_ROBERTA_NAME,
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "dropout": 0.0,
        "head_hidden": 64,
        "use_sarcasm_branch": True,
        "use_fusion": True,
        "use_cue_features": True,
        "cue_hidden": 16,
        "use_clip": False,
    }
    model = ToxiFusionNet(enc, cfg).eval()

    # Save components
    run_dir = tmp_path / "test_run"
    run_dir.mkdir(parents=True)
    tok.save_pretrained(run_dir / "tokenizer")
    enc.config.save_pretrained(run_dir / "encoder_config")
    torch.save({"state_dict": model.state_dict(), "cfg": {"model": cfg}}, run_dir / "model.pt")

    # Reload from saved directory
    rebuilt_enc, rebuilt_tok = rebuild_encoder_from_dir(run_dir)
    rebuilt_model = ToxiFusionNet(rebuilt_enc, cfg).eval()
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    rebuilt_model.load_state_dict(ckpt["state_dict"])

    # Forward pass parity
    text = "ye post bahut accha hai par kuch log hate faila rahe hain"
    inp = tok(text, return_tensors="pt")
    cue = torch.zeros(1, 16)

    with torch.no_grad():
        out1 = model(inp["input_ids"], inp["attention_mask"], cue=cue)
        out2 = rebuilt_model(inp["input_ids"], inp["attention_mask"], cue=cue)

    for task in ["sarcasm", "hate", "target", "severity"]:
        assert torch.allclose(out1["logits"][task], out2["logits"][task], atol=1e-5), f"Mismatch in {task} logits"


def test_hingroberta_smoke_optimizer_step():
    """Minimal real-model smoke test: loads HingRoBERTa, computes multi-task loss, backprops, and steps optimizer."""
    tok = AutoTokenizer.from_pretrained(HING_ROBERTA_NAME)
    enc = AutoModel.from_pretrained(HING_ROBERTA_NAME)

    cfg = {
        "encoder_name": HING_ROBERTA_NAME,
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "dropout": 0.1,
        "head_hidden": 64,
        "use_sarcasm_branch": True,
        "use_fusion": True,
        "use_cue_features": True,
        "cue_hidden": 16,
        "use_clip": False,
    }
    model = ToxiFusionNet(enc, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = MultiTaskLoss(model.tasks, weighting="fixed", weights={t: 1.0 for t in model.tasks})

    texts = [
        "bhai kya baat hai zabardast meme banaya hai",
        "tere jaise logon ko desh se nikal dena chahiye ghatiya insaan",
    ]
    inp = tok(texts, padding=True, truncation=True, return_tensors="pt")
    cue = torch.zeros(2, 16)

    out = model(inp["input_ids"], inp["attention_mask"], cue=cue)
    for task in ["sarcasm", "hate", "target", "severity"]:
        assert task in out["logits"]
        assert torch.isfinite(out["logits"][task]).all()

    batch = {
        "labels": {
            "sarcasm": torch.tensor([0, 1]),
            "hate": torch.tensor([0, 1]),
            "target": torch.tensor([-100, 2]),
            "severity": torch.tensor([-100, 1]),
        }
    }
    loss, parts = criterion(out, batch)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert any(g > 0.0 for g in grad_norms)

    opt.step()
    # Smoke test completed cleanly without error
