import pytest
import torch

from toxifusionnet.config import Config
from toxifusionnet.labels import IGNORE_INDEX
from toxifusionnet.models.encoder import build_encoder_and_tokenizer
from toxifusionnet.models.losses import MultiTaskLoss
from toxifusionnet.models.toxifusionnet import SarcasmAwareFusion, ToxiFusionNet


def make(**over):
    cfg = Config.wrap({"encoder_name": "tiny", "tiny_hidden": 32, "tiny_layers": 1, "head_hidden": 32,
                       "tasks": ["sarcasm", "hate", "target", "severity"], **over})
    enc, tok = build_encoder_and_tokenizer(cfg, [["wah", "kya", "baat", "hai"], ["tum", "log", "bekaar"]])
    return ToxiFusionNet(enc, cfg), tok


def batch(model, B=4, T=7):
    ids = torch.randint(5, 40, (B, T)); mask = torch.ones(B, T, dtype=torch.long); mask[0, 5:] = 0
    return ids, mask, torch.rand(B, 16)


def test_forward_shapes_and_attention_normalised():
    m, _ = make()
    ids, mask, cue = batch(m)
    out = m(ids, mask, cue)
    assert out["logits"]["sarcasm"].shape == (4, 2) and out["logits"]["hate"].shape == (4, 2)
    assert out["logits"]["target"].shape == (4, 5) and out["logits"]["severity"].shape == (4, 3)
    assert torch.allclose(out["attn"]["hate"].sum(-1), torch.ones(4), atol=1e-5)
    assert out["attn"]["hate"][0, 5:].sum() == 0  # padded positions get no attention
    assert out["gate"] is not None


@pytest.mark.parametrize("over,has_sarc,has_fusion", [
    ({}, True, True), ({"use_fusion": False}, True, False), ({"use_sarcasm_branch": False}, False, False),
    ({"tasks": ["hate"]}, False, False)])
def test_ablation_switches(over, has_sarc, has_fusion):
    m, _ = make(**over)
    ids, mask, cue = batch(m)
    out = m(ids, mask, cue)
    assert ("sarcasm" in out["logits"]) == has_sarc and (out["gate"] is not None) == has_fusion
    assert "hate" in out["logits"]


def test_fusion_gate_changes_output_with_sarcasm_probability():
    f = SarcasmAwareFusion(8)
    h, s = torch.randn(2, 8), torch.randn(2, 8)
    a, _ = f(h, s, torch.zeros(2, 1)); b, _ = f(h, s, torch.ones(2, 1))
    assert not torch.allclose(a, b)


def test_loss_masks_missing_labels_and_backprops():
    m, _ = make()
    ids, mask, cue = batch(m)
    out = m(ids, mask, cue)
    y = {"sarcasm": torch.tensor([0, 1, IGNORE_INDEX, 1]), "hate": torch.tensor([1, 0, 1, 1]),
         "target": torch.tensor([2, IGNORE_INDEX, IGNORE_INDEX, 0]), "severity": torch.full((4,), IGNORE_INDEX)}
    crit = MultiTaskLoss(m.tasks, weights={"target": .5})
    loss, parts = crit(out, {"labels": y})
    assert "severity" not in parts and {"sarcasm", "hate", "target"} <= set(parts)
    loss.backward()
    assert m.hate_pool.proj.weight.grad is not None


def test_all_missing_labels_gives_zero_loss():
    m, _ = make()
    ids, mask, cue = batch(m)
    out = m(ids, mask, cue)
    y = {t: torch.full((4,), IGNORE_INDEX) for t in m.tasks}
    loss, parts = MultiTaskLoss(m.tasks)(out, {"labels": y})
    assert float(loss) == 0.0 and parts == {}


def test_uncertainty_weighting_has_learnable_params_and_attention_loss():
    m, _ = make()
    crit = MultiTaskLoss(m.tasks, weighting="uncertainty", attention_lambda=0.5)
    assert len(list(crit.log_vars.parameters())) == 4
    ids, mask, cue = batch(m)
    out = m(ids, mask, cue)
    tgt = torch.softmax(torch.randn(4, 7), -1) * mask
    y = {"sarcasm": torch.tensor([0, 1, 0, 1]), "hate": torch.tensor([1, 0, 1, 1]),
         "target": torch.tensor([2, 1, 0, 0]), "severity": torch.tensor([0, 1, 2, 0])}
    b = {"labels": y, "hate_attn_target": tgt, "hate_attn_valid": torch.tensor([True, True, False, False]),
         "sarc_attn_target": tgt, "sarc_attn_valid": torch.tensor([False] * 4)}
    loss, parts = crit(out, b)
    assert "attention" in parts and parts["attention"] > 0
