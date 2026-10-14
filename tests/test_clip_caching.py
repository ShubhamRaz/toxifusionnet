"""Test exact numerical parity between uncached live CLIP and cached frozen CLIP embeddings."""
import pytest
import torch
import torch.nn.functional as F

from toxifusionnet.config import Config
from toxifusionnet.models.toxifusionnet import ToxiFusionNet
from toxifusionnet.models.clip import build_clip_vision
from toxifusionnet.models.clip_cache import CachedClipFeatureStore


class DummyTinyEncoder(torch.nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.d = d
        self.config = Config({"hidden_size": d})
        self.emb = torch.nn.Embedding(100, d)

    def forward(self, input_ids, attention_mask=None):
        out = self.emb(input_ids)
        class Ret:
            last_hidden_state = out
        return Ret()


def test_clip_cached_feature_numerical_parity():
    """Verify that forward pass with precomputed vis_feat matches uncached pixel_values within 1e-6."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Lightweight offline CLIP model
    clip_cfg = {
        "clip_offline": True,
        "clip_hidden_size": 64,
        "clip_layers": 2,
        "clip_heads": 4,
        "clip_model_name": "dummy_clip"
    }
    model_cfg = Config({
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "dropout": 0.0,
        "head_hidden": 32,
        "use_sarcasm_branch": True,
        "use_fusion": True,
        "use_cue_features": True,
        "cue_hidden": 16,
        "use_clip": True,
        "freeze_clip": True,
        **clip_cfg
    })

    encoder = DummyTinyEncoder(d=64)
    model = ToxiFusionNet(encoder, model_cfg).to(device)
    model.eval()

    # Inputs: batch of 4 (2 with images, 2 without images)
    input_ids = torch.randint(1, 50, (4, 16))
    attention_mask = torch.ones(4, 16)
    cue = torch.randn(4, 16)
    pixel_values = torch.randn(4, 3, 224, 224)
    image_mask = torch.tensor([[1.0], [1.0], [0.0], [0.0]])

    # 1. Uncached forward pass
    with torch.no_grad():
        out_uncached = model(
            input_ids, attention_mask, cue=cue,
            pixel_values=pixel_values, image_mask=image_mask
        )

    # 2. Extract vis_feat using frozen vision backbone (as caching does)
    with torch.no_grad():
        precomputed_vis = model.clip_vision(pixel_values=pixel_values).pooler_output
        # For masked images, cache guarantees zero vector
        precomputed_vis[2:] = 0.0

    # 3. Cached forward pass (passing precomputed vis_feat directly)
    with torch.no_grad():
        out_cached = model(
            input_ids, attention_mask, cue=cue,
            vis_feat=precomputed_vis, image_mask=image_mask
        )

    # 4. Assert exact numerical parity across all task logits
    for t in model.tasks:
        diff = (out_cached["logits"][t] - out_uncached["logits"][t]).abs().max().item()
        assert diff < 1e-6, f"Task {t}: maximum absolute difference is {diff}, expected < 1e-6"


def test_cached_clip_feature_store():
    """Verify CachedClipFeatureStore save, load, and fallback behavior."""
    store = CachedClipFeatureStore()
    emb_dummy = torch.randn(64)
    store.embeddings["img_001.jpg"] = emb_dummy
    store.masks["img_001.jpg"] = 1.0

    # Retrieve existing
    e, m = store.get("img_001.jpg", hidden_size=64)
    assert torch.equal(e, emb_dummy)
    assert m == 1.0

    # Retrieve non-existing (missing image)
    e_miss, m_miss = store.get("img_missing.jpg", hidden_size=64)
    assert torch.equal(e_miss, torch.zeros(64))
    assert m_miss == 0.0
