"""Test exact numerical parity between uncached live CLIP and cached frozen CLIP embeddings."""
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from toxifusionnet.config import Config
from toxifusionnet.models.toxifusionnet import ToxiFusionNet
from toxifusionnet.models.clip import build_clip_vision
from toxifusionnet.models.clip_cache import (
    CachedClipFeatureStore,
    build_or_load_clip_cache,
    get_cache_signature,
    get_multibully_mask,
)


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


def test_clip_cache_signature():
    """Verify that get_cache_signature produces deterministic signatures containing model weights and preprocessing info."""
    from toxifusionnet.models.clip_cache import get_cache_signature

    # Signature with model name only
    sig1 = get_cache_signature(model_name="openai/clip-vit-base-patch32", image_size=224)
    assert "openai_clip_vit_base_patch32" in sig1
    assert "sz224" in sig1

    # Signature with mock vision model and processor
    m = torch.nn.Linear(10, 10)
    class MockProc:
        size = 224
        crop_size = 224
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        resample = 3

    sig2 = get_cache_signature("openai/clip-vit-base-patch32", vision_model=m, image_processor=MockProc())
    assert "proc_" in sig2
    assert "w_" in sig2


def test_real_pretrained_clip_cached_vs_uncached_and_benchmark():
    """Test real pretrained CLIP ViT-B/32 numerical parity with valid and missing images, and benchmark runtime."""
    from transformers import CLIPVisionModel, CLIPImageProcessor
    from toxifusionnet.models.clip_cache import benchmark_clip_caching

    try:
        vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        pytest.skip(f"Pretrained CLIP model not available locally: {e}")

    device = torch.device("cpu")
    clip_cfg = {
        "clip_offline": False,
        "clip_model_name": "openai/clip-vit-base-patch32"
    }
    model_cfg = Config({
        "tasks": ["sarcasm", "hate", "target", "severity"],
        "dropout": 0.0,
        "head_hidden": 64,
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
    # Replace internal vision model with preloaded model
    model.clip_vision = vision_model.to(device)
    model.clip_processor = processor
    model.eval()

    # Batch of 4 samples: indices 0, 1 valid images; indices 2, 3 missing images
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

    # 2. Extract vis_feat using frozen vision backbone
    with torch.no_grad():
        precomputed_vis = model.clip_vision(pixel_values=pixel_values).pooler_output
        precomputed_vis[2:] = 0.0  # Missing images receive zero tensor

    # 3. Cached forward pass
    with torch.no_grad():
        out_cached = model(
            input_ids, attention_mask, cue=cue,
            vis_feat=precomputed_vis, image_mask=image_mask
        )

    # 4. Parity check within numerical tolerance (1e-4)
    for t in model.tasks:
        diff = (out_cached["logits"][t] - out_uncached["logits"][t]).abs().max().item()
        assert diff < 1e-4, f"Real CLIP task {t}: maximum difference {diff} exceeds 1e-4 tolerance"

    # 5. Measure and benchmark runtime
    bench = benchmark_clip_caching(
        model, input_ids, attention_mask, cue, pixel_values, precomputed_vis, image_mask, n_iters=5
    )
    assert bench["speedup_factor"] > 1.0, f"Expected speedup > 1.0, got {bench['speedup_factor']:.2f}x"
    assert bench["latency_reduction_pct"] > 0.0
    print(f"\nEmpirically Measured Benchmark on Real Pretrained CLIP: {bench['speedup_factor']:.2f}x speedup ({bench['latency_reduction_pct']:.1f}% latency reduction)")


class MockVisionOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class MockVisionModel(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.linear = torch.nn.Linear(10, hidden_size)
        class Config:
            pass
        self.config = Config()
        self.config.hidden_size = hidden_size

    def forward(self, pixel_values):
        bs = pixel_values.shape[0]
        return MockVisionOutput(torch.ones(bs, self.config.hidden_size) * 0.42)


class MockProcessor:
    def __init__(self, size=224):
        self.size = size
        self.crop_size = size
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.resample = 3

    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(len(images), 3, self.size, self.size)}


def test_get_multibully_mask_without_source_dataset():
    """Verify mask handles absent source_dataset by treating all rows as MultiBully."""
    df = pd.DataFrame({"id": ["1.jpg", "2.jpg"], "text": ["a", "b"]}, index=[10, 20])
    mask = get_multibully_mask(df)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
    assert (mask.index == df.index).all()
    assert mask.all()

    empty_df = pd.DataFrame({"id": [], "text": []})
    empty_mask = get_multibully_mask(empty_df)
    assert len(empty_mask) == 0
    assert empty_mask.dtype == bool


def test_get_multibully_mask_with_combined_and_malformed_sources():
    """Verify mask selects only explicit multibully records and rejects HOT, NaN, or malformed sources."""
    df = pd.DataFrame({
        "id": ["mb1.jpg", "mb2.jpg", "mb3.jpg", "hot1.txt", "hot2.txt", "corrupt.jpg", "nan.jpg"],
        "source_dataset": ["multibully", " MultiBully ", "MULTIBULLY", "hot", "HOT", "invalid_source", None],
        "text": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
    }, index=[100, 101, 102, 103, 104, 105, 106])

    mask = get_multibully_mask(df)
    assert (mask.index == df.index).all()
    assert mask.tolist() == [True, True, True, False, False, False, False]
    selected_ids = df[mask]["id"].tolist()
    assert selected_ids == ["mb1.jpg", "mb2.jpg", "mb3.jpg"]


def test_build_and_validate_clip_cache_without_source_dataset(tmp_path):
    """Verify construction and validation when source_dataset is absent (fixes KeyError: True)."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cache_dir = tmp_path / "cache"

    img1 = Image.new("RGB", (32, 32), color="red")
    img1.save(img_dir / "101.jpg")
    img2 = Image.new("RGB", (32, 32), color="green")
    img2.save(img_dir / "102.jpg")

    df = pd.DataFrame({"id": ["101.jpg", "102.jpg"], "text": ["text1", "text2"]})

    vision_model = MockVisionModel(hidden_size=32)
    processor = MockProcessor(size=32)

    # 1. Construction without source_dataset
    store = build_or_load_clip_cache(
        df, image_dir=img_dir, vision_model=vision_model, image_processor=processor,
        cache_dir=cache_dir, model_name="dummy_mock_model", batch_size=2
    )
    assert len(store) == 2
    feat1, mask1 = store.get("101.jpg", hidden_size=32)
    assert mask1 == 1.0
    assert feat1.shape == (32,)
    assert torch.allclose(feat1, torch.ones(32) * 0.42)

    # Verify metadata
    cache_files = list(cache_dir.glob("*.pt"))
    assert len(cache_files) == 1
    meta_json = cache_files[0].with_suffix(".json")
    assert meta_json.exists()
    meta = json.loads(meta_json.read_text())
    assert meta["total_items"] == 2
    assert meta["valid_images"] == 2
    assert meta["missing_images"] == 0

    # 2. Validation and reuse of existing cache without source_dataset
    store_loaded = build_or_load_clip_cache(
        df, image_dir=img_dir, vision_model=vision_model, image_processor=processor,
        cache_dir=cache_dir, model_name="dummy_mock_model", batch_size=2
    )
    assert len(store_loaded) == 2
    assert "101.jpg" in store_loaded
    assert "102.jpg" in store_loaded


def test_clip_cache_missing_and_corrupt_images_accounting(tmp_path):
    """Verify correct accounting for valid, missing, and corrupted/unreadable image files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cache_dir = tmp_path / "cache"

    Image.new("RGB", (32, 32), color="white").save(img_dir / "valid.jpg")
    (img_dir / "corrupt.jpg").write_text("this is definitely not a valid image format")
    # missing.jpg is omitted

    df = pd.DataFrame({"id": ["valid.jpg", "corrupt.jpg", "missing.jpg"], "text": ["v", "c", "m"]})

    vision_model = MockVisionModel(hidden_size=32)
    processor = MockProcessor(size=32)

    store = build_or_load_clip_cache(
        df, image_dir=img_dir, vision_model=vision_model, image_processor=processor,
        cache_dir=cache_dir, model_name="mock_accounting_test", batch_size=2
    )

    assert len(store) == 3
    # Valid image
    fv, mv = store.get("valid.jpg", hidden_size=32)
    assert mv == 1.0
    assert torch.allclose(fv, torch.ones(32) * 0.42)

    # Corrupt image
    fc, mc = store.get("corrupt.jpg", hidden_size=32)
    assert mc == 0.0
    assert torch.equal(fc, torch.zeros(32))

    # Missing image
    fm, mm = store.get("missing.jpg", hidden_size=32)
    assert mm == 0.0
    assert torch.equal(fm, torch.zeros(32))

    # Verify accounting metadata
    cache_files = list(cache_dir.glob("*.pt"))
    meta = json.loads(cache_files[0].with_suffix(".json").read_text())
    assert meta["total_items"] == 3
    assert meta["valid_images"] == 1
    assert meta["missing_images"] == 2


def test_clip_cache_combined_dataset_filtering_and_incomplete_update(tmp_path):
    """Verify combined datasets exclude HOT records and incomplete caches are updated."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cache_dir = tmp_path / "cache"

    Image.new("RGB", (32, 32), color="yellow").save(img_dir / "mb_1.jpg")
    Image.new("RGB", (32, 32), color="purple").save(img_dir / "mb_2.jpg")
    Image.new("RGB", (32, 32), color="orange").save(img_dir / "mb_3.jpg")
    Image.new("RGB", (32, 32), color="blue").save(img_dir / "hot_1.jpg")

    df_combined = pd.DataFrame({
        "id": ["mb_1.jpg", "mb_2.jpg", "hot_1.jpg", "hot_2.txt", "bad_source.jpg"],
        "source_dataset": ["multibully", "multibully", "hot", "hot", "unknown"],
        "text": ["m1", "m2", "h1", "h2", "b1"]
    })

    vision_model = MockVisionModel(hidden_size=32)
    processor = MockProcessor(size=32)

    # 1. Build cache with combined dataset: only MultiBully rows cached
    store = build_or_load_clip_cache(
        df_combined, image_dir=img_dir, vision_model=vision_model, image_processor=processor,
        cache_dir=cache_dir, model_name="combined_clip_test", batch_size=2
    )
    assert len(store) == 2
    assert "mb_1.jpg" in store
    assert "mb_2.jpg" in store
    assert "hot_1.jpg" not in store
    assert "hot_2.txt" not in store
    assert "bad_source.jpg" not in store

    meta = json.loads(list(cache_dir.glob("*.pt"))[0].with_suffix(".json").read_text())
    assert meta["total_items"] == 2
    assert meta["valid_images"] == 2
    assert meta["missing_images"] == 0

    # 2. Add a new MultiBully item (mb_3.jpg) -> Incomplete cache detection and update
    df_updated = pd.DataFrame({
        "id": ["mb_1.jpg", "mb_2.jpg", "mb_3.jpg", "hot_1.jpg"],
        "source_dataset": ["multibully", "multibully", "multibully", "hot"],
        "text": ["m1", "m2", "m3", "h1"]
    })
    store_updated = build_or_load_clip_cache(
        df_updated, image_dir=img_dir, vision_model=vision_model, image_processor=processor,
        cache_dir=cache_dir, model_name="combined_clip_test", batch_size=2
    )
    assert len(store_updated) == 3
    assert "mb_1.jpg" in store_updated
    assert "mb_2.jpg" in store_updated
    assert "mb_3.jpg" in store_updated
    assert "hot_1.jpg" not in store_updated

    meta_updated = json.loads(list(cache_dir.glob("*.pt"))[0].with_suffix(".json").read_text())
    assert meta_updated["total_items"] == 3
    assert meta_updated["valid_images"] == 3
    assert meta_updated["missing_images"] == 0


def test_clip_cache_corrupted_cache_file_recovery(tmp_path):
    """Verify corrupted cache files on disk trigger safe rebuild rather than crashing."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    img1 = Image.new("RGB", (32, 32), color="red")
    img1.save(img_dir / "recovery_01.jpg")

    df = pd.DataFrame({"id": ["recovery_01.jpg"], "text": ["rec"]})
    vision_model = MockVisionModel(hidden_size=32)
    processor = MockProcessor(size=32)

    # Write corrupt binary bytes into the expected cache path
    sig = get_cache_signature("mock_recovery_model", vision_model, processor)
    corrupt_cache = cache_dir / f"{sig}.pt"
    corrupt_cache.write_bytes(b"corrupted binary data not a valid torch file")

    store = build_or_load_clip_cache(
        df, image_dir=img_dir, vision_model=vision_model, image_processor=processor,
        cache_dir=cache_dir, model_name="mock_recovery_model", batch_size=2
    )
    assert len(store) == 1
    feat, mask = store.get("recovery_01.jpg", hidden_size=32)
    assert mask == 1.0
    assert torch.allclose(feat, torch.ones(32) * 0.42)


def test_clip_cache_prunes_extraneous_hot_ids_from_existing_cache(tmp_path):
    """Verify pre-populated existing cache with HOT/extraneous IDs is pruned and metadata updated."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create dummy MultiBully images on disk
    Image.new("RGB", (32, 32), color="red").save(img_dir / "mb_1.jpg")
    Image.new("RGB", (32, 32), color="green").save(img_dir / "mb_2.jpg")

    vision_model = MockVisionModel(hidden_size=32)
    processor = MockProcessor(size=32)
    sig = get_cache_signature("mock_pruning_model", vision_model, processor)
    cache_path = cache_dir / f"{sig}.pt"

    # Pre-populate existing cache on disk with MultiBully features + HOT contaminant feature
    pre_store = CachedClipFeatureStore()
    pre_store.embeddings["mb_1.jpg"] = torch.ones(32) * 0.42
    pre_store.masks["mb_1.jpg"] = 1.0
    pre_store.embeddings["mb_2.jpg"] = torch.ones(32) * 0.42
    pre_store.masks["mb_2.jpg"] = 1.0
    pre_store.embeddings["hot_contaminant.jpg"] = torch.ones(32) * 9.99
    pre_store.masks["hot_contaminant.jpg"] = 1.0

    meta_initial = {
        "signature": sig,
        "model_name": "mock_pruning_model",
        "total_items": 3,
        "valid_images": 3,
        "missing_images": 0,
        "created_at": "2026-10-15 00:00:00"
    }
    pre_store.save(cache_path, metadata=meta_initial)

    # Combined dataframe with MultiBully and HOT rows
    df_combined = pd.DataFrame({
        "id": ["mb_1.jpg", "mb_2.jpg", "hot_contaminant.jpg"],
        "source_dataset": ["multibully", "multibully", "hot"],
        "text": ["t1", "t2", "t3"]
    })

    # Call build_or_load_clip_cache
    store = build_or_load_clip_cache(
        df_combined, image_dir=img_dir, vision_model=vision_model, image_processor=processor,
        cache_dir=cache_dir, model_name="mock_pruning_model", batch_size=2
    )

    # Assert HOT contaminant is pruned and only MultiBully features remain
    assert "hot_contaminant.jpg" not in store
    assert "mb_1.jpg" in store
    assert "mb_2.jpg" in store
    assert len(store) == 2

    # Assert cached features for MultiBully were reused without alteration
    f1, m1 = store.get("mb_1.jpg", hidden_size=32)
    assert m1 == 1.0
    assert torch.allclose(f1, torch.ones(32) * 0.42)

    # Assert disk cache and metadata are updated and consistent
    meta_json = cache_path.with_suffix(".json")
    assert meta_json.exists()
    meta_updated = json.loads(meta_json.read_text())
    assert meta_updated["total_items"] == 2
    assert meta_updated["valid_images"] == 2
    assert meta_updated["missing_images"] == 0

    # Assert subsequent cache reload preserves pruned state
    reloaded_store = CachedClipFeatureStore(cache_path)
    assert "hot_contaminant.jpg" not in reloaded_store
    assert len(reloaded_store) == 2
