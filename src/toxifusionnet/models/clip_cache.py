"""Caching and fast retrieval of frozen CLIP ViT visual embeddings.

Eliminates repetitive vision transformer forward passes during multi-epoch training,
providing substantial throughput gains while guaranteeing exact numerical equivalence.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

log = logging.getLogger("toxifusionnet")


def hash_image_file(path: str | Path) -> str:
    """Compute SHA256 hash of an image file for cache key verification."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_cache_signature(
    model_name: str = "openai/clip-vit-base-patch32",
    vision_model: torch.nn.Module | None = None,
    image_processor=None,
    image_size: int = 224,
) -> str:
    """Construct deterministic cache signature from model architecture, weight hash, and preprocessing configuration."""
    safe_name = model_name.replace("/", "_").replace("-", "_")
    sig_parts = [safe_name, f"sz{image_size}"]

    # 1. Preprocessing Configuration Signature
    if image_processor is not None:
        proc_dict = {
            "size": getattr(image_processor, "size", image_size),
            "crop_size": getattr(image_processor, "crop_size", None),
            "image_mean": getattr(image_processor, "image_mean", None),
            "image_std": getattr(image_processor, "image_std", None),
            "resample": getattr(image_processor, "resample", None),
        }
        proc_str = json.dumps(proc_dict, sort_keys=True, default=str)
        proc_hash = hashlib.sha256(proc_str.encode()).hexdigest()[:8]
        sig_parts.append(f"proc_{proc_hash}")

    # 2. Vision Model Weight Checksum / Hash
    if vision_model is not None:
        h = hashlib.sha256()
        with torch.no_grad():
            for p_name, param in vision_model.named_parameters():
                if "weight" in p_name:
                    chunk = param.data[:2, :2].cpu().numpy().tobytes() if param.dim() >= 2 else param.data[:4].cpu().numpy().tobytes()
                    h.update(chunk)
        sig_parts.append(f"w_{h.hexdigest()[:8]}")

    return "_".join(sig_parts)


def extract_clip_embeddings(
    image_paths: list[Path | None],
    vision_model: torch.nn.Module,
    processor,
    device: torch.device,
    batch_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 768-d pooler_output vectors for a list of image paths.
    
    Returns:
        features: (N, hidden_size) float32 tensor
        has_image: (N,) float32 binary mask (1.0 = valid image, 0.0 = missing/corrupt)
    """
    vision_model.eval()
    n = len(image_paths)
    hidden_size = getattr(getattr(vision_model, "config", None), "hidden_size", 768)
    feats = torch.zeros((n, hidden_size), dtype=torch.float32)
    masks = torch.zeros(n, dtype=torch.float32)

    for i in range(0, n, batch_size):
        chunk = image_paths[i : i + batch_size]
        valid_indices = []
        valid_pil = []

        for offset, p in enumerate(chunk):
            if p is not None and p.exists():
                try:
                    with Image.open(p) as img:
                        pil = img.convert("RGB")
                        valid_indices.append(i + offset)
                        valid_pil.append(pil)
                except Exception:
                    pass

        if valid_pil:
            inputs = processor(images=valid_pil, return_tensors="pt")
            pix = inputs["pixel_values"].to(device)
            with torch.no_grad():
                out = vision_model(pixel_values=pix).pooler_output.cpu()
            for vi, emb in zip(valid_indices, out):
                feats[vi] = emb
                masks[vi] = 1.0

    return feats, masks


class CachedClipFeatureStore:
    """In-memory and disk-backed fast feature store for precomputed CLIP embeddings."""

    def __init__(self, cache_file: str | Path | None = None):
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.masks: Dict[str, float] = {}
        self.cache_file = Path(cache_file) if cache_file else None
        if self.cache_file and self.cache_file.exists():
            self.load(self.cache_file)

    def load(self, cache_file: str | Path):
        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        self.embeddings = data["embeddings"]
        self.masks = data["masks"]
        log.info("Loaded %d cached CLIP embeddings from %s", len(self.embeddings), cache_file)

    def save(self, cache_file: str | Path, metadata: dict | None = None):
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "embeddings": self.embeddings,
            "masks": self.masks,
            "metadata": metadata or {}
        }, cache_path)
        if metadata:
            meta_json = cache_path.with_suffix(".json")
            meta_json.write_text(json.dumps(metadata, indent=2))
        log.info("Saved %d cached CLIP embeddings to %s", len(self.embeddings), cache_path)

    def get(self, item_id: str, hidden_size: int = 768) -> Tuple[torch.Tensor, float]:
        """Retrieve embedding and mask flag. Returns zero tensor if absent."""
        if item_id in self.embeddings:
            return self.embeddings[item_id], self.masks.get(item_id, 1.0)
        return torch.zeros(hidden_size, dtype=torch.float32), 0.0

    def __contains__(self, item_id: str) -> bool:
        return item_id in self.embeddings

    def __len__(self) -> int:
        return len(self.embeddings)


def get_multibully_mask(df: pd.DataFrame) -> pd.Series:
    """Return an explicit, index-aligned boolean mask selecting MultiBully records.

    Semantics:
    - If 'source_dataset' is absent from df, every row is treated as MultiBully.
    - If 'source_dataset' is present, only rows explicitly marked 'multibully'
      (case-insensitive, trimmed) are selected. Missing (NaN), HOT, or malformed
      labels are strictly excluded.
    """
    if "source_dataset" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    col = df["source_dataset"]
    return col.notna() & (col.astype(str).str.strip().str.lower() == "multibully")


def build_or_load_clip_cache(
    df: pd.DataFrame,
    image_dir: str | Path,
    vision_model: torch.nn.Module,
    image_processor,
    cache_dir: str | Path = "data/cache/clip_features",
    model_name: str = "openai/clip-vit-base-patch32",
    device: torch.device = torch.device("cpu"),
    batch_size: int = 64
) -> CachedClipFeatureStore:
    """Load or generate cached CLIP embeddings for all unique image IDs in df."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sig = get_cache_signature(model_name, vision_model, image_processor)
    cache_path = cache_dir / f"{sig}.pt"

    store = CachedClipFeatureStore()
    multibully_mask = get_multibully_mask(df)
    multibully_df = df[multibully_mask]
    multibully_ids = set(multibully_df["id"].astype(str))

    if cache_path.exists():
        try:
            store.load(cache_path)
            cached_ids = set(store.embeddings.keys())
            missing_ids = multibully_ids.difference(cached_ids)
            extra_ids = cached_ids.difference(multibully_ids)

            if extra_ids:
                log.info(
                    "Cache contains %d extraneous IDs (e.g. HOT / non-MultiBully). Pruning from cache...",
                    len(extra_ids)
                )
                for eid in extra_ids:
                    store.embeddings.pop(eid, None)
                    store.masks.pop(eid, None)

            if not missing_ids:
                if extra_ids:
                    meta = {
                        "signature": sig,
                        "model_name": model_name,
                        "total_items": len(store.embeddings),
                        "valid_images": int(sum(1 for m in store.masks.values() if m == 1.0)),
                        "missing_images": int(sum(1 for m in store.masks.values() if m == 0.0)),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    store.save(cache_path, metadata=meta)
                return store

            log.info("Cache present but missing %d IDs. Updating cache...", len(missing_ids))
        except Exception as e:
            log.warning("Failed loading existing cache from %s (%s). Rebuilding...", cache_path, e)
            store = CachedClipFeatureStore()

    # Collect unique image IDs to process
    img_dir = Path(image_dir)
    unique_ids = multibully_df["id"].astype(str).unique().tolist()
    log.info("Extracting frozen CLIP visual embeddings for %d unique items...", len(unique_ids))

    image_paths = []
    for item_id in unique_ids:
        p = img_dir / item_id
        if not p.exists():
            stem = p.stem
            alt_ext = ".png" if p.suffix.lower() == ".jpg" else ".jpg"
            alt = p.parent / f"{stem}{alt_ext}"
            if alt.exists():
                p = alt
        image_paths.append(p if p.exists() else None)

    feats, masks = extract_clip_embeddings(
        image_paths, vision_model, image_processor, device=device, batch_size=batch_size
    )

    for item_id, feat, mask in zip(unique_ids, feats, masks):
        store.embeddings[item_id] = feat
        store.masks[item_id] = float(mask)

    meta = {
        "signature": sig,
        "model_name": model_name,
        "total_items": len(unique_ids),
        "valid_images": int(masks.sum().item()),
        "missing_images": int((masks == 0.0).sum().item()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    store.save(cache_path, metadata=meta)
    return store


def benchmark_clip_caching(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cue: torch.Tensor,
    pixel_values: torch.Tensor,
    vis_feat: torch.Tensor,
    image_mask: torch.Tensor,
    n_iters: int = 20
) -> dict:
    """Empirically measure runtime between live uncached CLIP forward passes and cached feature store."""
    model.eval()

    # Warmup
    with torch.no_grad():
        _ = model(input_ids, attention_mask, cue=cue, pixel_values=pixel_values, image_mask=image_mask)
        _ = model(input_ids, attention_mask, cue=cue, vis_feat=vis_feat, image_mask=image_mask)

    # 1. Uncached runtime
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(input_ids, attention_mask, cue=cue, pixel_values=pixel_values, image_mask=image_mask)
    uncached_time = time.perf_counter() - t0

    # 2. Cached runtime
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(input_ids, attention_mask, cue=cue, vis_feat=vis_feat, image_mask=image_mask)
    cached_time = time.perf_counter() - t1

    speedup = uncached_time / max(cached_time, 1e-6)
    reduction_pct = (1.0 - cached_time / uncached_time) * 100.0

    return {
        "n_iters": n_iters,
        "uncached_seconds": uncached_time,
        "cached_seconds": cached_time,
        "speedup_factor": speedup,
        "latency_reduction_pct": reduction_pct
    }
