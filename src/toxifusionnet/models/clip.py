"""CLIP Vision encoder and cross-modal gated fusion for ToxiFusionNet."""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

log = logging.getLogger("toxifusionnet")


class GatedVisionFusion(nn.Module):
    """Fuses projected visual features into text representation via a gated mechanism with strict modality masking.

    When image_mask == 0.0 (e.g. text-only HOT samples or missing MultiBully images),
    the output is mathematically identical to text_vec with zero visual perturbation.
    """

    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Linear(2 * d, d)
        self.v_trans = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, text_vec: torch.Tensor, vis_vec: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Args:

        text_vec: (B, d) contextual text representation
        vis_vec: (B, d) projected visual representation
        image_mask: (B, 1) or (B,) binary indicator: 1.0 if image exists, 0.0 if absent
        """
        g = torch.sigmoid(self.gate(torch.cat([text_vec, vis_vec], dim=-1)))
        fused = self.norm(text_vec + g * torch.tanh(self.v_trans(vis_vec)))
        if image_mask is not None:
            if image_mask.dim() == 1:
                image_mask = image_mask.unsqueeze(-1)
            # Strict guarantee: pure text representation preserved when image_mask == 0
            return text_vec * (1.0 - image_mask) + fused * image_mask
        return fused


def build_clip_vision(cfg):
    """Instantiate CLIP vision model and image processor.

    Supports pretrained checkpoint or lightweight random config for offline smoke testing.
    """
    model_name = cfg.get("clip_model_name", "openai/clip-vit-base-patch32")
    offline = bool(cfg.get("clip_offline", False))

    if offline:
        clip_cfg = CLIPVisionConfig(
            hidden_size=int(cfg.get("clip_hidden_size", 768)),
            image_size=224,
            patch_size=32,
            num_hidden_layers=int(cfg.get("clip_layers", 2)),
            num_attention_heads=int(cfg.get("clip_heads", 4)),
            intermediate_size=1024,
        )
        vision_model = CLIPVisionModel(clip_cfg)
        processor = CLIPImageProcessor(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224})
        log.info("Initialized offline/synthetic CLIP Vision model (%d parameters)", sum(p.numel() for p in vision_model.parameters()))
        return vision_model, processor, clip_cfg.hidden_size

    try:
        processor = CLIPImageProcessor.from_pretrained(model_name)
        vision_model = CLIPVisionModel.from_pretrained(model_name, use_safetensors=False)
        log.info("Loaded pretrained CLIP Vision model '%s' (hidden_size=%d)", model_name, vision_model.config.hidden_size)
        return vision_model, processor, vision_model.config.hidden_size
    except Exception as e:
        log.warning("Failed to load pretrained CLIP from Hub (%s). Falling back to offline architecture.", e)
        clip_cfg = CLIPVisionConfig(hidden_size=768, image_size=224, patch_size=32, num_hidden_layers=2, num_attention_heads=4)
        vision_model = CLIPVisionModel(clip_cfg)
        processor = CLIPImageProcessor(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224})
        return vision_model, processor, clip_cfg.hidden_size
