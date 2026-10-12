"""ToxiFusionNet: sarcasm-aware multi-task transformer for hate speech detection.

                       +--> sarcasm attention pooling + cue features --> sarcasm head (2)
 text --> encoder --H--+                                          \\
                       +--> hate attention pooling -- gated sarcasm fusion --> hate head (2)
                                                                   \\--> target head (5)  [hate only]
                                                                    \\-> severity head (3) [hate only]

Switches in the model config allow all ablations: ``use_sarcasm_branch``, ``use_fusion``, ``use_cue_features``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..labels import num_classes
from ..preprocessing.features import CUE_DIM
from .clip import GatedVisionFusion, build_clip_vision


class AttentionPooling(nn.Module):
    """Additive attention over tokens; the attention weights double as a token-level rationale."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.score = nn.Linear(d, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        s = self.score(torch.tanh(self.proj(h))).squeeze(-1).float()
        s = s.masked_fill(mask == 0, -1e4)
        a = torch.softmax(s, dim=-1).to(h.dtype)
        return torch.bmm(a.unsqueeze(1), h).squeeze(1), a


class SarcasmAwareFusion(nn.Module):
    """Gated fusion: the hate representation is corrected by a sarcasm representation, gate conditioned on P(sarcasm).

    fused = LayerNorm(h + g * tanh(W s)),  g = sigmoid(W_g [h ; s ; p_sarcasm])
    Intuition: when a post is sarcastic the surface sentiment is inverted, so the gate lets the sarcasm signal re-write
    the hate evidence instead of the classifier trusting the (positive-looking) surface form.
    """

    def __init__(self, d: int):
        super().__init__()
        self.s_proj = nn.Linear(d, d)
        self.gate = nn.Linear(2 * d + 1, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h_vec, s_vec, p_sarc):
        g = torch.sigmoid(self.gate(torch.cat([h_vec, s_vec, p_sarc], dim=-1)))
        return self.norm(h_vec + g * torch.tanh(self.s_proj(s_vec))), g


def _mlp(i: int, h: int, o: int, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class ToxiFusionNet(nn.Module):
    def __init__(self, encoder: nn.Module, cfg):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        p = float(cfg.get("dropout", 0.1))
        hid = int(cfg.get("head_hidden", 256))
        self.tasks = list(cfg.get("tasks", ["sarcasm", "hate", "target", "severity"]))
        self.use_sarc = bool(cfg.get("use_sarcasm_branch", True)) and "sarcasm" in self.tasks
        self.use_fusion = bool(cfg.get("use_fusion", True)) and self.use_sarc
        self.use_cue = bool(cfg.get("use_cue_features", True)) and self.use_sarc
        self.detach_gate = bool(cfg.get("detach_gate_prob", False))
        self.drop = nn.Dropout(p)
        self.hate_pool = AttentionPooling(d)

        # Multimodal CLIP vision integration
        self.use_clip = bool(cfg.get("use_clip", False))
        self.freeze_clip = bool(cfg.get("freeze_clip", True))
        if self.use_clip:
            self.clip_vision, self.clip_processor, clip_dim = build_clip_vision(cfg)
            if self.freeze_clip:
                for param in self.clip_vision.parameters():
                    param.requires_grad = False
            self.vis_proj = nn.Sequential(nn.Linear(clip_dim, d), nn.GELU(), nn.Dropout(p))
            self.vis_fusion_hate = GatedVisionFusion(d)
            if self.use_sarc:
                self.vis_fusion_sarc = GatedVisionFusion(d)

        if self.use_sarc:
            self.sarc_pool = AttentionPooling(d)
            cue_h = int(cfg.get("cue_hidden", 32))
            self.cue_proj = nn.Sequential(nn.Linear(int(cfg.get("cue_dim", CUE_DIM)), cue_h), nn.GELU()) if self.use_cue else None
            self.sarc_mlp = nn.Sequential(nn.Linear(d + (cue_h if self.use_cue else 0), hid), nn.GELU(), nn.Dropout(p))
            self.sarc_out = nn.Linear(hid, num_classes("sarcasm"))
            self.sarc_to_d = nn.Linear(hid, d)
        self.fusion = SarcasmAwareFusion(d) if self.use_fusion else None
        self.heads = nn.ModuleDict({t: _mlp(d, hid, num_classes(t), p) for t in self.tasks if t != "sarcasm"})

    def forward(self, input_ids, attention_mask, cue=None, pixel_values=None, image_mask=None, **_):
        h = self.drop(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        hate_vec, hate_attn = self.hate_pool(h, attention_mask)

        # Multimodal fusion if CLIP vision is enabled and inputs are provided
        vis_proj = None
        if self.use_clip and pixel_values is not None:
            with torch.set_grad_enabled(not self.freeze_clip and self.training):
                vis_feat = self.clip_vision(pixel_values=pixel_values).pooler_output
            vis_proj = self.vis_proj(vis_feat)
            hate_vec = self.vis_fusion_hate(hate_vec, vis_proj, image_mask)

        out: dict = {"logits": {}, "attn": {"hate": hate_attn}, "gate": None}
        fused = hate_vec
        if self.use_sarc:
            s_vec, sarc_attn = self.sarc_pool(h, attention_mask)
            if self.use_clip and vis_proj is not None:
                s_vec = self.vis_fusion_sarc(s_vec, vis_proj, image_mask)
            parts = [s_vec]
            if self.use_cue:
                parts.append(self.cue_proj(cue if cue is not None else torch.zeros(h.size(0), self.cue_proj[0].in_features, device=h.device)))
            s_hid = self.sarc_mlp(torch.cat(parts, dim=-1))
            s_logits = self.sarc_out(s_hid)
            out["logits"]["sarcasm"] = s_logits
            out["attn"]["sarcasm"] = sarc_attn
            if self.fusion is not None:
                p = F.softmax(s_logits.float(), dim=-1)[:, 1:2].to(h.dtype)
                fused, out["gate"] = self.fusion(hate_vec, self.sarc_to_d(s_hid), p.detach() if self.detach_gate else p)
        for t, head in self.heads.items():
            out["logits"][t] = head(fused)
        return out

    def param_groups(self, lr_encoder: float, lr_head: float, weight_decay: float = 0.01, extra=(), emb_lr_mult: float = 1.0):
        enc, head, emb, vis = [], [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("clip_vision"):
                vis.append(p)
            elif n.startswith("encoder.embeddings.word_embeddings"):
                emb.append(p)
            elif n.startswith("encoder."):
                enc.append(p)
            else:
                head.append(p)
        groups = [{"params": enc, "lr": lr_encoder, "weight_decay": weight_decay},
                  {"params": head, "lr": lr_head, "weight_decay": weight_decay}]
        if emb:
            groups.append({"params": emb, "lr": lr_encoder * emb_lr_mult, "weight_decay": 0.0})
        if vis:
            groups.append({"params": vis, "lr": lr_encoder, "weight_decay": weight_decay})
        if extra:
            groups.append({"params": list(extra), "lr": lr_head, "weight_decay": 0.0})
        return groups
