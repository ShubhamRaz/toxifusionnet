"""Multi-task loss with missing-label masking, optional class weights / focal loss, uncertainty weighting
(Kendall et al., 2018) and rationale-attention supervision (Mamun et al., 2024; Mathew et al., 2021)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..labels import IGNORE_INDEX


def focal_ce(logits, targets, weight=None, gamma: float = 2.0):
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
    return ((1 - pt) ** gamma * ce).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks, weighting: str = "fixed", weights: dict | None = None, class_weights: dict | None = None,
                 focal_gamma: float = 0.0, attention_lambda: float = 0.0,
                 severity_loss_type: str = "ce", ordinal_penalty: float = 0.5,
                 focal_use_class_weights: bool = False):
        super().__init__()
        self.tasks = list(tasks)
        self.weighting = weighting
        self.weights = {t: float((weights or {}).get(t, 1.0)) for t in self.tasks}
        self.focal_gamma = focal_gamma
        self.attention_lambda = attention_lambda
        self.severity_loss_type = severity_loss_type.lower()
        self.ordinal_penalty = float(ordinal_penalty)
        self.focal_use_class_weights = focal_use_class_weights
        for t, w in (class_weights or {}).items():
            self.register_buffer(f"cw_{t}", torch.as_tensor(w, dtype=torch.float32))
        if weighting == "uncertainty":
            self.log_vars = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in self.tasks})

    def _cw(self, t):
        return getattr(self, f"cw_{t}", None)

    @staticmethod
    def attention_loss(attn, target, valid):
        """Soft cross-entropy between model attention and ground-truth attention, over rows with a rationale."""
        if valid is None or not bool(valid.any()):
            return attn.new_zeros(())
        a, t = attn[valid], target[valid]
        return -(t * torch.log(a + 1e-9)).sum(-1).mean()

    @staticmethod
    def ordinal_distance_penalty(logits, targets):
        probs = F.softmax(logits.float(), dim=-1)
        classes = torch.arange(logits.size(-1), device=logits.device, dtype=torch.float32)
        dist = (classes.unsqueeze(0) - targets.unsqueeze(1).float()).abs()
        return (probs * dist).sum(dim=-1).mean()

    def forward(self, out, batch):
        parts: dict[str, torch.Tensor] = {}
        total = 0.0
        for t in self.tasks:
            if t not in out["logits"]:
                continue
            y = batch["labels"][t]
            valid = y != IGNORE_INDEX
            if not bool(valid.any()):
                continue
            lg, yy = out["logits"][t][valid], y[valid]
            cw = self._cw(t)
            cw = cw.to(lg.device) if cw is not None else None
            if self.focal_gamma > 0:
                focal_w = cw if self.focal_use_class_weights else None
                loss = focal_ce(lg, yy, focal_w, self.focal_gamma)
            else:
                loss = F.cross_entropy(lg, yy, weight=cw)
            if t == "severity" and self.severity_loss_type == "ordinal":
                dist_pen = self.ordinal_distance_penalty(lg, yy)
                loss = loss + self.ordinal_penalty * dist_pen
                parts["severity_ordinal_dist"] = dist_pen.detach()
            parts[t] = loss.detach()
            if self.weighting == "uncertainty":
                s = self.log_vars[t]
                total = total + 0.5 * torch.exp(-s) * loss + 0.5 * s
            else:
                total = total + self.weights[t] * loss
        if self.attention_lambda > 0:
            la = self.attention_loss(out["attn"]["hate"], batch["hate_attn_target"], batch["hate_attn_valid"])
            if "sarcasm" in out["attn"]:
                la = la + self.attention_loss(out["attn"]["sarcasm"], batch["sarc_attn_target"], batch["sarc_attn_valid"])
            parts["attention"] = la.detach()
            total = total + self.attention_lambda * la
        if not torch.is_tensor(total):
            total = out["logits"][self.tasks[0]].sum() * 0.0
        return total, {k: float(v) for k, v in parts.items()}
