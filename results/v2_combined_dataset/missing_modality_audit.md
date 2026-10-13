# ToxiFusionNet E6/E7: Missing Modality & Visual Masking Audit

**Document:** `results/v2_combined_dataset/missing_modality_audit.md`  
**Date:** October 13, 2026  
**Status:** MATHEMATICALLY & EMPIRICALLY VERIFIED  
**Architecture:** ToxiFusionNet with XLM-RoBERTa + Frozen CLIP ViT-B/32  

---

## 1. Executive Summary

This audit verifies that missing visual modalities (which occur for all 3,189 text-only HOT samples and 32 unreadable MultiBully meme files) are handled without fabricating images, corrupting representations, or leaking dummy visual features.

### Audit Verdict: **PASSED (ZERO VISUAL PERTURBATION GUARANTEED)**
1. **No Fabricated Images**: HOT samples are never paired with arbitrary memes, black frames treated as real images, or random placeholders.
2. **Explicit Missing Modality Flag**: `HateSarcasmDataset` assigns `has_image = 0.0` for all HOT samples.
3. **Mathematical Identity Proof**: In `GatedVisionFusion`, when `image_mask == 0.0`, the output tensor is mathematically identical to the pure text representation:
   $$\vec{h}_{\text{fused}} = \vec{h}_{\text{text}} \cdot (1 - m) + \vec{h}_{\text{vision\_fused}} \cdot m = \vec{h}_{\text{text}} \quad \text{when } m = 0.0$$
4. **Gradient Isolation**: Visual projection weights $\mathbf{W}_{\text{vis}}$ receive exactly zero gradient from image-absent samples.
5. **Fail-Fast Safeguard Active**: Pretrained CLIP model loading (`build_clip_vision`) strictly forbids silent random-weight fallback in research runs.

---

## 2. Mathematical & Architectural Specification

### 2.1 The Gated Vision Fusion Mechanism
In `src/toxifusionnet/models/clip.py`:
```python
class GatedVisionFusion(nn.Module):

  def __init__(self, d: int):
    super().__init__()
    self.gate = nn.Linear(2 * d, d)
    self.v_trans = nn.Linear(d, d)
    self.norm = nn.LayerNorm(d)

  def forward(
      self,
      text_vec: torch.Tensor,
      vis_vec: torch.Tensor,
      image_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    g = torch.sigmoid(self.gate(torch.cat([text_vec, vis_vec], dim=-1)))
    fused = self.norm(text_vec + g * torch.tanh(self.v_trans(vis_vec)))
    if image_mask is not None:
      if image_mask.dim() == 1:
        image_mask = image_mask.unsqueeze(-1)
      return text_vec * (1.0 - image_mask) + fused * image_mask
    return fused
```

### 2.2 Proof of Invariance Under Missing Modality
Let $\vec{x}_{\text{text}} \in \mathbb{R}^d$ be the contextual text representation pooled from XLM-RoBERTa.  
Let $\vec{v}_{\text{dummy}} \in \mathbb{R}^d$ be any arbitrary dummy visual vector passed through the vision projection.  
When $m = 0.0$:
$$\text{Output} = \vec{x}_{\text{text}} \cdot (1.0 - 0.0) + \text{fused}(\vec{x}_{\text{text}}, \vec{v}_{\text{dummy}}) \cdot 0.0 = \vec{x}_{\text{text}}$$
The maximum absolute difference between `Output` and `text_vec` is bounded by:
$$\max_{i} |\text{Output}_i - \vec{x}_{\text{text}, i}| = 0.000000000$$

### 2.3 Empirical Unit Test Verification
Automated test `test_missing_image_modality_masking` in `tests/test_combined_dataset_cv.py`:
- Generated random $8 \times 64$ text vectors and random $8 \times 64$ visual vectors.
- Evaluated batch with `image_mask = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]`.
- Verified `(out[4:] - text_vec[4:]).abs().max().item() == 0.0`.
- Verified `(out[:4] - text_vec[:4]).abs().max().item() > 0.0`.
- **Status: PASSED (Verified across both CPU and CUDA)**.

---

## 3. Data Loading Invariant Summary

| Modality / Field | MultiBully Memes ($N=5,718$) | HOT Tweets ($N=3,189$) |
|---|:---|:---|
| **Text Modality** | Present (OCR text from meme) | Present (Clean Hindi-English tweet) |
| **Image Modality** | Present (Decoded JPEG/PNG image) | **Absent (`has_image = 0.0`)** |
| **Image Tensor** | Normalized $3 \times 224 \times 224$ float32 | Zero Tensor $\mathbf{0}^{3 \times 224 \times 224}$ |
| **Image Mask** | `1.0` (or `0.0` if corrupt file) | **Strictly `0.0`** |
| **CLIP Vision Encoder** | Evaluated (Features used in fusion) | **Masked (Features discarded by $m=0$)** |
