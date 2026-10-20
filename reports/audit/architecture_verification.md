# Model Architecture & Implementation Verification Audit

---

## 1. Architectural Component Verification Matrix

We inspected the underlying PyTorch source code in [`src/toxifusionnet/models/toxifusionnet.py`](file:///c:/Users/SHUBHAM%20RAJ/Desktop/Others/Toxic/toxifusionnet/toxifusionnet/src/toxifusionnet/models/toxifusionnet.py) and compared each component against documented claims:

| Documented Component | Claimed Description in Reports | Actual Implementation in Code | Verification Status | Correction Required |
| :--- | :--- | :--- | :---: | :--- |
| **Encoder Backbone** | Multilingual XLM-RoBERTa (`xlm-roberta-base`) | `AutoModel.from_pretrained("xlm-roberta-base")` | **VERIFIED ✅** | None |
| **Pragmatic Surface Cues** | Vector of emojis, caps, punctuation | `CUE_DIM = 7`, `cue_proj = nn.Linear(7, 32)` | **VERIFIED ✅** | None |
| **Sarcasm Branch** | Dedicated branch predicting sarcasm | `sarc_mlp` + `sarc_out` predicting binary logits | **VERIFIED ✅** | None |
| **Fusion Mechanism** | **"Gated Sarcasm Cross-Attention Fusion"** | **Gated Residual Fusion Block** with LayerNorm | **CONTRADICTED ❌** | **Rename from "Cross-Attention" to "Gated Residual Fusion"** |
| **Sarcasm Influence on Hate**| Conditioned gate modulates hate logits | $g = \sigma(W_g [h; s; P_s])$, $f = \text{LN}(h + g \odot \tanh(W s))$ | **VERIFIED ✅** | None |
| **Hierarchical Heads** | Multi-task heads for hate, target, severity | Shared MLP heads conditioned on `fused` | **VERIFIED ✅** | None |
| **Explainability / Rationales**| "Token-level attribution rationales" | Additive Attention Weights from pooling layer | **PARTIALLY VERIFIED ⚠️** | **Clarify as attention weights, not SHAP/IG** |

---

## 2. In-Depth Analysis: Why "Cross-Attention" is Technically Inaccurate

### The Code Implementation (`toxifusionnet.py`, Lines 37–54)
```python
class SarcasmAwareFusion(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.s_proj = nn.Linear(d, d)
        self.gate = nn.Linear(2 * d + 1, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h_vec, s_vec, p_sarc):
        g = torch.sigmoid(self.gate(torch.cat([h_vec, s_vec, p_sarc], dim=-1)))
        return self.norm(h_vec + g * torch.tanh(self.s_proj(s_vec))), g
```

### Technical Critique
1. **Mathematical Reality:**
   - Multi-Head Cross-Attention requires Key ($K$), Query ($Q$), and Value ($V$) projections across sequence dimensions ($L \times D$) followed by scaled dot-product attention: $\text{softmax}(QK^T / \sqrt{d_k})V$.
   - The code above operates on pooled sentence-level vectors ($h_{\text{vec}}, s_{\text{vec}} \in \mathbb{R}^d$) using an element-wise sigmoid gating vector $g \in [0, 1]^d$.
2. **Proper Terminology:**
   - This architecture is properly described as a **Gated Residual Fusion Block with Sarcasm Probability Conditioning** or **Gated Feature Fusion**.
   - An academic reviewer will flag "Cross-Attention" as a factual error.

---

## 3. Parameter Count & Model Geometry

Directly computed from the instantiated `ToxiFusionNet` model object:

```text
======================================================
ToxiFusionNet Model Parameters Breakdown
======================================================
Backbone Encoder (xlm-roberta-base) : 278,043,648 parameters (98.60%)
Task Heads & Gated Fusion Block     :   3,952,684 parameters ( 1.40%)
──────────────────────────────────────────────────────
Total Parameters                    : 281,996,332 parameters
Trainable Parameters                : 281,996,332 parameters (100.0%)
======================================================
```

- **Encoder Hidden Dimension ($d$):** 768
- **Head Hidden Dimension:** 256
- **Surface Cue Dimension:** 7
- **Dropout:** 0.10

---

## 4. Token-Level Attribution Rationale Verification

### How Rationales are Generated
In `cli.py` (`_explain_text`), token rationales are extracted from `AttentionPooling`:
```python
s = self.score(torch.tanh(self.proj(h))).squeeze(-1).float()
a = torch.softmax(s, dim=-1)
```
- **Finding:** Rationale scores are **additive attention weights** normalized across the sequence.
- **Academic Standard:** While attention weights can serve as heuristic feature importance, recent literature (e.g., *Jain & Wallace, "Attention is not Explanation"*, NAACL 2019) mandates that papers explicitly specify that rationales are derived from attention weights, rather than gradient-based attribution methods.
