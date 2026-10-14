# ToxiFusionNet v3: Publication Evaluation & Verification Plan

**Author:** Antigravity AI  
**Repository:** `/workspace/toxifusionnet`  
**Date:** October 2026  
**Audience:** Peer-Reviewed Venue (ACL / EMNLP / IEEE Trans)  

---

## 1. Evaluation Methodology

All model variants (E1c, E1d, E6b) will be evaluated on the canonical **5-Fold Grouped Cross-Validation Protocol** using identical held-out test partitions ($N_{\text{total}} = 5,718$ unique MultiBully samples).

### 1.1 Core Principles
1. **Zero Data Leakage:** Evaluated out-of-fold test samples are never seen during training or checkpoint selection.
2. **Strict Test Distribution Invariance:** No sampling, re-weighting, or synthetic augmentation is applied to validation or test distributions. Evaluation metrics reflect the true empirical distribution.
3. **No Metric-Peeking Model Selection:** Early stopping and checkpoint selection use inner validation Mean Macro-F1 exclusively; test scores are computed strictly after training concludes.
4. **Empirical Gate:** No scientific claim regarding the superiority of per-task class balancing will be asserted until E1c, E1d, and E6b have finished 5-fold evaluation and statistical significance has been computed.

---

## 2. Planned Publication Tables

### Table 1: Primary 5-Fold Pooled Macro Benchmark
Evaluated on pooled out-of-fold test predictions ($N = 5,718$):

| Model | Hate Macro-F1 | Sarcasm Macro-F1 | Target Macro-F1 | Severity Macro-F1 | Overall Mean Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **E1c (Unweighted Control)** | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ |
| **E1d (Square-Root Balanced)** | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ |
| **E6b (Combined Gold + Balanced)**| $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ | $[ \cdot ] \pm \text{CI}_{95}$ |

### Table 2: Minority-Class Performance Breakdown
Evaluates whether square-root class weighting successfully recovers representation on ultra-rare categories without collapsing majority precision:

| Class | Support ($N_{\text{eval}}$) | E1c (Unweighted) F1 / Rec | E1d (Balanced) F1 / Rec | E6b (Combined) F1 / Rec | $\Delta (\text{E1d} - \text{E1c})$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **High Severity** | 30 | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ |
| **Low Severity** | 141 | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ |
| **Moderate Severity** | 1,409 | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ |
| **Organization (Target)** | 81 | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ |
| **Society (Target)** | 68 | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ |
| **Individual (Target)** | 1,037 | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ |
| **Community (Target)** | 251 | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ / $[ \cdot ]$ | $[ \cdot ]$ |

---

## 3. Statistical Testing Framework

For each pair of comparisons:
1. **$\text{E1d vs. E1c}$ (Balancing Effect):**
   - Paired cluster-aware bootstrap test ($B = 1,000$, grouped by normalized OCR text) on Macro-F1 differences.
   - McNemar's test on binary heads (Hate, Sarcasm) with continuity correction.
   - 95% Confidence Intervals: $\Delta \in [\text{CI}_{\text{lower}}, \text{CI}_{\text{upper}}]$.
2. **$\text{E6b vs. E1d}$ (Auxiliary Text Transfer Effect):**
   - Paired test to determine whether auxiliary gold hate text significantly moves the hate Macro-F1 needle when class weighting is held strictly constant.
   - Degradation test on target and severity heads to verify that auxiliary text does not degrade conditional heads.

---

## 4. Execution Commands for DGX Environment

Upon user confirmation, the evaluation suite can be executed sequentially or in parallel on the DGX cluster:

```bash
# 1. Pull latest verified implementation
git pull origin main

# 2. Run E1c: Unweighted Control Baseline
python -m toxifusionnet.cli cv \
  --config configs/exp1c_multibully_text_unweighted.yaml \
  --out-dir results/v3_evaluation/e1c_text_unweighted

# 3. Run E1d: Per-Task Square-Root Class-Weighted Training
python -m toxifusionnet.cli cv \
  --config configs/exp1d_multibully_text_weighted.yaml \
  --out-dir results/v3_evaluation/e1d_text_weighted

# 4. Run E6b: MultiBully + HOT Gold with Identical Class Weights
python -m toxifusionnet.cli cv \
  --config configs/exp6b_combined_text_gold_weighted.yaml \
  --out-dir results/v3_evaluation/e6b_text_gold_weighted

# 5. Statistical Comparison: E1c vs. E1d (Balancing Efficacy)
python -m toxifusionnet.cli compare \
  results/v3_evaluation/e1c_text_unweighted/pooled_model.csv \
  results/v3_evaluation/e1d_text_weighted/pooled_model.csv \
  --name-a E1c_Unweighted_Control \
  --name-b E1d_Sqrt_Weighted \
  --full \
  --out-dir results/v3_evaluation/comparison_e1c_vs_e1d

# 6. Statistical Comparison: E1d vs. E6b (Auxiliary Transfer under Balancing)
python -m toxifusionnet.cli compare \
  results/v3_evaluation/e1d_text_weighted/pooled_model.csv \
  results/v3_evaluation/e6b_text_gold_weighted/pooled_model.csv \
  --name-a E1d_MultiBully_Balanced \
  --name-b E6b_Combined_Balanced \
  --full \
  --out-dir results/v3_evaluation/comparison_e1d_vs_e6b
```
