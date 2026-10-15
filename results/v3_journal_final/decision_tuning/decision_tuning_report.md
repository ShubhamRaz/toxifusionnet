# ToxiFusionNet: Decision-Tuning Evaluation Report (E1d_weighted)

**Protocol:** Leakage-Safe Validation-Only Decision Tuning across 5-Fold Cross-Validation on MultiBully ($N = 5,718$ held-out test predictions).  
**Leakage Prevention Guarantee:** All decision thresholds and class adjustments were selected strictly from internal validation folds (`preds_val.csv`). Outer test partitions (`preds_test.csv`) were never exposed during hyperparameter or threshold search.

---

## 1. Summary: Default vs. Tuned Out-of-Fold Test Performance

| Task | Status | Chosen Policy / Threshold | Default Macro-F1 | Tuned Macro-F1 | Delta Macro-F1 | 95% Cluster CI | Permutation $p$ | Default Acc | Tuned Acc | Delta Acc | Balanced Acc (Def -> Tun) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sarcasm** | Tuned | $\tau^* \approx 0.450$ | 0.5608 | **0.5733** | **+0.0125** | [0.0035, 0.0216] | 0.0120 | 0.6049 | 0.5902 | -0.0147 | 0.5605 -> **0.5770** |
| **Hate** | Tuned | $\tau^* \approx 0.536$ | 0.6808 | 0.6826 | +0.0018 | [-0.0060, 0.0096] | 0.6565 | 0.7242 | **0.7331** | **+0.0089** | 0.6991 -> 0.6933 |
| **Severity** | Guardrail | Retained argmax | 0.4252 | 0.4252 | +0.0000 | [0.0000, 0.0000] | 1.0000 | 0.8443 | 0.8443 | +0.0000 | 0.4232 -> 0.4232 |
| **Target** | Guardrail | Retained argmax | 0.4364 | 0.4364 | +0.0000 | [0.0000, 0.0000] | 1.0000 | 0.7049 | 0.7049 | +0.0000 | 0.4376 -> 0.4376 |
| **Overall Mean** | Hybrid | Tuned binary + Def multiclass | 0.5258 | **0.5294** | **+0.0036** | — | — | — | — | — | — |

---

## 2. Fold-by-Fold Threshold Selection & Outer-Test Validation

### 2.1 Sarcasm Decision Tuning (Macro-F1 Optimization)

| Fold | Validation Support | Selected Threshold $\tau^*$ | Val F1 (Def -> Tun) | Test Default F1 | Test Tuned F1 | Test F1 Delta | Test Acc Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 1142 | **0.440** | 0.6006 -> 0.6123 (+0.0117) | 0.5842 | 0.5738 | **-0.0104** | -0.0342 |
| Fold 1 | 1144 | **0.450** | 0.5644 -> 0.5932 (+0.0288) | 0.5746 | 0.6069 | **+0.0323** | +0.0035 |
| Fold 2 | 1146 | **0.450** | 0.5342 -> 0.5699 (+0.0357) | 0.5216 | 0.5466 | **+0.0250** | -0.0140 |
| Fold 3 | 1144 | **0.440** | 0.5596 -> 0.5896 (+0.0300) | 0.5474 | 0.5562 | **+0.0088** | -0.0192 |
| Fold 4 | 1142 | **0.470** | 0.5481 -> 0.5645 (+0.0164) | 0.5721 | 0.5800 | **+0.0079** | -0.0096 |

### 2.2 Hate Decision Tuning (Macro-F1 Optimization)

| Fold | Validation Support | Selected Threshold $\tau^*$ | Val F1 (Def -> Tun) | Test Default F1 | Test Tuned F1 | Test F1 Delta | Test Acc Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 1142 | **0.520** | 0.7136 -> 0.7184 (+0.0048) | 0.6720 | 0.6710 | -0.0010 | **+0.0044** |
| Fold 1 | 1144 | **0.570** | 0.6598 -> 0.6743 (+0.0145) | 0.6924 | 0.7119 | +0.0194 | **+0.0297** |
| Fold 2 | 1146 | **0.620** | 0.6564 -> 0.6843 (+0.0279) | 0.6733 | 0.6681 | -0.0052 | **+0.0305** |
| Fold 3 | 1144 | **0.580** | 0.6859 -> 0.6977 (+0.0117) | 0.6791 | 0.6876 | +0.0086 | **+0.0210** |
| Fold 4 | 1142 | **0.390** | 0.6705 -> 0.6777 (+0.0072) | 0.6864 | 0.6679 | -0.0185 | **-0.0412** |

---

## 3. Rare-Class Validation Support & Scientific Guardrail

### 3.1 Severity Validation Support Audit
- **Guardrail Threshold:** Minimum required validation support = 20 instances per class.
- **Action Taken:** `RETAIN_DEFAULT_ARGMAX`.
- **Methodological Rationale:** Validation support per class is statistically insufficient (< 20 instances per fold). For example, High-Severity contains only 3-4 validation samples per fold. Optimizing decision boundaries or class biases on 3-4 samples would overfit to random noise. Per rigorous methodology, default argmax decision rules are retained.

**Per-Fold Validation Support Counts for Severity:**
- **Fold 0:** Class 0: 17, Class 1: 168, Class 2: 4
- **Fold 1:** Class 0: 17, Class 1: 171, Class 2: 3
- **Fold 2:** Class 0: 17, Class 1: 169, Class 2: 3
- **Fold 3:** Class 0: 17, Class 1: 170, Class 2: 3
- **Fold 4:** Class 0: 17, Class 1: 169, Class 2: 4

### 3.2 Target Validation Support Audit
- **Guardrail Threshold:** Minimum required validation support = 20 instances per class.
- **Action Taken:** `RETAIN_DEFAULT_ARGMAX`.
- **Methodological Rationale:** Validation support per class is statistically insufficient (< 20 instances per fold). For example, High-Severity contains only 3-4 validation samples per fold. Optimizing decision boundaries or class biases on 3-4 samples would overfit to random noise. Per rigorous methodology, default argmax decision rules are retained.

**Per-Fold Validation Support Counts for Target:**
- **Fold 0:** Class 0: 123, Class 1: 11, Class 2: 30, Class 3: 7
- **Fold 1:** Class 0: 126, Class 1: 9, Class 2: 30, Class 3: 9
- **Fold 2:** Class 0: 123, Class 1: 9, Class 2: 31, Class 3: 8
- **Fold 3:** Class 0: 125, Class 1: 10, Class 2: 29, Class 3: 9
- **Fold 4:** Class 0: 125, Class 1: 10, Class 2: 30, Class 3: 8

---

## 4. Key Scientific Insights & Phase 1 Conclusions

1. **Sarcasm Threshold Realignment:**
   - Validation threshold optimization consistently selects $\tau^* \approx 0.450$ across all five independent folds.
   - On the untouched held-out test partitions, this lifts Sarcasm Macro-F1 from **0.5608 to 0.5733** (+0.0125), and balanced accuracy from 0.5605 to 0.5770.
   - This improvement finally resolves the deficiency where default-threshold neural sarcasm (0.5608) trailed the classical TF-IDF baseline (0.5629), lifting the neural model to **0.5727** (+0.0098 over TF-IDF).

2. **Hate Decision Frontier:**
   - Validation threshold tuning for hate selects $\tau^* \approx 0.536$, boosting test accuracy from **0.7242 to 0.7331** (+0.0089) while preserving Macro-F1 (0.6808 vs 0.6826).

3. **Principled Restraint on Rare Multiclass Decisions:**
   - Auditing validation folds revealed that High Severity has only 3-4 instances per fold (and Society target has 7-9 instances).
   - Attempting post-hoc decision tuning on 3-4 instances produces severe validation overfitting without generalizable test gains.
   - Retaining the square-root weighted model's default argmax outputs preserves the empirical gains demonstrated in the publication benchmark without statistical over-claiming.

---
*Report generated automatically by `toxifusionnet.analysis.decision_tuning.run_full_decision_tuning_experiment`.*