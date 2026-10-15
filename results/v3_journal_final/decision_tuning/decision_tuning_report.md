# ToxiFusionNet: Decision-Tuning Evaluation Report (E1d_weighted)

**Protocol:** Leakage-Safe Validation-Only Decision Tuning across 5-Fold Cross-Validation on MultiBully ($N = 5,718$ held-out test predictions).  
**Leakage Prevention Guarantee:** All decision thresholds and class adjustments were selected strictly from internal validation folds (`preds_val.csv`). Outer test partitions (`preds_test.csv`) were never exposed during hyperparameter or threshold search.

---

## 1. Summary: Default vs. Tuned Out-of-Fold Test Performance

| Task | Status | Chosen Policy / Threshold | Default Macro-F1 | Tuned Macro-F1 | Delta Macro-F1 | 95% Cluster CI | Raw Perm $p$ | Holm Adj $p$ | Default Acc | Tuned Acc | Delta Acc | Balanced Acc (Def -> Tun) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sarcasm** | Tuned | $\tau^* \approx 0.450$ | 0.5608 | **0.5733** | **+0.0125** | [0.0035, 0.0216] | 0.0120 | 0.0240 | 0.6049 | 0.5902 | -0.0147 | 0.5605 -> **0.5770** |
| **Hate** | Tuned | $\tau^* \approx 0.536$ | 0.6808 | 0.6826 | +0.0018 | [-0.0060, 0.0096] | 0.6565 | 0.6565 | 0.7242 | **0.7331** | **+0.0089** | 0.6991 -> 0.6933 |
| **Severity** | Guardrail | Retained argmax | 0.4252 | 0.4252 | +0.0000 | [0.0000, 0.0000] | — | — | 0.8443 | 0.8443 | +0.0000 | 0.4232 -> 0.4232 |
| **Target** | Guardrail | Retained argmax | 0.4364 | 0.4364 | +0.0000 | [0.0000, 0.0000] | — | — | 0.7049 | 0.7049 | +0.0000 | 0.4376 -> 0.4376 |
| **Overall Mean** | Hybrid | Tuned binary + Def multiclass | 0.5258 | **0.5294** | **+0.0036** | — | — | — | — | — | — | — |

**Hypothesis Testing & Multiplicity Control:**
- **Hypothesis Testing Family:** $M = 2$ tuned binary decision tasks ({sarcasm, hate}) evaluated on untouched outer test folds.
- **Multiplicity Correction:** Holm-Bonferroni step-down FWER control ($\alpha = 0.05$). Sarcasm gain is statistically significant after correction ($p_{raw} = 0.0120, p_{Holm} = 0.0240 < 0.05$). Hate F1 difference is not statistically significant ($p_{raw} = 0.6565, p_{Holm} = 0.6565$).
- **Multiclass Tasks:** Retained default argmax decisions under the sample-support guardrail ($H_0$ maintained; no post-hoc tuning).
- **Evidence Level:** All reported evaluations represent single-experiment, five-fold cross-validation evidence on reference run E1d. They do not claim invariance across different training seeds until multi-seed experiments are verified.

---

## 2. Fold-by-Fold Threshold Selection & Outer-Test Validation

### 2.1 Sarcasm Decision Tuning (Macro-F1 Optimization)

| Fold | Val Support | Test Support | Selected Threshold $\tau^*$ | Val F1 (Def -> Tun) | Test Default F1 | Test Tuned F1 | Test F1 Delta | Test Acc Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 687 | 1142 | **0.440** | 0.6006 -> 0.6123 (+0.0117) | 0.5842 | 0.5738 | **-0.0104** | -0.0342 |
| Fold 1 | 689 | 1144 | **0.450** | 0.5644 -> 0.5932 (+0.0288) | 0.5746 | 0.6069 | **+0.0323** | +0.0035 |
| Fold 2 | 684 | 1146 | **0.450** | 0.5342 -> 0.5699 (+0.0357) | 0.5216 | 0.5466 | **+0.0250** | -0.0140 |
| Fold 3 | 693 | 1144 | **0.440** | 0.5596 -> 0.5896 (+0.0300) | 0.5474 | 0.5562 | **+0.0088** | -0.0192 |
| Fold 4 | 689 | 1142 | **0.470** | 0.5481 -> 0.5645 (+0.0164) | 0.5721 | 0.5800 | **+0.0079** | -0.0096 |

### 2.2 Hate Decision Tuning (Macro-F1 Optimization)

| Fold | Val Support | Test Support | Selected Threshold $\tau^*$ | Val F1 (Def -> Tun) | Test Default F1 | Test Tuned F1 | Test F1 Delta | Test Acc Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 687 | 1142 | **0.520** | 0.7136 -> 0.7184 (+0.0048) | 0.6720 | 0.6710 | -0.0010 | **+0.0044** |
| Fold 1 | 689 | 1144 | **0.570** | 0.6598 -> 0.6743 (+0.0145) | 0.6924 | 0.7119 | +0.0194 | **+0.0297** |
| Fold 2 | 684 | 1146 | **0.620** | 0.6564 -> 0.6843 (+0.0279) | 0.6733 | 0.6681 | -0.0052 | **+0.0305** |
| Fold 3 | 693 | 1144 | **0.580** | 0.6859 -> 0.6977 (+0.0117) | 0.6791 | 0.6876 | +0.0086 | **+0.0210** |
| Fold 4 | 689 | 1142 | **0.390** | 0.6705 -> 0.6777 (+0.0072) | 0.6864 | 0.6679 | -0.0185 | **-0.0412** |

---

## 3. Rare-Class Validation Support & Scientific Guardrail

### 3.1 Severity Validation Support Audit
- **Guardrail Threshold:** Minimum required validation support = 20 instances per class.
- **Action Taken:** `RETAIN_DEFAULT_ARGMAX`.
- **Methodological Rationale:** Validation support for severity classes ('low', 'moderate', 'high') is statistically insufficient (< 20 instances per fold). Specifically, High-Severity contains only 3-4 validation samples and Low-Severity contains only 17 validation samples per fold. Optimizing decision boundaries or class biases on 3-4 validation samples would overfit to random noise. Per rigorous methodology, default argmax decision rules are retained.

**Per-Fold Validation Support Counts for Severity:**
- **Fold 0:** low: 17, moderate: 168, high: 4
- **Fold 1:** low: 17, moderate: 171, high: 3
- **Fold 2:** low: 17, moderate: 169, high: 3
- **Fold 3:** low: 17, moderate: 170, high: 3
- **Fold 4:** low: 17, moderate: 169, high: 4

### 3.2 Target Validation Support Audit
- **Guardrail Threshold:** Minimum required validation support = 20 instances per class.
- **Action Taken:** `RETAIN_DEFAULT_ARGMAX`.
- **Methodological Rationale:** Validation support for MultiBully target classes ('individual', 'organization', 'community', 'society') is statistically insufficient (< 20 instances per fold). Specifically, 'society' has only 7-9 validation instances and 'organization' has only 9-11 validation instances per fold. Optimizing decision boundaries or class biases on 7-9 validation samples risks extreme variance and severe validation overfitting. Per rigorous methodology, default argmax decision rules are retained.

**Per-Fold Validation Support Counts for Target (MultiBully Taxonomy):**
- **Fold 0:** individual: 123, organization: 11, community: 30, society: 7
- **Fold 1:** individual: 126, organization: 9, community: 30, society: 9
- **Fold 2:** individual: 123, organization: 9, community: 31, society: 8
- **Fold 3:** individual: 125, organization: 10, community: 29, society: 9
- **Fold 4:** individual: 125, organization: 10, community: 30, society: 8

---

## 4. Key Scientific Insights & Phase 1 Conclusions

1. **Sarcasm Threshold Realignment:**
   - Validation threshold optimization consistently selects $\tau^* \approx 0.450$ across all five independent folds.
   - On the untouched held-out test partitions, this lifts Sarcasm Macro-F1 from **0.5608 to 0.5733** (+0.0125; raw permutation $p = 0.0120$, Holm-adjusted $p = 0.0240$), and balanced accuracy from 0.5605 to 0.5770.
   - Compared against the classical TF-IDF baseline (0.5629), validation-tuned neural sarcasm achieves **0.5733** (+0.0104 over TF-IDF), resolving the deficiency where default-threshold neural sarcasm (0.5608) trailed TF-IDF.

2. **Hate Decision Frontier:**
   - Validation threshold tuning for hate selects $\tau^* \approx 0.536$, boosting test accuracy from **0.7242 to 0.7331** (+0.0089) while preserving Macro-F1 (0.6808 vs 0.6826; raw permutation $p = 0.6565$, Holm-adjusted $p = 0.6565$).

3. **Principled Restraint on Rare Multiclass Decisions:**
   - Auditing validation folds revealed that High-Severity has only 3-4 instances per fold (and Society target has 7-9 instances).
   - Attempting post-hoc decision tuning on 3-4 instances produces severe validation overfitting without generalizable test gains.
   - Retaining the square-root weighted model's default argmax outputs preserves the empirical gains demonstrated in the publication benchmark without statistical over-claiming.

4. **Scope of Evidence & Isolation Guarantee:**
   - **Single-Experiment Evidence:** All reported evaluations represent 5-fold cross-validation evidence from reference run E1d. They do not claim invariance across different random training seeds until multi-seed experiments are verified.
   - **Strict Isolation:** Original predictions, checkpoints, and outer test partitions (`preds_test.csv`) were never exposed during hyperparameter search, threshold selection, or model selection.

---
*Report generated automatically by `toxifusionnet.analysis.decision_tuning.run_full_decision_tuning_experiment`.*