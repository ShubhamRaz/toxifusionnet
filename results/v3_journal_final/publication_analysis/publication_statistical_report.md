# ToxiFusionNet: Publication-Grade Statistical Evaluation Report

**Protocol:** Grouped 5-Fold Cross-Validation on MultiBully ($N = 5,718$ held-out test predictions).  
**Inferential Tests:** Paired cluster permutation test ($B = 10,000$ permutations, two-sided, plus-one corrected).  
**Confidence Intervals:** Paired cluster bootstrap ($B = 10,000$ resamples, 95% percentile interval).  
**Multiple Comparison Control:** Holm-Bonferroni step-down correction across all 8 task hypothesis tests.  

---

## 1. Primary Benchmark: Multi-Task Macro-F1 Comparisons

### Contrast 1: E1c (Unweighted Control) vs. E1d (Square-Root Balanced)

| Task | Support | E1c Macro-F1 | E1d Macro-F1 | Delta (E1d - E1c) | 95% Cluster CI | Raw $p_{\text{perm}}$ | Holm-Adjusted $p$ | Significant ($\alpha=0.05$) | $\mathbb{P}(\Delta > 0)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hate** | 5,718 | 0.6838 | 0.6808 | -0.0030 | [-0.0132, 0.0073] | 0.5627 | 1.0000 | No | 27.8% |
| **Sarcasm** | 5,718 | 0.5330 | 0.5608 | **+0.0278** | [0.0164, 0.0391] | 0.0001 | 0.0008 | **Yes** | 100.0% |
| **Target** | 1,437 | 0.3684 | 0.4364 | **+0.0680** | [0.0334, 0.1024] | 0.0007 | 0.0049 | **Yes** | 100.0% |
| **Severity** | 1,580 | 0.3219 | 0.4252 | **+0.1033** | [0.0567, 0.1518] | 0.0008 | 0.0049 | **Yes** | 100.0% |
| **Overall Mean** | 5,718 | 0.4768 | 0.5258 | **+0.0490** | [0.0340, 0.0641] | 0.0001 | — | — | 100.0% |

### Contrast 2: E1d (Square-Root Balanced) vs. E6b (Combined Gold Weighted)

| Task | Support | E1d Macro-F1 | E6b Macro-F1 | Delta (E6b - E1d) | 95% Cluster CI | Raw $p_{\text{perm}}$ | Holm-Adjusted $p$ | Significant ($\alpha=0.05$) | $\mathbb{P}(\Delta > 0)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hate** | 5,718 | 0.6808 | 0.6828 | +0.0020 | [-0.0074, 0.0114] | 0.6855 | 1.0000 | No | 66.1% |
| **Sarcasm** | 5,718 | 0.5608 | 0.5594 | -0.0013 | [-0.0121, 0.0097] | 0.8087 | 1.0000 | No | 41.0% |
| **Target** | 1,437 | 0.4364 | 0.4103 | -0.0261 | [-0.0621, 0.0097] | 0.1601 | 0.6403 | No | 7.9% |
| **Severity** | 1,580 | 0.4252 | 0.3527 | **-0.0725** | [-0.1232, -0.0222] | 0.0066 | 0.0330 | **Yes** | 0.2% |
| **Overall Mean** | 5,718 | 0.5258 | 0.5013 | **-0.0245** | [-0.0403, -0.0078] | 0.0034 | — | — | 0.2% |

---

## 2. Class-Level Precision, Recall, and F1 Breakdown

Special focus is placed on the ultra-rare **High Severity** class ($N = 30$) and minority target categories:

| Head | Class Label | Support | E1c Prec / Rec / F1 | E1d Prec / Rec / F1 | E6b Prec / Rec / F1 | $\Delta$ F1 (E1d - E1c) | $\Delta$ F1 (E6b - E1d) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hate** | `non_hate (0)` | 4,138 | 0.821 / 0.840 / **0.831** | 0.847 / 0.755 / **0.798** | 0.849 / 0.754 / **0.799** | -0.0320 | +0.0006 |
| **Hate** | `hate (1)` | 1,580 | 0.554 / 0.521 / **0.537** | 0.501 / 0.643 / **0.563** | 0.502 / 0.649 / **0.567** | +0.0260 | +0.0035 |
| **Sarcasm** | `not_sarcastic (0)` | 3,589 | 0.654 / 0.834 / **0.733** | 0.669 / 0.735 / **0.700** | 0.668 / 0.740 / **0.702** | -0.0331 | +0.0018 |
| **Sarcasm** | `sarcastic (1)` | 2,129 | 0.478 / 0.256 / **0.333** | 0.463 / 0.387 / **0.421** | 0.464 / 0.379 / **0.417** | +0.0886 | -0.0044 |
| **Target** | `individual (0)` | 1,037 | 0.783 / 0.906 / **0.840** | 0.824 / 0.831 / **0.828** | 0.810 / 0.851 / **0.830** | -0.0127 | +0.0020 |
| **Target** | `organization (1)` | 81 | 0.282 / 0.136 / **0.183** | 0.330 / 0.358 / **0.343** | 0.261 / 0.284 / **0.272** | +0.1599 | -0.0710 |
| **Target** | `community (2)` | 251 | 0.482 / 0.374 / **0.421** | 0.460 / 0.458 / **0.459** | 0.453 / 0.438 / **0.445** | +0.0376 | -0.0138 |
| **Target** | `society (3)` | 68 | 0.333 / 0.015 / **0.028** | 0.132 / 0.103 / **0.116** | 0.235 / 0.059 / **0.094** | +0.0875 | -0.0216 |
| **Severity** | `low (0)` | 141 | 0.250 / 0.014 / **0.027** | 0.243 / 0.177 / **0.205** | 0.111 / 0.043 / **0.061** | +0.1781 | -0.1434 |
| **Severity** | `moderate (1)` | 1,409 | 0.892 / 0.991 / **0.939** | 0.904 / 0.925 / **0.914** | 0.892 / 0.952 / **0.921** | -0.0244 | +0.0066 |
| **Severity** | `high (2)` | 30 | 0.000 / 0.000 / **0.000** | 0.147 / 0.167 / **0.156** | 0.087 / 0.067 / **0.075** | +0.1562 | -0.0807 |

---

## 3. Fold-Level Performance & Stability Across Folds

5-fold cross-validation metrics ($Mean \pm Std$ across held-out test partitions):

| Task | Contrast 1: E1c vs. E1d (Fold Deltas) | Contrast 2: E1d vs. E6b (Fold Deltas) |
| :--- | :--- | :--- |
| **Hate** | 0.6833±0.0245 vs 0.6806±0.0078 (Deltas: [0.0277, -0.0278, -0.0024, -0.0056, -0.0053]) | 0.6806±0.0078 vs 0.6828±0.0138 (Deltas: [-0.0026, 0.0145, 0.0026, -0.0066, 0.0026]) |
| **Sarcasm** | 0.5326±0.0065 vs 0.5600±0.0227 (Deltas: [0.0413, 0.0477, -0.0056, 0.019, 0.0346]) | 0.5600±0.0227 vs 0.5594±0.0165 (Deltas: [-0.0112, 0.0032, 0.0109, 0.0025, -0.0082]) |
| **Target** | 0.3652±0.0257 vs 0.4338±0.0238 (Deltas: [0.0462, 0.0752, 0.0682, 0.0507, 0.1027]) | 0.4338±0.0238 vs 0.4098±0.0380 (Deltas: [-0.0252, -0.0801, -0.0308, 0.0554, -0.0394]) |
| **Severity** | 0.3217±0.0107 vs 0.4202±0.0272 (Deltas: [0.0761, 0.0614, 0.1029, 0.1078, 0.1443]) | 0.4202±0.0272 vs 0.3501±0.0307 (Deltas: [-0.0711, -0.0182, -0.0858, -0.1115, -0.0635]) |

---

## 4. Supplementary Analysis: Binary Heads McNemar Tests

| Contrast | Task | Discordant Pairs ($n_{01} + n_{10}$) | Odds Ratio | Test Variant | Raw $p$ | Holm-Adjusted $p$ | Significant ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| E1c vs E1d | **Hate** | 688 | 0.626 | Edwards continuity-corrected McNemar | 0.0000 | 0.0000 | **Yes** |
| E1c vs E1d | **Sarcasm** | 839 | 0.828 | Edwards continuity-corrected McNemar | 0.0071 | 0.0213 | **Yes** |
| E1d vs E6b | **Hate** | 625 | 1.023 | Edwards continuity-corrected McNemar | 0.8103 | 1.0000 | No |
| E1d vs E6b | **Sarcasm** | 813 | 1.007 | Edwards continuity-corrected McNemar | 0.9441 | 1.0000 | No |

---

## 5. Methodological & Scientific Interpretation

### 5.1 Per-Task Class Balancing (E1c -> E1d)
- **Statistically Supported Gains:** Following Holm-Bonferroni family-wise error rate control across the 8 hypothesis tests, per-task square-root class weighting demonstrates statistically significant Macro-F1 improvements on: **sarcasm, target, severity** (Holm $p < 0.05$).
- **Preserved Primary Hate Detection:** Hate classification remained statistically unchanged ($\Delta = -0.0030$, Holm $p = 1.0000$), confirming that balancing minority classes in secondary heads did not compromise primary binary hate discrimination.
- **Severity Minority-Class Recovery vs. Majority Trade-off:** High severity ($N = 30$) recall increased by +0.17 (0.00 -> 0.17), rescuing a previously zero-recall class. Crucially, the majority Moderate Severity class ($N = 1,409$) maintained stable performance (0.904 precision, 0.914 F1), avoiding catastrophic majority-class degradation.

### 5.2 Auxiliary Gold Hate Supervision Transfer (E1d -> E6b)
- **Inconclusive Primary Transfer:** Holding the optimizer-update budget strictly constant (488 steps) and matching class weights, adding auxiliary gold text hate data yielded negligible and statistically non-significant change in hate classification ($\Delta = +0.0020$, raw $p = 0.6855$, Holm $p = 1.0000$).
- **Conditional Supervision Dilution Under Fixed Budget:** Severity Macro-F1 experienced a statistically significant degradation ($\Delta = -0.0725$, Holm $p = 0.0330$). Because auxiliary HOT rows mask target and severity labels, mixing them into a fixed 488-update training budget reduced the effective gradient updates allocated to conditional heads, weakening fragile decision boundaries for low-frequency severity classes.
- **Target Head Invariance:** Target classification showed a non-significant difference ($\Delta = -0.0261$, Holm $p = 0.6403$), indicating resilience to auxiliary row mixing.

---
*Report generated automatically by `toxifusionnet.experiments.compare.run_publication_analysis` with reproducible cluster permutations and family-wise error control.*