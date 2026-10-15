# ToxiFusionNet: Multi-Task Diagnostic Analysis Report

This report establishes the baseline diagnostic profile for **E1c** (unweighted control), **E1d** (per-task square-root class weighted), **E6b** (combined gold auxiliary weighted), and the classical **TF-IDF + Logistic Regression** baseline across all four tasks on MultiBully ($N = 5,718$ held-out test predictions).

---

## 1. Overall Multi-Task Performance Matrix

| Task | Support | TF-IDF Baseline F1 / Acc | E1c Unweighted F1 / Acc | E1d Weighted F1 / Acc | E6b Combined F1 / Acc | E1d vs. TF-IDF ($\Delta$ F1) | E1d vs. E1c ($\Delta$ F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hate** | 5,718 | 0.6759 / 0.7419 | 0.6838 / 0.7518 | **0.6808** / 0.7242 | 0.6828 / 0.7254 | +0.0049 | -0.0030 |
| **Sarcasm** | 5,718 | 0.5629 / 0.5946 | 0.5330 / 0.6187 | **0.5608** / 0.6049 | 0.5594 / 0.6055 | -0.0021 | +0.0278 |
| **Target** | 1,437 | 0.3668 / 0.7383 | 0.3684 / 0.7279 | **0.4364** / 0.7049 | 0.4103 / 0.7091 | +0.0696 | +0.0680 |
| **Severity** | 1,580 | 0.3366 / 0.8823 | 0.3219 / 0.8848 | **0.4252** / 0.8443 | 0.3527 / 0.8538 | +0.0886 | +0.1033 |
| **Overall Mean** | 5,718 | 0.4855 / — | 0.4768 / — | **0.5258** / — | 0.5013 / — | +0.0403 | +0.0490 |

---

## 2. Balanced Accuracy & Macro-Recall Summary

Balanced accuracy corresponds to unweighted macro-averaged recall across classes ($BA = \frac{1}{K}\sum_k Recall_k$):

| Task | TF-IDF Balanced Acc | E1c Balanced Acc | E1d Balanced Acc | E6b Balanced Acc | Gain (E1d - E1c) | Gain (E1d - TF-IDF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hate** | 0.6751 | 0.6805 | **0.6991** | 0.7019 | +0.0186 | +0.0240 |
| **Sarcasm** | 0.5625 | 0.5449 | **0.5605** | 0.5594 | +0.0156 | -0.0020 |
| **Target** | 0.3496 | 0.3579 | **0.4376** | 0.4079 | +0.0797 | +0.0880 |
| **Severity** | 0.3425 | 0.3350 | **0.4232** | 0.3537 | +0.0882 | +0.0807 |

---

## 3. Binary Prediction Score Distributions (Hate & Sarcasm)

### 3.1 Probability Distribution Statistics (E1d Reference)

| Task | Overall Mean +/- Std | Positive Class Median [IQR] | Negative Class Median [IQR] | Mean Separation ($\mu_+ - \mu_-$) | AUROC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hate** | 0.367 +/- 0.276 | 0.615 [0.401, 0.749] | 0.219 [0.063, 0.494] | +0.273 | 0.7825 | 0.1754 |
| **Sarcasm** | 0.417 +/- 0.148 | 0.461 [0.358, 0.550] | 0.398 [0.287, 0.507] | +0.056 | 0.6071 | 0.2316 |

**Key Distribution Findings:**
- **Sarcasm Score Distribution Shift:** For sarcasm in E1d, the positive class median score is **0.461**, with 25% of positives scoring below 0.358. Because the default decision threshold is 0.50, 1,306 of 2,129 true sarcastic examples (61.34%, fraction = 0.6134) have $p_{\text{sarcasm}} < 0.50$ and are misclassified as non-sarcastic by the default 0.50 decision boundary. In the borderline range `[0.36, 0.50)`, the model assigns probabilities to 765 of 2,129 true sarcastic examples (35.93%, fraction = 0.3593).
- **Hate Score Calibration:** For hate in E1d, positive class median is **0.615** and negative class median is **0.219**, giving healthy separation ($\Delta \mu = +0.273$).

---

## 4. Severity & Target Error Patterns: Focus on Rare Classes

### 4.1 Rare High-Severity Class Diagnostics ($N = 30$)

| Metric | E1c (Unweighted) | E1d (Class Weighted) | E6b (Combined Gold) |
| :--- | :---: | :---: | :---: |
| **Support (Held-Out Test)** | 30 | 30 | 30 |
| **Correctly Identified (TP)** | 0 | **5** | 2 |
| **High Severity Recall** | 0.0000 | **0.1667** | 0.0667 |
| **Misclassified as Moderate (FN)** | 30 | 24 | 28 |
| **Misclassified as Low (FN)** | 0 | 1 | 0 |
| **Total False Positives** | 7 | 29 | 21 |
| **High Severity Precision** | 0.0000 | **0.1471** | 0.0870 |

### 4.2 Target Minority Absorption Patterns (E1d Reference)

| Minority Category | Support | Correct (TP) | Recall | Misclassified as Majority 'Individual' | % Absorbed by Majority |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Organization** | 81 | 29 | 0.3580 | 38 | 46.9% |
| **Community** | 251 | 115 | 0.4582 | 105 | 41.8% |
| **Society** | 68 | 7 | 0.1029 | 41 | 60.3% |

---

## 5. Label Masking & Conditional Cascading Diagnostics

### 5.1 Supervision Coverage & Masking Checks
- **Hate & Sarcasm:** 100% supervision coverage ($N = 5,718$ valid instances; 0 masked).
- **Target:** Supervised on hate posts ($N = 1,437$ instances; 4,281 masked with `-100`).
- **Severity:** Supervised on hate posts ($N = 1,580$ instances; 4,138 masked with `-100`).

### 5.2 End-to-End Conditional Metrics (Cascading Hate Errors)
When evaluating the full pipeline (non-hate -> 'none'), false negatives in binary hate detection cascade into target and severity errors:

| Experiment | Primary Target F1 | End-to-End Target F1 | Primary Severity F1 | End-to-End Severity F1 |
| :--- | :---: | :---: | :---: | :---: |
| **E1c Unweighted** | 0.3684 | 0.3340 | 0.3219 | 0.3355 |
| **E1d Weighted** | 0.4364 | 0.3544 | 0.4252 | 0.3934 |
| **E6b Combined Gold** | 0.4103 | 0.3414 | 0.3527 | 0.3544 |

---
*Report generated automatically by `toxifusionnet.analysis.diagnostic.run_full_diagnostic_analysis`.*