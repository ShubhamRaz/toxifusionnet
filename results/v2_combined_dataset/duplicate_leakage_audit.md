# ToxiFusionNet E6/E7: Cross-Dataset Duplicate & Evaluation Leakage Audit

**Document:** `results/v2_combined_dataset/duplicate_leakage_audit.md`  
**Date:** October 13, 2026  
**Status:** PASSED (ZERO LEAKAGE DETECTED)  
**Primary Dataset:** MultiBully ($N=5,718$, 5,633 unique normalized OCR text groups)  
**Auxiliary Dataset:** HOT ($N=3,189$, 3,080 unique normalized tweet text groups)  
**Canonical Folds:** `data/processed/multibully_v2_cv_folds.csv` (28,590 rows)  

---

## 1. Executive Summary

This audit verifies that the incorporation of the HOT dataset into training does not cause data leakage across cross-validation boundaries.

### Audit Verdict: **PASSED (100% ISOLATION VERIFIED)**
1. **Zero Text Overlap**: Exact and normalized text analysis reveals **0 overlapping texts** between MultiBully memes and HOT tweets ($0 / 5,633$ MultiBully groups, $0 / 3,080$ HOT groups).
2. **Strict Evaluation Isolation**:
   - In all 5 folds, HOT samples are injected **exclusively into the `train` partition**.
   - The validation partition in every fold contains **0 HOT samples** ($N_{\text{val}} \approx 688$, 100% MultiBully).
   - The held-out test partition in every fold contains **0 HOT samples** ($N_{\text{test}} \approx 1,144$, 100% MultiBully).
3. **Exact MultiBully Out-of-Fold Coverage**:
   - Across the 5 folds, exactly **5,718 out-of-fold predictions** are evaluated.
   - Every single MultiBully meme is evaluated exactly once in test.
   - No MultiBully evaluation sample is ever present in the training set of that fold.

---

## 2. Text Normalization & Duplicate Audit Methodology

Text strings from both corpora were normalized using `normalize_ocr_text`:
$$\text{norm}(t) = \text{strip}(\text{regex\_replace}(\text{lower}(t), \text{"\s+"}, \text{" "}))$$

### Empirical Group Statistics:
| Metric | MultiBully | HOT | Combined Pool |
|---|:---:|:---:|:---:|
| **Raw Samples ($N$)** | 5,718 | 3,189 | 8,907 |
| **Unique Normalized Groups** | 5,633 | 3,080 | 8,713 |
| **Internal Duplicates** | 85 rows (grouped together) | 109 rows | — |
| **Cross-Corpus Overlapping Groups** | — | — | **0 (Zero Overlap)** |

Because the corpora originate from distinct web modalities (MultiBully: Facebook/Instagram meme OCR; HOT: Twitter microblog text), their vocabularies and sentence structures have zero identical text matches.

---

## 3. Fold-by-Fold Partition Breakdown

Under the canonical grouped cross-validation cache, partition sizes for Experiments E6 and E7 are:

| Fold Index | MultiBully Train ($N$) | HOT Train ($N$) | Total Inner Train ($N$) | Inner Val ($N$, MultiBully Only) | Held-Out Test ($N$, MultiBully Only) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | 3,889 | 3,189 | **7,078** | **687 (100% MultiBully)** | **1,142 (100% MultiBully)** |
| **Fold 1** | 3,885 | 3,189 | **7,074** | **689 (100% MultiBully)** | **1,144 (100% MultiBully)** |
| **Fold 2** | 3,888 | 3,189 | **7,077** | **684 (100% MultiBully)** | **1,146 (100% MultiBully)** |
| **Fold 3** | 3,881 | 3,189 | **7,070** | **693 (100% MultiBully)** | **1,144 (100% MultiBully)** |
| **Fold 4** | 3,887 | 3,189 | **7,076** | **689 (100% MultiBully)** | **1,142 (100% MultiBully)** |
| **Pooled Test** | — | — | — | — | **5,718 (100% MultiBully)** |

### Guarantees:
- Model checkpoint selection and early stopping depend strictly on MultiBully validation performance.
- Out-of-fold test evaluation assesses generalization strictly on unseen MultiBully memes.
- Results are paired sample-by-sample against E1, E3, and E5 baselines.
