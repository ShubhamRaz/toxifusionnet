# Dataset Integrity & Data Leakage Audit Report

---

## 1. Corpus Scale & Component Verification

The canonical unified dataset file [`data/processed/combined_all_3_datasets.csv`](file:///c:/Users/SHUBHAM%20RAJ/Desktop/Others/Toxic/toxifusionnet/toxifusionnet/data/processed/combined_all_3_datasets.csv) was programmatically audited row-by-row.

### Measured Size & Composition
- **Total Physical Rows:** Exactly **56,907** rows.
- **Unique Texts:** **56,799** unique texts ($108$ duplicate texts, $0.19\%$ redundancy).
- **File Size on Disk:** $15,601,048$ bytes ($15.6$ MB).
- **Metadata Configuration:** [`data/processed/combined_all_3_datasets.meta.json`](file:///c:/Users/SHUBHAM%20RAJ/Desktop/Others/Toxic/toxifusionnet/toxifusionnet/data/processed/combined_all_3_datasets.meta.json).

```
Total Corpus: 56,907 Rows
├── Hinglish Toxic Hate Dataset (Kaggle) : 48,000 rows (84.35%)
├── MultiBully Meme Text OCR (SIGIR '22) :  5,718 rows (10.05%)
└── HOT Twitter/X Posts (HASOC / FIRE)   :  3,189 rows ( 5.60%)
```

---

## 2. Partition Audit & Disjoint Split Verification

### Partition Cross-Tabulation ($N = 56,907$)

| Source Dataset | Train Split | Validation Split | Test Split | Total Rows | Source Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hinglish Balanced (Kaggle)** | 33,600 | 7,200 | 7,200 | **48,000** | $84.35\%$ |
| **MultiBully Gold (Meme OCR)** | 4,002 | 858 | 858 | **5,718** | $10.05\%$ |
| **HOT Tweet Corpus (Twitter/X)**| 2,277 | 457 | 455 | **3,189** | $5.60\%$ |
| **Total Unified Corpus** | **39,879** | **8,515** | **8,513** | **56,907** | **100.0%** |
| **Split Fraction** | **$70.08\%$** | **$14.96\%$** | **$14.96\%$** | **$100.0\%$** | — |

---

## 3. Data Leakage & Cross-Split Overlap Analysis

We tested for three forms of train/test leakage:

### 3.1 Exact String Matching
- **Train vs. Test Exact Duplicates:** **`0`** exact matches.
- **Train vs. Validation Exact Duplicates:** **`0`** exact matches.
- **Validation vs. Test Exact Duplicates:** **`0`** exact matches.
- **Finding:** There is zero verbatim text leakage between splits.

### 3.2 Normalized String Matching (Lowercased, Alphanumeric Only, Whitespace Canonicalized)
- **Train vs. Test Normalized Overlap:** **`27`** instances ($0.31\%$ of test set).
- **Train vs. Validation Normalized Overlap:** **`30`** instances ($0.35\%$ of val set).
- **Validation vs. Test Normalized Overlap:** **`5`** instances ($0.06\%$).
- **Finding:** Minimal punctuation-variation leakage ($< 0.35\%$), well within acceptable NLP benchmark thresholds.

### 3.3 Synthetic Template Overlap (The Root of Score Inflation)
- **Investigation:** The 48,000-row Hinglish Balanced dataset is built on slot-filling templates using a vocabulary of approximately 840 words.
- **Finding:** While exact strings do not repeat across splits, **the identical sentence syntactic skeletons and structural templates appear in both the training set and the test set**.
- **Impact:** The XLM-RoBERTa model memorizes the template-to-label mapping, leading to a literal **$100.00\%$ accuracy on the synthetic test partition**.

---

## 4. Label Distribution & Missing Label Audit

### Task Supervision Across the Full Corpus ($N = 56,907$)

| Task Column | Supervised (Non-Null) | Masked / Null | Percentage Supervised | Masking Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **`hate`** | 56,907 | 0 | **100.0%** | Fully supervised across all three datasets |
| **`sarcasm`** | 56,175 | 732 | **98.71%** | 732 unannotated rows in HOT dataset ($118$ in test split) |
| **`target`** | 26,472 | 30,435 | **46.52%** | Masked with `IGNORE_INDEX = -100` on non-hate instances |
| **`severity`** | 26,600 | 30,307 | **46.74%** | Masked with `IGNORE_INDEX = -100` on non-hate instances |

### Resolution of the "8,395 Sarcasm Support" Discrepancy
The test set contains 8,513 posts, but Sarcasm evaluation evaluates only 8,395 posts:
- **Exact Cause:** 118 posts in the HOT test split have `sarcasm = NaN` (stored as `-100` in predictions).
- $8,513 - 118 = \mathbf{8,395}$.
- **Finding:** This is an intentional and scientifically correct missing-label mask, not a data loss error.

---

## 5. Label Provenance & Ground Truth Authenticity

| Dataset Source | Rows | Hate Annotations | Sarcasm Annotations | Target & Severity Annotations | Provenance Rating |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MultiBully** | 5,718 | Human Gold | Human Gold | Human Gold | **Tier 1 (High)** |
| **HOT Tweets** | 3,189 | Human Gold | Human / Qwen | Qwen LLM Pseudo-Labels | **Tier 2 (Mixed)** |
| **Hinglish Balanced** | 48,000 | Synthetic Rule | Synthetic Rule | Synthetic Rule | **Tier 3 (Synthetic)** |

- **Academic Integrity Warning:** Reports must not state or imply that the entire 56,907 dataset consists of "gold human annotations." Only the MultiBully portion ($5,718$) and the primary hate labels of HOT ($3,189$) are human ground truth.
