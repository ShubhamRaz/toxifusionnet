# ToxiFusionNet v3: Data Integrity & Reproducibility Audit

**Author:** Antigravity AI  
**Repository:** `/workspace/toxifusionnet`  
**Date:** October 2026  
**Status:** All Guarantees Verified & Enforced  

---

## 1. Authoritative Fold Cache Checksum Verification

To prevent regression or accidental split reshuffling across repeated experimental phases, the canonical 5-fold grouped split cache is verified:

- **File Path:** `data/processed/multibully_v2_cv_folds.csv`
- **Total Records:** 28,590 rows ($5 \text{ folds} \times 5,718 \text{ MultiBully samples}$)
- **Canonical SHA-256 Checksum (in-memory CRLF normalized):** `73d4dbb4858907f43765dea0acff36aa4ae21ddc01e1694c0fcf85791fb2d1a5`
- **Linux raw LF Checksum:** `da3ea2286e88cda00896bdc10610ef0ea76cbcde4218a43ffa97aec7237c99d2` (Git checkout on Linux preserves LF line endings; in-memory CRLF normalization produces the canonical hash)
- **Byte Size:** 633,997 bytes (CRLF) / 605,406 bytes (LF)

### 1.1 Out-of-Fold Partition Structure
Every MultiBully sample ($N = 5,718$) appears as an evaluated test sample in **exactly one fold**. Across all 5 folds, the unique evaluated test IDs equal exactly 5,718 ($100.0\%$ coverage):

| Fold | Training Size | Validation Size | Held-Out Test Size | Test MultiBully Coverage |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 4,002 | 574 | **1,142** | 19.97% |
| **Fold 1** | 4,002 | 572 | **1,144** | 20.01% |
| **Fold 2** | 4,003 | 569 | **1,146** | 20.04% |
| **Fold 3** | 4,003 | 571 | **1,144** | 20.01% |
| **Fold 4** | 4,004 | 572 | **1,142** | 19.97% |
| **Total Pooled** | — | — | **5,718** | **100.0%** |

---

## 2. Leakage Prevention Guarantees

### 2.1 Group Leakage Prevention
MultiBully memes frequently share identical OCR text templates paired with differing background imagery. If identical texts span both train and test partitions, the model can memorize the text template, causing inflated test performance.
- **Normalization:** Texts are standardized via `normalize_ocr_text` (lowercased, stripped, whitespace collapsed).
- **Partitioning:** Grouped splitting (`assign_group_ids` + `make_grouped_cv_splits`) guarantees that all samples sharing the same normalized OCR text belong to the exact same partition.
- **Verification:**
  $$\text{Group}(\text{Train}_f) \cap \text{Group}(\text{Test}_f) = \emptyset, \quad \forall f \in \{0, 1, 2, 3, 4\}$$
  Confirmed: Zero overlapping groups across all folds.

### 2.2 Cross-Dataset Leakage Prevention (MultiBully vs. HOT)
An exhaustive $100\%$ pairwise comparison of all 5,718 MultiBully samples against all 3,189 HOT gold samples ($18,234,702$ comparisons) confirms:
- **Exact Matches:** 0 matches.
- **Near-Duplicate Matches (Jaccard similarity $\ge 0.85$):** 0 matches.
- **Result:** No cross-dataset contamination exists between MultiBully test samples and HOT auxiliary training samples.

---

## 3. Auxiliary Data Routing & Label Masking

### 3.1 Strict Training Split Routing
In multi-dataset training (E6, E6a, E6b, E7, E7a):
- Auxiliary HOT samples ($N = 3,189$) are injected **strictly into the training partition** of each fold (`split == "train"`).
- Validation partitions contain **0 HOT samples** ($100\%$ MultiBully).
- Test partitions contain **0 HOT samples** ($100\%$ MultiBully).
- All out-of-fold benchmark metrics and confusion matrices evaluate strictly against the identical 5,718 MultiBully ground-truth annotations.

### 3.2 Label Masking with `IGNORE_INDEX` (-100)
To prevent ungrounded supervision from degrading model heads:
1. **HOT Auxiliary Samples:**
   - `hate`: Ground-truth human gold binary labels (`non_hate` $\rightarrow 0$, `hate` $\rightarrow 1$).
   - `hate` for offensive samples (`orig_hot_label == 2.0`): Masked to `IGNORE_INDEX` (-100).
   - `sarcasm`: Masked to `IGNORE_INDEX` (-100).
   - `target`: Masked to `IGNORE_INDEX` (-100).
   - `severity`: Masked to `IGNORE_INDEX` (-100).
2. **MultiBully Conditional Samples:**
   - For all non-hate MultiBully posts (`hate == "non_hate"`), `target` and `severity` are set to `NaN` and mapped to `IGNORE_INDEX` (-100).
3. **Loss Invariance:**
   - In `MultiTaskLoss`, all samples with $y = \text{IGNORE\_INDEX}$ are excluded from loss calculations before computing cross-entropy:
     $$\mathcal{L}_t = \text{CrossEntropy}(\text{logits}[y \neq -100], \text{targets}[y \neq -100], \text{weight}=w_t)$$
   - Masked samples receive identically zero gradient contribution ($\nabla_{\theta} \mathcal{L}_t = 0$).

---

### 3.3 Zero-Supervision Auxiliary Row Exclusion from Training Loader
In `hot_gold_only.csv`, 1,765 rows correspond to offensive language where `orig_hot_label == 2.0` (hate is masked to NaN) and sarcasm, target, and severity are unannotated (NaN).
- **Audit:**
  - Raw on-disk rows in `data/processed/hot_gold_only.csv`: **3,189** (preserved untouched).
  - Rows with zero supervision across all 4 tasks: **1,765** ($55.35\%$).
  - Rows with active human gold hate supervision: **1,424** ($44.65\%$, 1,121 `non_hate`, 303 `hate`).
- **Loader Filter Implementation:**
  - Both `cv.py` and `trainer.py` filter out any auxiliary row where all configured task labels map to `IGNORE_INDEX` (-100).
  - The 1,765 zero-supervision rows are completely excluded from entering the training DataLoader, preventing wasteful forward passes that produce 0 loss.
  - All 1,424 valid human gold hate annotations are retained and injected into the training partition.
  - Excluded-row counts are logged at `INFO` level during CV split assembly and trainer initialization.

---

## 4. Class Weight Derivation Equivalence (E1d vs. E6b)

To ensure that differences between E1d (MultiBully-only) and E6b (MultiBully + HOT) reflect only the transfer effect of auxiliary hate supervision, class weights are derived strictly from MultiBully training samples:
- `class_weight_source: multibully` filters training samples to `source_dataset == "multibully"` during weight calculation.
- As verified in [class_weights_by_fold.json](file:///c:/Users/SHUBHAM%20RAJ/Desktop/Others/Toxic/toxifusionnet/toxifusionnet/results/v3_balancing_audit/class_weights_by_fold.json), the fold-for-fold class weights for E1d and E6b are **bit-for-bit identical**.

---

## 5. Non-Overwriting Canonical Fold Cache Policy

- The canonical fold cache `data/processed/multibully_v2_cv_folds.csv` is protected against automatic modification or regeneration if it already exists on disk.
- When `folds_cache` is specified:
  1. The SHA-256 checksum is computed with in-memory CRLF normalization and logged (`73d4dbb4858907f43765dea0acff36aa4ae21ddc01e1694c0fcf85791fb2d1a5`).
  2. The sample IDs and fold distributions are validated against the active dataset.
  3. If validation passes, the cached assignments are read in read-only mode.
  4. If validation fails or a checksum mismatch occurs on DGX, execution fails fast with an explanatory error instead of silently overwriting the canonical cache.
