# ToxiFusionNet: MultiBully & HOT Data-Integrity Audit Report

**Audit Status:** `PASS`  
**Evaluation Date:** `2026-10-15 05:08:02 UTC`  
**Repository:** `https://github.com/ShubhamRaz/toxifusionnet`  

---

## Executive Summary & Verdict

| Section | Audit Domain | Status | Key Finding |
| :--- | :--- | :---: | :--- |
| **A** | **Data Provenance** | `PASS` | E1d strictly uses MultiBully; E6b strictly uses MultiBully + `hot_gold_only.csv`. Zero pseudo-labels reach E6b. |
| **B** | **HOT Label Mapping** | `PASS` | Normal (1,121) -> non_hate; Hate (303) -> hate; Offensive (1,765) -> masked to NaN; needs_review (732) -> ignored. |
| **C** | **Task Supervision** | `PASS` | Zero-supervision auxiliary rows (1,765) filtered dynamically; no rows dropped for conditional missing labels. |
| **D** | **Cross-Dataset Leakage** | `PASS` | Exact matches = 0, Duplicate IDs = 0, Max Char-TFIDF Cosine = 0.7703 (0 pairs >= 0.85 across 18.2M pairs). |
| **E** | **Label Definitions** | `PASS` | MultiBully Text-Label strictly preserved for text models; 1,559 meme-only bully instances prevented from contaminating text heads. |

**Overall Audit Verdict: PASS.**  
All canonical splits, human annotations, fold caches, and loss masks satisfy strict scientific and methodological criteria. No model retraining or dataset modification is required.

---

## A. Exact Data Provenance & File Audit

### 1. Actual Experiment Inputs
- **Experiment 1d (`exp1d_multibully_text_weighted`):**
  - Config: `configs\exp1d_multibully_text_weighted.yaml`
  - Primary Data: `data/processed/multibully.csv` ($N = 5,718$)
  - Auxiliary Data: `None` (Zero auxiliary injection)
  - Folds Cache: `data/processed/multibully_v2_cv_folds.csv`
- **Experiment 6b (`exp6b_combined_text_gold_weighted`):**
  - Config: `configs\exp6b_combined_text_gold_weighted.yaml`
  - Primary Data: `data/processed/multibully.csv` ($N = 5,718$)
  - Auxiliary Data: `data/processed/hot_gold_only.csv` ($N = 3,189$ raw; 1,424 retained in training loader; 1,765 zero-supervision excluded)
  - Folds Cache: `data/processed/multibully_v2_cv_folds.csv`

### 2. Distinction Between Processed HOT Datasets

| Artifact | Rows | SHA-256 (Raw) | Hate Provenance | Sarcasm / Target / Severity | Role / Experiment |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `hot_cleaned.csv` | 3,189 | `6b9489f714241e9d...` | Qwen pseudo-labels (1,004 hate, 2,185 non-hate) | Qwen pseudo-labels (N=2,457 valid) | Historical baseline (Contaminated; NOT in E1d/E6b) |
| `hot_v2.csv` | 3,189 | `db5652f6a09aaa82...` | Human gold (303 hate, 1,121 non-hate, 1,765 NaN) | Qwen pseudo-labels (N=2,457 valid) | Historical reference (Contaminated; NOT in E1d/E6b) |
| `hot_gold_only.csv` | 3,189 | `a655e0204afa53d5...` | Human gold (303 hate, 1,121 non-hate, 1,765 NaN) | **100% Masked to NaN (Zero pseudo-labels)** | **Actual E6b Auxiliary Input** (`exp6b_combined_text_gold_weighted`) |
| `combined_multibully_hot_v2.csv` | 8,907 | `c3496e411dbced3c...` | Human gold (1,883 hate, 5,259 non-hate) | Contains Qwen pseudo-labels from hot_v2 | Combined reference file (NOT used by E1d/E6b) |

### 3. Auxiliary Pseudo-Label Verification in E6b
- **Sarcasm Pseudo-labels in E6b:** `0` (100% masked to `NaN`).
- **Target Pseudo-labels in E6b:** `0` (100% masked to `NaN`).
- **Severity Pseudo-labels in E6b:** `0` (100% masked to `NaN`).
- **Rationales in E6b:** `0` (No pseudo-label attention supervision).
- **Conclusion:** E6b receives **strictly human ground truth** for hate detection and zero synthetic/pseudo supervision on any head.

---

## B. Audit of HOT Labels Across Processing Stages

| Original Category | Original Code | Count | `hot_cleaned.csv` Mapping | `hot_gold_only.csv` Mapping | Canonical Model ID | Status / Policy |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **Normal** | `0.0` | 1,121 | non_hate: 1095, hate: 26 | **non_hate: 1121** | `0` | `PASS` |
| **Hate Speech** | `1.0` | 303 | hate: 169, non_hate: 134 | **hate: 303** | `1` | `PASS` |
| **Offensive Language** | `2.0` | 1,765 | non_hate: 956, hate: 809 | **nan: 1765** | `-100` | `PASS` |

**Verification of Rule Compliance:**
1. **Normal -> non_hate:** Verified. Exactly 1,121 of 1,121 instances map to `non_hate` (Canonical ID 0).
2. **Hate Speech -> hate:** Verified. Exactly 303 of 303 instances map to `hate` (Canonical ID 1).
3. **Offensive -> ignored for binary hate supervision:** Verified. Exactly 1,765 of 1,765 instances map to `NaN` (Canonical ID -100). They are excluded from binary cross-entropy loss computation.
4. **needs_review & missing pseudo-labels -> ignored:** Verified. Exactly 732 instances with `uncertain` status in raw annotations are masked to `NaN`, never converted into negative labels.
5. **Generic `build_canonical()` audit:** Fixed in `schema.py` to raise `ValueError` on unknown classes rather than silently mapping to hate, and updated row filtering to retain rows with supervision on any task.

---

## C. Task-Wise Supervision & Loss Consumption

| Source Dataset | Task | Total Rows | Valid Gold | Valid Pseudo | Missing / Masked | Loss Consumption (Rows) | Loss Policy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| MultiBully (Processed) | **Hate** | 5,718 | 5,718 | 0 | 0 | **5,718** | Primary evaluation benchmark; 100% human gold |
| MultiBully (Processed) | **Sarcasm** | 5,718 | 5,718 | 0 | 0 | **5,718** | Primary evaluation benchmark; 100% human gold |
| MultiBully (Processed) | **Target** | 5,718 | 1,437 | 0 | 4,281 | **1,437** | Primary evaluation benchmark; 100% human gold |
| MultiBully (Processed) | **Severity** | 5,718 | 1,580 | 0 | 4,138 | **1,580** | Primary evaluation benchmark; 100% human gold |
| HOT Gold-Only (E6b) | **Hate** | 3,189 | 1,424 | 0 | 1,765 | **1,424** | Strict human gold; all auxiliary tasks 100% masked to NaN |
| HOT Gold-Only (E6b) | **Sarcasm** | 3,189 | 0 | 0 | 3,189 | **0** | Strict human gold; all auxiliary tasks 100% masked to NaN |
| HOT Gold-Only (E6b) | **Target** | 3,189 | 0 | 0 | 3,189 | **0** | Strict human gold; all auxiliary tasks 100% masked to NaN |
| HOT Gold-Only (E6b) | **Severity** | 3,189 | 0 | 0 | 3,189 | **0** | Strict human gold; all auxiliary tasks 100% masked to NaN |
| HOT v2 (Reference) | **Hate** | 3,189 | 1,424 | 0 | 1,765 | **1,424** | Contains Qwen pseudo-labels for sarcasm, target, severity |
| HOT v2 (Reference) | **Sarcasm** | 3,189 | 0 | 2,457 | 732 | **2,457** | Contains Qwen pseudo-labels for sarcasm, target, severity |
| HOT v2 (Reference) | **Target** | 3,189 | 0 | 1,035 | 2,154 | **1,035** | Contains Qwen pseudo-labels for sarcasm, target, severity |
| HOT v2 (Reference) | **Severity** | 3,189 | 0 | 1,020 | 2,169 | **1,020** | Contains Qwen pseudo-labels for sarcasm, target, severity |

**Zero-Supervision Row Handling:**
- In `hot_gold_only.csv`, 1,765 rows correspond to offensive tweets where hate is masked to `NaN` and all other heads are `NaN`.
- In `src/toxifusionnet/experiments/cv.py`, lines 123-137 automatically exclude these 1,765 zero-supervision rows upon loading.
- Exactly 1,424 valid human-gold hate rows are retained in the training pool.
- On-disk files remain strictly untouched ($N = 3,189$).

---

## D. Cross-Dataset Leakage & Overlap Audit

A full-corpus pairwise leakage audit was executed comparing all MultiBully test/validation examples against all HOT training examples:

| Leakage Measure | Scope | Threshold | Observed Matches | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Exact Normalized OCR Text** | 100% Full Corpus | Exact string equality | **0** | `PASS` |
| **Full Normalizer Pipeline Text** | 100% Full Corpus | Exact string equality | **0** | `PASS` |
| **Duplicate ID Collision** | 100% Full Corpus | ID equality | **0** | `PASS` |
| **Conflicting Hate Labels** | Matching text | Label discordance | **0** | `PASS` |
| **Char N-Gram TF-IDF Overlap** | 18,234,702 pairs | Cosine $\ge 0.85$ | **0** (Max = 0.7703) | `PASS` |
| **Word-Set Jaccard Overlap** | 18,234,702 pairs | Jaccard $\ge 0.80$ | **0** | `PASS` |

**Per-Fold Validation & Test Leakage:**
- **Fold_0:** Validation Overlap = `0`, Test Overlap = `0`.
- **Fold_1:** Validation Overlap = `0`, Test Overlap = `0`.
- **Fold_2:** Validation Overlap = `0`, Test Overlap = `0`.
- **Fold_3:** Validation Overlap = `0`, Test Overlap = `0`.
- **Fold_4:** Validation Overlap = `0`, Test Overlap = `0`.

---

## E. Label-Definition Consistency & Confounding Analysis

### 1. MultiBully Text-Label vs. Img-Text-Label Consistency
- **Text-Label (Text Alone):** Nonbully = 4,138 (72.37%), Bully = 1,580 (27.63%).
- **Img-Text-Label (Whole Meme):** Nonbully = 2,579 (45.10%), Bully = 3,139 (54.90%).
- **Meme-Image Divergence:** In **1,725 memes (29.41%)**, the text alone is harmless/non-bullying, but the combined image+text meme is bullying.
- **Methodological Rule:** Text-only models (E1c, E1d, E6b) must strictly evaluate against `Text-Label`. Conflating `Img-Text-Label` into a text-only model would penalize the model for visual bullying that text cannot observe.

### 2. Source-Label Confounding (If Pseudo-Labels Were Injected)

| Task Head | Class | MultiBully Gold ($N = 5,718$) | HOT Qwen Pseudo ($N = 3,189$) | Confounding Mechanism / Risk |
| :--- | :--- | :---: | :---: | :--- |
| **Severity** | **High** | **30 (1.9%)** | **713 (69.9%)** | Severe positive skew. Model associates Twitter text with high severity, exploding false positives on memes. |
| **Severity** | **Low** | **141 (8.9%)** | **0 (0.0%)** | Complete absence of low severity in pseudo-labels destroys low-severity sensitivity. |
| **Target** | **Organization** | **81 (5.6%)** | **6 (0.58%)** | Severe dilution of rare organizational targets. |
| **Target** | **Society** | **68 (4.7%)** | **7 (0.68%)** | Severe dilution of rare societal targets. |

- **Audit Verdict:** The severe distribution shift confirms that masking HOT auxiliary heads in `hot_gold_only.csv` was scientifically mandatory and averted catastrophic source-label confounding.

---

*Report generated automatically by `scripts/run_data_integrity_audit.py`.*