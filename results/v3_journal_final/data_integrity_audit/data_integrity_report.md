# ToxiFusionNet: MultiBully & HOT Data-Integrity Audit Report

**Audit Verdict:** `WARNING`  
**Evaluation Date:** `2026-10-15 08:44:01 UTC`  
**Repository:** `https://github.com/ShubhamRaz/toxifusionnet`  

---

## Executive Summary & Data-Driven Verdict

| Section | Audit Domain | Status | Key Evaluated Finding |
| :--- | :--- | :---: | :--- |
| **A** | **Data Provenance** | `WARNING` | Upstream raw HOT dataset ('dataset.csv') is absent from workspace; local curated files (hot_cleaned.csv, hot_v2.csv, hot_gold_only.csv) verified with zero pseudo-labels, but upstream pre-curation provenance cannot be independently completed on disk. |
| **B** | **HOT Label Mapping** | `PASS` | Observed distributions verified: Normal (1,121)->all 'non_hate'; Hate Speech (303)->all 'hate'; Offensive (1,765)->all masked (NaN). |
| **C** | **Task Supervision & Folds** | `PASS` | 1,765 zero-supervision rows excluded; 1,424 supervised rows retained in train; 5 folds verify pairwise ID disjointness (0 train/val/test overlaps) and partition 5,718 unique test IDs. |
| **D** | **Cross-Dataset Leakage** | `PASS` | 0 exact matches, 0 ID collisions, 0 near-duplicates (char >=0.85 or jaccard >=0.80), 0 fold test/val overlaps. |
| **E** | **Label Definitions & Divergence** | `PASS` | 1-to-1 ID merge verified (5,718 rows); discordance (1,670 vs 111, delta=1,559) reconciled; confounding averted. |

**Overall Audit Verdict: `WARNING`**  
All critical experiment inputs, loss masks, fold caches, and split disjointness are verified, but upstream raw HOT source provenance cannot be completed on local disk because 'dataset.csv' is absent from the repository workspace. No model retraining or dataset modification is required.

---

## A. Exact Data Provenance & Per-Fold Sample Counts

> [!WARNING]
> **Upstream Provenance Status: WARNING.** Upstream raw HOT dataset ('dataset.csv') is absent from workspace; local curated files (hot_cleaned.csv, hot_v2.csv, hot_gold_only.csv) verified with zero pseudo-labels, but upstream pre-curation provenance cannot be independently completed on disk.

### 1. Actual Experiment Inputs & Split Counts
- **Experiment 1d (`exp1d_multibully_text_weighted`):**
  - Config: `configs\exp1d_multibully_text_weighted.yaml`
  - Primary Data: `data/processed/multibully.csv` ($N = 5,718$ unique rows)
  - Auxiliary Data: `None` (Zero auxiliary injection)
  - Folds Cache: `data/processed/multibully_v2_cv_folds.csv` (Canonical 5-fold CV)
  - Outer Test Coverage: `5718` rows (100% of MultiBully)
- **Experiment 6b (`exp6b_combined_text_gold_weighted`):**
  - Config: `configs\exp6b_combined_text_gold_weighted.yaml`
  - Primary Data: `data/processed/multibully.csv` ($N = 5,718$ unique rows)
  - Auxiliary Data: `data/processed/hot_gold_only.csv` ($N = 3,189$ raw; 1,424 retained in training loader; 1,765 zero-supervision excluded)
  - Folds Cache: `data/processed/multibully_v2_cv_folds.csv`
  - Outer Test Coverage: `5718` rows (100% of MultiBully, strictly zero auxiliary data)

### 2. Per-Fold Sample Distribution (Computed from Canonical Cache)

| Fold | MultiBully Train | MultiBully Val | MultiBully Test | Total MB Rows | E1d Actual Train | Retained HOT Aux (Train Only) | E6b Actual Train |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 3,889 | 687 | 1,142 | 5,718 | **3,889** | 1,424 | **5,313** |
| **Fold 1** | 3,885 | 689 | 1,144 | 5,718 | **3,885** | 1,424 | **5,309** |
| **Fold 2** | 3,888 | 684 | 1,146 | 5,718 | **3,888** | 1,424 | **5,312** |
| **Fold 3** | 3,881 | 693 | 1,144 | 5,718 | **3,881** | 1,424 | **5,305** |
| **Fold 4** | 3,887 | 689 | 1,142 | 5,718 | **3,887** | 1,424 | **5,311** |

> [!IMPORTANT]
> **Partition Separation Rule:** In every fold, MultiBully validation sets (684 to 693 rows) are isolated strictly for model selection and checkpointing. They are **never** included in training. In E6b, all 1,424 retained HOT auxiliary rows are assigned strictly to training only.

### 3. Distinction Between Processed HOT Datasets

| Artifact | Rows | SHA-256 (Raw) | Hate Provenance | Sarcasm / Target / Severity | Role / Experiment |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `hot_cleaned.csv` | 3,189 | `6b9489f714241e9d...` | Qwen pseudo-labels (1,004 hate, 2,185 non-hate) | Qwen pseudo-labels (N=2,457 valid) | Historical baseline (Contaminated; NOT in E1d/E6b) |
| `hot_v2.csv` | 3,189 | `db5652f6a09aaa82...` | Human gold (303 hate, 1,121 non-hate, 1,765 NaN) | Qwen pseudo-labels (N=2,457 valid) | Historical reference (Contaminated; NOT in E1d/E6b) |
| `hot_gold_only.csv` | 3,189 | `a655e0204afa53d5...` | Human gold (303 hate, 1,121 non-hate, 1,765 NaN) | **100% Masked to NaN (Zero pseudo-labels)** | **Actual E6b Auxiliary Input** (`exp6b_combined_text_gold_weighted`) |
| `combined_multibully_hot_v2.csv` | 8,907 | `c3496e411dbced3c...` | Human gold (1,883 hate, 5,259 non-hate) | Contains Qwen pseudo-labels from hot_v2 | Combined reference file (NOT used by E1d/E6b) |

### 4. Auxiliary Pseudo-Label Verification in E6b
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

### 1. Overall Task Supervision by Dataset Source

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

### 2. Per-Fold Supervision Breakdown in Training Partitions

| Fold | E1d Hate Labels | E6b Hate Labels | E1d/E6b Sarcasm Labels | E1d/E6b Target Labels | E1d/E6b Severity Labels |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 3,889 | **5,313** | 3,889 | 705 | 1,073 |
| **Fold 1** | 3,885 | **5,309** | 3,885 | 703 | 1,071 |
| **Fold 2** | 3,888 | **5,312** | 3,888 | 710 | 1,078 |
| **Fold 3** | 3,881 | **5,305** | 3,881 | 703 | 1,075 |
| **Fold 4** | 3,887 | **5,311** | 3,887 | 705 | 1,074 |

**Zero-Supervision Row Handling:**
- In `hot_gold_only.csv`, 1,765 rows correspond to offensive tweets where hate is masked to `NaN` and all other heads are `NaN`.
- In `src/toxifusionnet/experiments/cv.py`, lines 123-137 automatically exclude these 1,765 zero-supervision rows upon loading.
- Exactly 1,424 valid human-gold hate rows are retained in the training pool.
- On-disk files remain strictly untouched ($N = 3,189$).

---

## D. Cross-Dataset Leakage & Near-Duplicate Overlap Audit

A full-corpus pairwise leakage audit was executed comparing all MultiBully test/validation examples against all HOT training examples:

### 1. Exact Leakage & Identification Checks

| Leakage Measure | Scope | Threshold | Observed Matches | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Exact Normalized OCR Text** | 100% Full Corpus | Exact string equality | **0** | `PASS` |
| **Full Normalizer Pipeline Text** | 100% Full Corpus | Exact string equality | **0** | `PASS` |
| **Duplicate ID Collision** | 100% Full Corpus | ID equality | **0** | `PASS` |
| **Conflicting Hate Labels** | Matching text | Label discordance | **0** | `PASS` |

### 2. Near-Duplicate Overlap & Evaluated Scope

| Near-Duplicate Method | Evaluated Scope | Filtering / Candidate Selection | Threshold | Matches | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Character 3-5 N-Gram TF-IDF** | 100% Full Corpus (18,234,702 pairs) | Evaluated across all pairs without token length filtering | Cosine $\ge 0.85$ | **0** (Max = 0.7703) | `PASS` |
| **Word-Set Jaccard Similarity** | Texts with $\ge 3$ whitespace tokens | Inverted-index candidate screening on shared tokens (2,846,571 candidate pairs) | Jaccard $\ge 0.80$ | **0** | `PASS` |

> [!NOTE]
> **Documented Methodological Scope & Limitation:** Lexical overlap metrics (character TF-IDF and word Jaccard) detect surface and verbatim duplicates. They cannot establish semantic equivalence between paraphrases. The word-set Jaccard procedure filters out texts with fewer than three tokens (113 MultiBully, 59 HOT) to prevent degenerate 1-2 token collisions. Character TF-IDF cosine similarity evaluates all 18,234,702 pairs across the full corpus.

**Per-Fold Validation & Test Leakage:**
- **Fold_0:** Validation Overlap = `0`, Test Overlap = `0`.
- **Fold_1:** Validation Overlap = `0`, Test Overlap = `0`.
- **Fold_2:** Validation Overlap = `0`, Test Overlap = `0`.
- **Fold_3:** Validation Overlap = `0`, Test Overlap = `0`.
- **Fold_4:** Validation Overlap = `0`, Test Overlap = `0`.

---

## E. Label-Definition Consistency & Confounding Analysis

### 1. MultiBully Text-Label vs. Img-Text-Label Consistency & Divergence Reconciliation

**Reconciliation of 1,559 versus 1,725:**
- **Resolution Summary:** The figure **1,559** is the **net marginal difference** between whole-meme hate and text-only hate in the processed dataset ($3,139 - 1,580 = 1,559$). Because 111 examples exhibit reverse discordance (text hate $\cap$ meme non-hate), the true count of visual-only cyberbullying examples (discordant pairs where text is non-hate but the meme is hate) is **1,670** in processed MultiBully ($N = 5,718$, 29.21%) and **1,725** in raw MultiBully ($N = 5,865$, 29.41% total; 29.47% of labeled).

**Processed MultiBully Dataset (1-to-1 ID Merged, $N = 5,718$ matched rows, 0 dropped):**
- **Text-Only Hate (`Text-Label` derived):** Non-Hate = 4,138 (72.37%), Hate = 1,580 (27.63%).
- **Whole-Meme Hate (`Img-Text-Label` derived):** Non-Hate = 2,579 (45.10%), Hate = 3,139 (54.90%).
- **Visual-Only Cyberbullying (Text Non-Hate $\cap$ Meme Hate):** **1,670 memes (29.21%)**.
- **Reverse Discordance (Text Hate $\cap$ Meme Non-Hate):** **111 memes (1.94%)**.
- **Net Marginal Difference:** **1,559** ($3,139 - 1,580 = 1,670 - 111 = 1,559$).

**Raw MultiBully Dataset ($N = 5,865$ total, $5,854$ labeled):**
- **Raw `Text-Label`:** Nonbully = 4,246, Bully = 1,608, Unlabeled = 11.
- **Raw `Img-Text-Label`:** Bully = 3,222, Nonbully = 2,632, Unlabeled = 11.
- **Raw Visual-Only Cyberbullying (`Text-Label == Nonbully` $\cap$ `Img-Text-Label == Bully`):** **1,725 memes (29.41% total, 29.47% labeled)**.
- **Raw Reverse Discordance (`Text-Label == Bully` $\cap` `Img-Text-Label == Nonbully`):** **111 memes**.

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