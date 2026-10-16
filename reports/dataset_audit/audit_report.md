# ToxiFusionNet: Dataset Quality & Pipeline Readiness Audit Report

**Audit Target:** `data/cleaned/` (`hot_cleaned.csv`, `multibully_cleaned.csv`, `combined_multibully_hot_v2.csv`, `combined_multibully_hot_cv_folds.csv`, `multibully_v2_cv_folds.csv`)  
**Audit Date:** October 16, 2026  
**Auditor:** Senior ML Data Engineer & Research Scientist  
**Overall Readiness Verdict:** `FAIL` — **NOT READY FOR TRAINING**

---

## Executive Summary & Scorecard

A thorough, read-only scientific audit of the five files in `data/cleaned/` was conducted against the ToxiFusionNet training pipeline, schema definitions, cross-validation protocols, and original human annotations. While the files exhibit valid CSV syntax, clean string columns, and correct task-conditional masking, **they suffer from six critical, disqualifying defects** that would compromise model training, introduce severe evaluation leakage, alter human ground truth, and trigger fatal pipeline runtime crashes.

| Check # | Audit Domain | Status | Severity | Primary Finding |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Pipeline Path Integration** | `FAIL` | **CRITICAL** | Zero configs in `configs/*.yaml` or scripts in `scripts/` reference `data/cleaned/`. All 24 active configs read from `data/processed/`. Any training run will silently ignore `data/cleaned/`. |
| **2** | **CV Checksum & Pipeline Crash** | `FAIL` | **FATAL** | `src/toxifusionnet/experiments/cv.py:45-46, 242-245` strictly asserts hardcoded SHA-256 hashes for 5,718-row / 8,907-row caches. The cleaned fold files will immediately crash `cv.py` with `ValueError`. |
| **3** | **Split Divergence & Leakage** | `FAIL` | **CRITICAL** | Re-splitting the combined dataset caused 48.7% of MultiBully (1,746 samples) and 48.0% of HOT (635 samples) to change splits. Exactly 501 MultiBully test samples leak directly into combined training. |
| **4** | **Image Duplicate CV Leakage** | `FAIL` | **CRITICAL** | `multibully_v2_cv_folds.csv` assigned `group_id` as sequential row indices (`0..N-1`). Exactly 34 duplicate image clusters cross between Train, Val, and Test in Fold 0. |
| **5** | **MultiBully Label Corruption** | `FAIL` | **CRITICAL** | Human gold annotations were silently altered: Target changed in 3,151 rows and Severity in 3,339 rows, while `label_source` falsely claims `gold` provenance. |
| **6** | **HOT Class 2.0 Pseudo-Labeling** | `FAIL` | **HIGH** | Original HOT class 2.0 ("Offensive Language") was not masked to NaN. 319 rows were mapped to `non_hate` and 184 to `hate` using unverified Qwen3:8b pseudo-labels. 231 human gold hate samples were discarded. |
| **7** | **Schema & Text Formatting** | `PASS` | `INFO` | Unique IDs, no missing text, length distributions valid, and non-hate conditional masking (target/severity = NaN) verified across all files. |

---

## Detailed Findings

### 1. Repository Inspection & Pipeline Path Binding (Task 1 & Task 2)
- **Status:** `FAIL`
- **Responsible Code Lines:**
  - `configs/multibully.yaml:5` (`processed_path: data/processed/multibully.csv`)
  - `configs/exp1d_multibully_text_weighted.yaml:6` (`processed_path: data/processed/multibully.csv`)
  - `configs/exp6b_combined_text_gold_weighted.yaml:7-8` (`processed_path: data/processed/multibully.csv`, `aux_path: data/processed/hot_gold_only.csv`)
  - `configs/exp_memehate_text.yaml:5-6` (`processed_path: data/processed/multibully_memehate.csv`, `cv_folds_path: data/processed/multibully_v2_cv_folds.csv`)
  - `src/toxifusionnet/cli.py:252, 254` (defaults to `data/processed/multibully.csv` and `data/processed/multibully_v2_cv_folds.csv`)
  - `scripts/run_final.sh:16` (`DATA=${DATA:-data/processed/multibully.csv}`)
- **Evidence:**
  Across the entire repository (`configs/*.yaml`, `src/`, `scripts/`), there are **zero references to `data/cleaned/`**. If a researcher executes `bash scripts/run_final.sh` or `python -m toxifusionnet.cli cv`, the pipeline will execute entirely on `data/processed/` artifacts, leaving `data/cleaned/` completely untouched.

---

### 2. Cross-Validation Checksums & Runtime Compatibility (Task 7)
- **Status:** `FAIL`
- **Responsible Code Lines:**
  - `src/toxifusionnet/experiments/cv.py:45-46`:
    ```python
    CANONICAL_FOLD_CACHE_SHA256 = "73d4dbb4858907f43765dea0acff36aa4ae21ddc01e1694c0fcf85791fb2d1a5"
    CANONICAL_COMBINED_FOLD_CACHE_SHA256 = "fd00161698dbfdc16381513b369b5d74d294dccf359f9cd937a70f33e7aa8ba7"
    ```
  - `src/toxifusionnet/experiments/cv.py:242-245`:
    ```python
    if cache_p.name == "multibully_v2_cv_folds.csv":
        validate_fold_cache_checksum(cache_p, CANONICAL_FOLD_CACHE_SHA256)
    elif cache_p.name == "combined_multibully_hot_cv_folds.csv":
        validate_fold_cache_checksum(cache_p, CANONICAL_COMBINED_FOLD_CACHE_SHA256)
    ```
- **Evidence:**
  The CRLF-normalized SHA-256 hashes of the fold caches in `data/cleaned/` are:
  - `data/cleaned/multibully_v2_cv_folds.csv`: `fe92c066149df0e9ff528c654f4dd18ce3b5b6b01a8e996c8780adb5260831b0`
  - `data/cleaned/combined_multibully_hot_cv_folds.csv`: `a7c10c45a9d494d54efe1a28fce7d389c08118173bd818710bd47bbf72bd4d00`
  Because `cv.py` explicitly validates the fold cache against the 5,718-row / 8,907-row canonical hashes to prevent unverified splits, passing either cleaned fold file to `cv.py` causes an immediate, unhandled `ValueError`.

---

### 3. Investigation of HOT Label Mapping & 319 Disagreements (Task 4)
- **Status:** `FAIL`
- **Responsible File:** `data/cleaned/hot_cleaned.csv`
- **Background & Taxonomy:**
  The original HOT (Hindi Offensive Tweets) dataset utilizes a 3-class human annotation schema:
  - `0.0`: **Normal** (Non-offensive, Non-hate)
  - `1.0`: **Hate Speech** (Attacks targeting protected identity, demographic, or community)
  - `2.0`: **Offensive Language** (Profanity, swearing, vulgar slang, banter, or personal abuse without protected group targeting)
- **What Occurred in `data/cleaned/hot_cleaned.csv`:**
  Instead of following the established repository gold standard (`data/processed/hot_gold_only.csv`), where class `2.0` is masked to `NaN` (`-100` IGNORE_INDEX) for binary hate supervision:
  1. The dataset directly adopted `qwen_hate` (pseudo-labels generated by Qwen3:8b) as ground truth.
  2. For the 503 retained rows with original label `2.0`:
     - **319 rows** were mapped to `non_hate`.
     - **184 rows** were mapped to `hate`.
  3. **231 human gold hate samples (class 1.0)** were discarded, leaving only 72 gold hate samples (down from 303 in `hot_gold_only.csv`).
- **Empirical Evidence from Sampled Tweets:**
  - **The 319 "disagreements" (Original 2.0 -> non_hate):**
    - `hot_18`: *"Deewar superhit nai aaj ke time ke hisaab se ATBB hai chutye dont compa."* (Casual film banter containing the slur "chutye", no identity targeted).
    - `hot_86`: *"kya zindagi h yaar bahenchod. dimag khrb. g m sb k sb. good night."* (Self-directed venting using profanity "bahenchod", no group targeted).
    - `hot_102`: *"Or koi gaali bhi nahi dega bahenchod😂"* (Conversational slang with laughing emoji).
    *Linguistic Verdict on the 319 samples:* They are genuine offensive language / profanity, but **not hate speech**. However, training a binary classifier with these samples as `non_hate` forces the encoder to associate explicit abusive keywords ("bahenchod", "chutye") with the negative hate class.
  - **The 184 samples (Original 2.0 -> hate):**
    - `hot_1`: *"Banti hai empowered woman, feminism pe gyan pelti hai... Pahle rona band kar madarchod!"* (Misogynistic attack targeting women/feminism).
    - `hot_7`: *"SUNNY LEONE BANA KE CHODEGE TERI MAA KO... MADARCHOD..."* (Extreme sexually violent threat targeting an individual female).
    - `hot_36`: *"Saniya, must say you are most madarchod randi on twitter..."* (Gendered sexual abuse).
    *Linguistic Verdict on the 184 samples:* These contain severe targeted harassment and misogyny. In a 3-class scheme, annotators labeled them "offensive", but in a binary hate scheme they cross into misogynistic abuse.
- **Root Conclusion:**
  Labeling class `2.0` into binary hate using an LLM creates an inconsistent, noisy supervision boundary and drops 76.2% of gold hate posts (231/303). The scientifically sound policy (already validated in `hot_gold_only.csv`) is to **mask class 2.0 to `NaN`**, computing cross-entropy loss only on unambiguous gold hate (1.0 -> hate) and gold normal (0.0 -> non_hate).

---

### 4. Split Assignment Divergence & Test Leakage (Task 5)
- **Status:** `FAIL`
- **Responsible Files:** `data/cleaned/multibully_cleaned.csv`, `data/cleaned/hot_cleaned.csv`, `data/cleaned/combined_multibully_hot_v2.csv`
- **Mechanistic Explanation:**
  - `multibully_cleaned.csv` was assigned a static 70/15/15 split derived from an isolated StratifiedGroupKFold on MultiBully alone (matching Fold 0 of `multibully_v2_cv_folds.csv` exactly: 3,582/3,582).
  - `hot_cleaned.csv` was similarly split in isolation.
  - When `combined_multibully_hot_v2.csv` was generated, a **global, re-stratified split** was run across the pooled 4,905 instances (matching Fold 0 of `combined_multibully_hot_cv_folds.csv`).
- **Disruption Metrics:**
  - MultiBully: **1,746 of 3,582 samples (48.7%) shifted split partition**.
    - 501 samples that are in the source MultiBully **test** split were placed into the combined **train** split!
    - 492 samples from source **train** became combined **test**.
  - HOT: **635 of 1,323 samples (48.0%) shifted split partition**.
    - 174 source **test** samples were placed into combined **train**.
- **Severity:**
  If a combined model is trained on `combined_multibully_hot_v2.csv`'s train partition and subsequently evaluated on the benchmark `multibully_cleaned.csv` test set, **501 test instances were present in the training set**. This is severe data leakage and invalidates any reported generalization performance.

---

### 5. CV Fold Isolation & Duplicate Image Leakage (Task 6)
- **Status:** `FAIL`
- **Responsible Files:** `data/cleaned/multibully_v2_cv_folds.csv`, `data/cleaned/combined_multibully_hot_cv_folds.csv`
- **Evidence:**
  - In `multibully_v2_cv_folds.csv`, the `group_id` column contains **3,582 unique values** (`0..3581`), and in `combined_multibully_hot_cv_folds.csv` it contains **4,905 unique values** (`0..4904`).
  - `group_id` is merely a sequential integer row index; it does **not** group related samples.
  - Image Hash Audit on `data/raw/bully_data/bully_data`:
    - 3,573 available images yield 3,486 unique MD5 hashes, revealing **66 duplicate image clusters**.
    - In Fold 0 of `multibully_v2_cv_folds.csv`, **34 of these duplicate image clusters leak across Train, Val, and Test**!
    - *Examples:*
      - Cluster `12f5732a`: `337.jpg` (train) and `3678.jpg` (test) have identical image content.
      - Cluster `c33490a2`: `577.jpg` (train) and `2193.jpg` (test) have identical image content.
      - Cluster `b4ae501e`: `1373.jpg` (train) and `1375.jpg` (test) have identical image content.
    - If multimodal / CLIP models are evaluated on these folds, test instances share identical visual inputs with training instances.

---

### 6. Human Ground Truth Alteration in MultiBully (Task 3)
- **Status:** `FAIL`
- **Responsible File:** `data/cleaned/multibully_cleaned.csv`
- **Evidence:**
  Comparing overlapping sample IDs between original processed MultiBully (`data/processed/multibully.csv`) and `multibully_cleaned.csv`:
  - **Target modified in 3,151 rows:**
    - 106 samples annotated as `individual` were changed to `organization`.
    - 89 samples annotated as `individual` were changed to `society`.
    - 72 samples annotated as `individual` were changed to `community`.
    - `0.jpg` (Shivam Girls meme): Raw annotation `target=individual` was changed to `target=community`.
  - **Severity modified in 3,339 rows:**
    - 195 samples annotated as `moderate` were changed to `high` (causing high severity to artificially jump from 30 to 231).
    - 414 samples annotated as `moderate` were changed to `low`.
  - **False Metadata Assertion:**
    The `label_source` column asserts `{"hate": "gold-derived", "target": "gold", "sarcasm": "gold", "severity": "gold"}`, which misrepresents synthetic/arbitrary label shifts as original human annotations.

---

## Specific Fixes Required Before Training

1. **Retain Canonical MultiBully Dataset (`data/processed/multibully.csv`):**
   - Do not replace human gold target/severity annotations with modified labels. Multi-task loss masking already handles rare classes without falsifying labels.
2. **Use Masked HOT Gold Auxiliary Data (`data/processed/hot_gold_only.csv`):**
   - Mask HOT class `2.0` (Offensive) to `NaN` for binary hate supervision.
   - Retain all 303 human gold hate samples and 1,121 normal samples. Mask sarcasm, target, and severity to `NaN` (0 pseudo-labels).
3. **Fix Group-Aware Fold Splitting:**
   - Group identical OCR texts AND duplicate image hashes into the same `group_id` before running `StratifiedGroupKFold`.
   - Ensure auxiliary HOT data is injected strictly into inner training folds, never into outer evaluation partitions.
4. **Update Pipeline Configurations & Checksums:**
   - If a new curated dataset version is released, update the configuration files in `configs/` and update `CANONICAL_FOLD_CACHE_SHA256` in `src/toxifusionnet/experiments/cv.py` with the newly computed, verified checksum.

---

## Verification Commands to Run Next

To verify the audit findings and run the automated validation test suite:

```powershell
# 1. Run the standalone dataset validation script
.venv\Scripts\python scripts/validate_cleaned_datasets.py

# 2. Run the repository test suite to verify baseline integrity
.venv\Scripts\python -m pytest tests/test_data_integrity_audit.py tests/test_matched_cv_integrity.py

# 3. Check data integrity report and provenance
.venv\Scripts\python scripts/verify_audit_integrity.py
```
