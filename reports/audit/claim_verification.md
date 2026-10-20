# Comprehensive Research Claim Verification Matrix

---

## 1. Master Claim Audit Table

| # | Scientific / Technical Claim | Source Document & Section | Physical Evidence Inspected | Audit Status | Measured Ground Truth | Required Correction |
| :-: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | **Global Mean Macro-F1 is 0.9714 across 4 tasks** | `README.md` (Line 52), `STAGE_3_FINAL_REPORT.md` (Line 19) | `preds_test.csv` (8,513 rows), `scratch/audit_metrics.py` | **VERIFIED ✅** | `0.971377` | Disclose that score is heavily driven by 84.6% synthetic slice. |
| **2** | **Hate Speech Macro-F1 is 0.9665, AUROC is 0.9964** | `README.md` (Line 53), `STAGE_3_FINAL_REPORT.md` (Line 20) | `preds_test.csv`, `metrics.json` | **VERIFIED ✅** | Macro-F1: `0.966496`<br>AUROC: `0.996428` | None on arithmetic; clarify real-world slice F1 is 0.7362. |
| **3** | **Sarcasm Macro-F1 is 0.9548, AUROC is 0.9943** | `README.md` (Line 54), `STAGE_3_FINAL_REPORT.md` (Line 21) | `preds_test.csv`, `metrics.json` | **VERIFIED ✅** | Macro-F1: `0.954785`<br>AUROC: `0.994323` | None on arithmetic; clarify real-world slice F1 is 0.6021. |
| **4** | **Target Macro-F1 is 0.9822** | `README.md` (Line 55), `STAGE_3_FINAL_REPORT.md` (Line 22) | `preds_test.csv` (3,961 supervised rows) | **VERIFIED ✅** | `0.982173` | None on arithmetic; real-world slice F1 is 0.4677. |
| **5** | **Severity Macro-F1 is 0.9821** | `README.md` (Line 56), `STAGE_3_FINAL_REPORT.md` (Line 23) | `preds_test.csv` (3,982 supervised rows) | **VERIFIED ✅** | `0.982051` | None on arithmetic; real-world slice F1 is 0.5569. |
| **6** | **"Miss Rate (FNR)" is 3.34% for Hate, 4.50% for Sarcasm** | `README.md` (Lines 53–54), `STAGE_3_FINAL_REPORT.md` (Table 1) | Confusion matrices in `preds_test.csv` | **CONTRADICTED ❌** | Hate Error: `3.34%`<br>True Hate FNR: **`3.62%`**<br>Sarcasm Error: `4.50%`<br>True Sarcasm FNR: **`6.95%`** | **Rename table header from "Miss Rate (FNR)" to "Error Rate (1 - Acc)"** or display true positive FNR. |
| **7** | **Hate FNR on Sarcastic Posts is 3.84%** | `README.md` (Line 57), `STAGE_3_FINAL_REPORT.md` (Line 24) | Sarcastic subset ($N=4,017$), `metrics.json` | **VERIFIED ✅** | `0.038384` ($3.84\%$) | None. Correctly calculated as $\text{FN}/(\text{FN}+\text{TP})$. |
| **8** | **Unified Corpus contains 56,907 rows** | All reports | `combined_all_3_datasets.csv` | **VERIFIED ✅** | Exactly `56,907` rows | None. |
| **9** | **Sarcasm test support is 8,395 (118 missing)** | `reports/toxifusionnet_final_tri_dataset_evaluation.txt` | `test_df['sarcasm'].isna()` | **VERIFIED ✅** | Exactly 118 NaN rows in HOT test partition | State in footnotes that 118 unannotated HOT tweets were masked. |
| **10** | **Architecture uses "Gated Cross-Attention Fusion"** | `README.md`, `STAGE_3_FINAL_REPORT.md` | `models/toxifusionnet.py` (Line 37) | **CONTRADICTED ❌** | Additive attention pooling + Gated Residual LayerNorm block | **Replace "Cross-Attention" with "Gated Residual Fusion Block"**. |
| **11** | **Token-level attribution rationales** | `README.md`, `STAGE_3_FINAL_REPORT.md` | `cli.py`, `models/toxifusionnet.py` | **PARTIALLY VERIFIED ⚠️** | Single-layer Additive Attention Pooling weights | Clarify that rationales are attention weights, not SHAP or Integrated Gradients. |
| **12** | **Zero Data Leakage between splits** | `STAGE_3_FINAL_REPORT.md` (Section 2) | Text matching scripts | **PARTIALLY VERIFIED ⚠️** | 0 exact duplicates; 27 normalized overlaps ($0.31\%$); synthetic template overlap | Disclose that synthetic templates repeat across splits. |
| **13** | **GitHub Delivery Ready on branch feature/hinglish-balanced-dataset** | Workspace state | `git branch -r`, `git status` | **PARTIALLY VERIFIED ⚠️** | Branch is clean locally, but **unpushed to remote** | Must push branch to origin; model checkpoint is gitignored. |
| **14** | **Real-World Generalization is 97.14%** | `STAGE_3_FINAL_REPORT.md` (Section 1) | `audit_by_source.py` | **CONTRADICTED ❌** | Real-world test slice Macro-F1 is **`59.07%`** (Hate: `73.62%`) | Clarify that 97.14% is the composite score dominated by synthetic data. |

---

## 2. Reconciling Experiments Across Project Lifecycle

| Benchmark / Model Phase | Dataset Composition & Rows | Backbone Encoder | Evaluation Protocol | Reported Mean Macro-F1 | Actual Human Gold Hate F1 | Scope of Validity |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Phase 1: MultiBully Baseline** | MultiBully ($5,718$) | GloVe / MuRIL | 5-Fold Stratified CV | **`0.445 - 0.544`** | `0.649 - 0.724` | Authentic baseline on meme text; severe rare-class degradation. |
| **Phase 2: Simplified Categories** | MultiBully Simplified ($5,718$) | L3Cube HingRoBERTa | 5-Fold Grouped CV | **`0.608`** | `0.724` | MultiBully only; reduced taxonomy; verified out-of-fold predictions. |
| **Phase 3: Synthetic Only (48k)** | Hinglish Balanced ($48,000$) | XLM-RoBERTa | 70/15/15 Split | **`0.999`** | Fails OOD | Overfit to 840 template words; false positives on clean text. |
| **Phase 4: Tri-Dataset Production** | MultiBully + HOT + 48k ($56,907$) | XLM-RoBERTa | 70/15/15 Disjoint Split | **`0.971` (Global)**<br>**`0.591` (Real-World)** | **`0.736` (Real-World)** | **Current Production Model:** Real-world text generalizes cleanly; metrics disaggregated. |
