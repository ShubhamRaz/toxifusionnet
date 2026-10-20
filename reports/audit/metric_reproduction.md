# Metric Reproduction & Statistical Verification Audit

---

## 1. Reproduction Methodology & Artifact Provenance

All metrics were re-computed from raw prediction records without relying on pre-aggregated JSON files.
- **Source Artifact:** [`runs/exp_tri_dataset_xlmr/preds_test.csv`](file:///c:/Users/SHUBHAM%20RAJ/Desktop/Others/Toxic/toxifusionnet/toxifusionnet/runs/exp_tri_dataset_xlmr/preds_test.csv) ($8,513$ rows).
- **Execution Script:** `scratch/audit_metrics.py`.
- **Python Environment:** Python 3.13, `scikit-learn 1.6.0`, `pandas 3.0.6`, `numpy 2.5.3`.

---

## 2. Global Test Score Verification ($N = 8,513$)

| Task / Metric | Reported Value | Recalculated Exact | Delta ($\Delta$) | Audit Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Mean Macro-F1** | **`0.9714`** | **`0.971377`** | $+0.000023$ | **VERIFIED ✅** |
| **Task 1: Sarcasm Macro-F1** | `0.9548` | `0.954785` | $+0.000015$ | **VERIFIED ✅** |
| Sarcasm Accuracy | `0.9550` | `0.954973` | $+0.000027$ | **VERIFIED ✅** |
| Sarcasm Balanced Accuracy | `0.9540` | `0.953966` | $+0.000034$ | **VERIFIED ✅** |
| Sarcasm AUROC | `0.9943` | `0.994323` | $-0.000023$ | **VERIFIED ✅** |
| Sarcasm Evaluated Samples | `8,395` | `8,395` | $0$ | **VERIFIED ✅** |
| **Task 2: Hate Speech Macro-F1**| `0.9665` | `0.966496` | $+0.000004$ | **VERIFIED ✅** |
| Hate Accuracy | `0.9666` | `0.966639` | $-0.000039$ | **VERIFIED ✅** |
| Hate Balanced Accuracy | `0.9665` | `0.966467` | $+0.000033$ | **VERIFIED ✅** |
| Hate AUROC | `0.9964` | `0.996428` | $-0.000028$ | **VERIFIED ✅** |
| Hate Evaluated Samples | `8,513` | `8,513` | $0$ | **VERIFIED ✅** |
| **Task 3: Target Macro-F1** | `0.9822` | `0.982173` | $+0.000027$ | **VERIFIED ✅** |
| Target Accuracy | `0.9813` | `0.981318` | $-0.000018$ | **VERIFIED ✅** |
| Target Evaluated Samples | `3,961` | `3,961` | $0$ | **VERIFIED ✅** |
| **Task 4: Severity Macro-F1** | `0.9821` | `0.982051` | $+0.000049$ | **VERIFIED ✅** |
| Severity Accuracy | `0.9817` | `0.981668` | $+0.000032$ | **VERIFIED ✅** |
| Severity Evaluated Samples | `3,982` | `3,982` | $0$ | **VERIFIED ✅** |

---

## 3. Explicit Arithmetic Check: "Miss Rate (FNR)" vs. "Error Rate"

### Critical Auditor Finding: Metric Heading Mislabeling
In the summary tables of `README.md` and `STAGE_3_FINAL_REPORT.md`, the column titled **"Miss Rate (FNR)"** actually displayed **$1 - \text{Accuracy}$ (Overall Error Rate)** for the four primary tasks:

| Evaluation Row | Confusion Matrix Values | True $\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}}$ | Reported Value in Table | What the Report Actually Displayed |
| :--- | :---: | :---: | :---: | :--- |
| **Hate Speech** | $\text{TN}=4393, \text{FP}=140$<br>$\text{FN}=144, \text{TP}=3836$ | **`3.62%`** ($144 / 3980$) | `3.34%` | **$1 - \text{Acc} = 3.34\%$ (Error Rate)** |
| **Sarcasm** | $\text{TN}=4279, \text{FP}=99$<br>$\text{FN}=279, \text{TP}=3738$ | **`6.95%`** ($279 / 4017$) | `4.50%` | **$1 - \text{Acc} = 4.50\%$ (Error Rate)** |
| **Target** | 74 misclassifications / 3961 | — | `1.87%` | **$1 - \text{Acc} = 1.87\%$ (Error Rate)** |
| **Severity** | 73 misclassifications / 3982 | — | `1.83%` | **$1 - \text{Acc} = 1.83\%$ (Error Rate)** |

### Sarcasm Subset Rows (Correct)
For the subset metrics at the bottom of the table:
- **Hate FNR on Sarcastic Posts:** True $\text{FNR} = 154 / 4017 =$ **`3.84%`** *(Reported correctly as 3.84%)*.
- **Hate FNR on Non-Sarcastic Posts:** True $\text{FNR} = 111 / 3871 =$ **`2.87%`** *(Reported correctly as 2.87%)*.

**Required Correction:** The report column heading must be corrected to **"Error Rate ($1 - \text{Acc}$)"**, or the numbers updated to the actual positive False Negative Rates ($3.62\%$ and $6.95\%$).

---

## 4. Disaggregated Performance Audit: Synthetic vs. Real-World Data

When the test predictions ($N = 8,513$) are disaggregated by data source, a major empirical divergence emerges:

| Evaluation Split | Test Samples | Hate Macro-F1 | Sarcasm Macro-F1 | Target Macro-F1 | Severity Macro-F1 | 4-Task Mean Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Synthetic Partition (`hinglish_balanced`)** | **$7,200$ ($84.6\%$)** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000` (100.0%)** |
| **MultiBully Gold Partition (Memes)** | **$858$ ($10.1\%$)** | **`0.7037`** | **`0.5283`** | **`0.4173`** | **`0.3068`** | **`0.4890` (48.9%)** |
| **HOT Tweet Partition (Twitter/X)** | **$455$ ($5.3\%$)** | **`0.7735`** | **`0.7526`** | **`0.5936`** | **`0.6007`** | **`0.6801` (68.0%)** |
| **Combined Real-World Test Slice** | **$1,313$ ($15.4\%$)** | **`0.7362`** | **`0.6021`** | **`0.4677`** | **`0.5569`** | **`0.5907` (59.07%)** |

### Research Interpretation
1. **The $97.14\%$ Score is Statistically Dominated by Synthetic Templates:**
   Because synthetic data comprises $84.58\%$ of the test set and scores $100\%$, the aggregate metric reflects synthetic pattern matching rather than real-world Hinglish comprehension.
2. **Defensible Real-World Baseline:**
   On authentic social media posts ($1,313$ test samples), ToxiFusionNet achieves:
   - **`73.62%` Hate Macro-F1** (an improvement over the classical TF-IDF baseline of $64.9\%$).
   - **`60.21%` Sarcasm Macro-F1** (substantially higher than MultiBully-only models of $\approx 52\%$).
   - **`59.07%` Real-World Multi-Task Mean F1**.
