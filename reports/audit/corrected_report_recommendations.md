# Corrected Report Recommendations & Publication Readiness Guide

---

## 1. Executive Editorial Overview

To transition ToxiFusionNet from an internal engineering prototype to a defensible, publication-grade academic paper or open-source deliverable, several factual and terminological corrections must be applied across the project reports.

None of these corrections diminish the engineering success of the project: the multi-task model genuinely converges, the pipeline runs flawlessly, and real-world out-of-distribution text (*"Aloo paratha..."*) is correctly handled. However, rigorous peer reviewers at venues such as ACL, EMNLP, or ACM SIGIR will immediately flag metric heading confusions or exaggerated architecture claims.

---

## 2. Essential Required Corrections by File

### 2.1 Architecture Description (`README.md`, `STAGE_3_FINAL_REPORT.md`, `TOXIFUSIONNET_FULL_PROJECT_REPORT.md`)
- **Current Text:** *"Gated Sarcasm Cross-Attention Fusion Module"* / *"Gated Cross-Attention"*
- **Audit Finding:** The code implements additive attention pooling on each sequence, followed by an element-wise gated residual LayerNorm block ($f = \text{LayerNorm}(h + g \odot \tanh(W s))$). It does **not** perform Multi-Head Cross-Attention.
- **Recommended Text:**
  > *"Gated Sarcasm Residual Fusion Module (incorporating Sarcasm Probability Conditioning and Additive Attention Pooling)"*

---

### 2.2 Table Header Mislabeling: "Miss Rate (FNR)" vs. "Error Rate"
- **Current Table Header:** `Miss Rate (FNR)`
- **Audit Finding:** The values displayed in the table (`3.34%` for Hate, `4.50%` for Sarcasm) are $1 - \text{Accuracy}$ (overall error rate), **not** positive class False Negative Rate ($\text{FN} / (\text{FN} + \text{TP})$). The true FNR is $3.62\%$ for Hate and $6.95\%$ for Sarcasm.
- **Recommended Fix:** Change the table column header to:
  > `Overall Error Rate (1 - Acc)`
- **And update the specific FNR rows:**
  - True Hate FNR (all test posts): `3.62%`
  - True Sarcasm FNR (all test posts): `6.95%`
  - Hate FNR on Sarcastic Posts: `3.84%`
  - Hate FNR on Non-Sarcastic Posts: `2.87%`

---

### 2.3 Transparent Disaggregation of Test Performance
- **Current Framing:** Reports present the $97.14\%$ Macro-F1 as a uniform global test score, implying $97\%$ accuracy on natural social media Hinglish.
- **Audit Finding:** The $97.14\%$ score is heavily weighted by the synthetic 48k partition ($84.58\%$ of the test set), where the model achieves $100.0\%$ accuracy. On authentic human-annotated memes and tweets ($N = 1,313$), performance is **`73.62%` Hate F1** and **`60.21%` Sarcasm F1** (`59.07%` Mean Macro-F1).
- **Recommended Framing for Academic Paper:**
  > *"When evaluated on the composite test set ($N = 8,513$), ToxiFusionNet achieves a Mean Macro-F1 of $97.14\%$. On the held-out real-world social media slice ($N = 1,313$ posts from MultiBully and HOT), ToxiFusionNet achieves a Hate Macro-F1 of $73.62\%$ and Sarcasm Macro-F1 of $60.21\%$, substantially outperforming classical baselines ($64.9\%$ Hate F1) and single-task MultiBully models ($51.6\%$ Sarcasm F1)."*

---

### 2.4 Footnote on Sarcasm Evaluation Support ($N = 8,395$)
- **Current State:** The report lists Sarcasm support as $8,395$ without explaining why $118$ test posts were omitted.
- **Recommended Addition:**
  > *"Note on Task Support: Of the $8,513$ test samples, $118$ posts originating from the HOT corpus did not contain ground-truth sarcasm annotations and were appropriately masked during evaluation ($N_{\text{evaluated}} = 8,395$)."*

---

### 2.5 Explainability / Attribution Rationales
- **Current Text:** *"Token-level attribution rationales"*
- **Audit Finding:** Rationales are extracted directly from single-layer additive attention pooling weights (`self.hate_pool.score`), not post-hoc gradient attribution frameworks like Integrated Gradients, SHAP, or LIME.
- **Recommended Clarification:**
  > *"Token-level rationales are derived from the additive attention pooling weights of the specialized task heads."*

---

## 3. Git & Release Recommendations Before Handoff

1. **Push Branch to GitHub:**
   Currently, `feature/hinglish-balanced-dataset` exists only locally. Push it to the remote repository:
   ```powershell
   git push -u origin feature/hinglish-balanced-dataset
   ```
2. **Model Weights Release Strategy:**
   Because `runs/` is gitignored (preventing massive 1.13 GB binary files from bloating Git history), provide an external download link for `runs/exp_tri_dataset_xlmr/model.pt` via GitHub Releases or Hugging Face Model Hub, or document the 1-click training reproduction command (`.\run.ps1`).
