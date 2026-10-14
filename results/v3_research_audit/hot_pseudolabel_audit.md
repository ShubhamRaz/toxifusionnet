# HOT Dataset Provenance and Pseudo-Label Audit

**Audit Date:** October 2026  
**Reference Git Commit:** `0c1025e64c30730556c6bb1fbbd1c7f088cfb176`  
**Datasets Analyzed:**  
- Raw HOT Dataset: `data/raw/hot.csv` ($N = 3,189$)  
- Historical Combined Dataset: `data/processed/hot_v2.csv` ($N = 3,189$)  
- Clean Gold-Only Dataset: `data/processed/hot_gold_only.csv` ($N = 3,189$)

---

## 1. Executive Summary & Core Ethical / Scientific Directives

To maintain publication-grade scientific integrity:
1. **Never present model-generated annotations as human ground truth.**
2. In the auxiliary HOT dataset, **only binary hate speech annotations were provided by human annotators**.
3. All annotations in `data/processed/hot_v2.csv` for **sarcasm, target, and severity were generated via LLM pseudo-labeling (Qwen3:8B / Qwen2.5-7B)**.
4. Out-of-fold validation and test sets in cross-validation must **always consist exclusively of MultiBully human gold annotations**. Model predictions are **never** evaluated against pseudo-labels.

---

## 2. Dataset Provenance Breakdown

### 2.1 Original Human Ground Truth in HOT (`data/raw/hot.csv`)
The HOT corpus comprises 3,189 human-annotated text posts with three original labels:
- **`normal`**: 1,121 samples (35.15%)
- **`hate`**: 303 samples (9.50%)
- **`offensive`**: 1,765 samples (55.35%)

#### Handling the `offensive` Class in Binary Hate Supervision
In human annotator guidelines for hate speech, *offensive language* (e.g., profanity, coarse vernacular, swearing without group degradation) is distinct from *hate speech* (direct incitement or vilification against protected demographics).
- Forcing `offensive` into `hate` severely inflates false positive rates on benign colloquial speech.
- Forcing `offensive` into `non_hate` causes severe label noise on borderline abusive content.
- **Scientific Solution:** In both `hot_v2.csv` and `hot_gold_only.csv`, `offensive` instances have `hate = NaN`. They are safely excluded from the binary hate loss computation via ToxiFusionNet's task masking mechanism (`torch.isnan(labels)`).

Thus, exactly **1,424 HOT samples** provide human gold supervision for binary hate:
- `hate`: 303
- `non_hate`: 1,121
- Masked (`offensive`): 1,765

---

### 2.2 Provenance of Auxiliary Tasks in `hot_v2.csv` (Historical E6/E7)
The original HOT dataset contains **zero human annotations** for `sarcasm`, `target`, or `severity`.

In the historical E6 and E7 preparation pipeline (`scratch/curate_hot_v2.py`), an open-source LLM (`Qwen/Qwen2.5-7B-Instruct` / `Qwen3:8B`) was prompted to assign pseudo-labels for the three missing auxiliary tasks:
- **`sarcasm`**: 2,457 pseudo-labeled samples (716 `sarcastic`, 1,741 `non_sarcastic`, 732 `NaN`)
- **`target`**: 1,035 pseudo-labeled samples (629 `individual`, 231 `community`, 175 `organization`, 2,154 `NaN`)
- **`severity`**: 1,020 pseudo-labeled samples (598 `low`, 348 `medium`, 74 `high`, 2,169 `NaN`)

#### Provenance and Limitations of LLM Pseudo-Labels:
1. **Model Bias & Calibration:** Zero-shot LLM prompts lack inter-annotator agreement metrics (such as Cohen's $\kappa$ or Krippendorff's $\alpha$).
2. **Distribution Distortion:** LLMs often exhibit conservative bias, skewing toward `low` severity and misinterpreting subtle irony or slang as non-sarcastic.
3. **Task Cross-Contamination:** When an LLM infers target and severity from text that it also judged as hateful, the pseudo-labels share correlated model heuristics rather than orthogonal human perceptual judgments.

---

## 3. Human vs. Pseudo Supervision Comparison Matrix

The exact supervision counts available during training across corpora are summarized below:

| Prediction Task | MultiBully Human Gold ($N=5,718$) | HOT v2 Supervised ($N=3,189$) | HOT v2 Provenance | HOT Gold-Only Supervised ($N=3,189$) | HOT Gold-Only Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`hate`** | 5,718 (100.0%) | 1,424 (44.65%) | **Human Gold** | 1,424 (44.65%) | **Human Gold** |
| **`sarcasm`** | 5,718 (100.0%) | 2,457 (77.05%) | **Qwen3:8B Pseudo** | **0 (0.0%)** | **Masked (`NaN`)** |
| **`target`** | 1,437 (25.13%) | 1,035 (32.45%) | **Qwen3:8B Pseudo** | **0 (0.0%)** | **Masked (`NaN`)** |
| **`severity`** | 1,580 (27.63%) | 1,020 (31.98%) | **Qwen3:8B Pseudo** | **0 (0.0%)** | **Masked (`NaN`)** |

*Note: In MultiBully, target and severity are only annotated when the sample is labeled as bullying, explaining the ~25% coverage.*

---

## 4. Controlled Experimental Protocol

To decouple the benefits of auxiliary data from the effects of synthetic pseudo-labels, we establish a clean three-way controlled progression:

### 1. `exp1b_multibully_loss_equalized.yaml` (Pure In-Domain MultiBully Baseline)
- MultiBully only ($5,718$ samples).
- Task weights: $[1.0, 1.0, 1.0, 1.0]$.
- Supervised purely by human gold MultiBully annotations.

### 2. `exp6a_combined_text_gold_only.yaml` (Clean Auxiliary Text Hate Data)
- MultiBully ($5,718$) + HOT Gold Only ($3,189$).
- Adds 1,424 human-labeled hate/non-hate samples.
- Auxiliary tasks (`sarcasm`, `target`, `severity`) for HOT samples are **100% masked to `NaN`**.
- Directly tests: *Does auxiliary human text hate data improve meme hate detection when auxiliary heads remain uncontaminated?*

### 3. `exp6_combined_text.yaml` (Historical Pseudo-Label Ablation)
- MultiBully ($5,718$) + HOT v2 ($3,189$).
- Adds 1,424 human gold hate samples **plus** 2,457 pseudo-sarcasm, 1,035 pseudo-target, and 1,020 pseudo-severity samples.
- Directly tests: *Does multi-task transfer from LLM pseudo-labels provide regularizing signal or induce negative transfer?*

The same three-way controlled progression applies to multimodal models:
`exp3_multibully_clip.yaml` $\to$ `exp7a_combined_clip_gold_only.yaml` $\to$ `exp7_combined_clip.yaml`.
