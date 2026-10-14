# ToxiFusionNet v3: Per-Task Dataset Balancing Experiment Design

**Author:** Antigravity AI  
**Repository:** `/workspace/toxifusionnet`  
**Date:** October 2026  
**Audience:** Peer-Reviewed NLP / Multimodal AI Publication  

---

## 1. Context & Motivation

ToxiFusionNet evaluates multi-task classification across four distinct prediction tasks on the MultiBully meme benchmark:
1. **Hate Speech Detection** (Binary: `non_hate`, `hate`)
2. **Sarcasm Detection** (Binary: `not_sarcastic`, `sarcastic`)
3. **Target Identification** (4-Class Conditional: `individual`, `organization`, `community`, `society`)
4. **Severity Estimation** (3-Class Conditional: `low`, `moderate`, `high`)

### 1.1 The Imbalance Bottleneck
In empirical evaluations across E1 (text-only) and E3 (CLIP multimodal), Macro-F1 is disproportionately penalized by severe label imbalance in conditional sub-tasks:
- **Target Imbalance:** `individual` ($N = 1,037$) heavily dominates `community` ($N = 251$), while `organization` ($N = 81$) and `society` ($N = 68$) are ultra-minority classes.
- **Severity Imbalance:** `moderate` ($N = 1,409$) heavily dominates `low` ($N = 141$), while `high` severity ($N = 30$) constitutes only $1.89\%$ of supervised severity annotations.
- **Consequence:** Standard empirical risk minimization (unweighted Cross-Entropy) biases model heads toward majority classes (`individual`, `moderate`), yielding low recall and erratic F1 scores on minority classes ($F1 < 0.20$ on high severity and organization).

---

## 2. Mathematical Formulation: Square-Root Inverse-Frequency Balancing

Linear inverse-frequency weighting ($w_c = \frac{N}{K \cdot n_c}$) creates radical gradient disparities when applied to extreme minority classes. For instance, high severity ($n_{\text{high}} \approx 20$ in a training split of $N \approx 1,073$) receives an unconstrained weight of $w_{\text{high}} \approx 18.0$, inducing destructive gradient spikes that destabilize shared backbone representations and lead to early training divergence.

### 2.1 The Square-Root Inverse-Frequency Formula
To provide robust gradient signal to minority classes without gradient explosion, we implement **square-root inverse-frequency class weighting**:

$$w_c^{\text{raw}} = \sqrt{\frac{N}{K_{\text{active}} \cdot n_c}}$$

where:
- $n_c = \sum_{i \in \text{train}} \mathbb{I}(y_i = c \land y_i \neq \text{IGNORE\_INDEX})$ is the fold-specific training count for class $c$.
- $K_{\text{active}} = \sum_{c} \mathbb{I}(n_c > 0)$ is the number of active classes with non-zero training support.
- $N = \sum_{c: n_c > 0} n_c$ is the total number of supervised samples for the given task.

### 2.2 Normalization & Safety Capping
1. **Mean Normalization:** Weights are normalized such that the average weight across all active classes is identically 1.0:
   $$w_c^{\text{norm}} = \frac{w_c^{\text{raw}}}{\frac{1}{K_{\text{active}}} \sum_{j \in \text{active}} w_j^{\text{raw}}}$$
2. **Ceiling Capping:** Extreme outliers are capped at a configurable upper threshold ($w_{\max} = 4.0$):
   $$w_c^{\text{final}} = \min(w_c^{\text{norm}}, w_{\max})$$
3. **Zero-Support Classes:** If a class has zero training support in a fold ($n_c = 0$), its weight is explicitly set to $0.0$ to prevent division by zero or spurious loss contributions.
4. **Decoupling from Focal Loss:** Accidental double-balancing is eliminated by setting `focal_use_class_weights: false` when `focal_gamma > 0`, and using pure cross-entropy (`focal_gamma: 0.0`) in the balanced benchmark runs.

---

## 3. Experimental Matrix: E1c vs. E1d vs. E6b

To rigorously isolate the contribution of per-task class balancing from auxiliary text pre-training, we define a three-way controlled experiment matrix:

| Parameter | **E1c (Unweighted Control)** | **E1d (Class-Weighted)** | **E6b (Combined Gold + Balanced)** |
| :--- | :---: | :---: | :---: |
| **Config File** | `configs/exp1c_multibully_text_unweighted.yaml` | `configs/exp1d_multibully_text_weighted.yaml` | `configs/exp6b_combined_text_gold_weighted.yaml` |
| **Output Directory** | `results/v3_evaluation/e1c_text_unweighted` | `results/v3_evaluation/e1d_text_weighted` | `results/v3_evaluation/e6b_text_gold_weighted` |
| **Primary Dataset** | MultiBully ($N = 5,718$) | MultiBully ($N = 5,718$) | MultiBully ($N = 5,718$) |
| **Auxiliary Dataset** | None | None | HOT Gold-Only ($N = 3,189$ raw, **1,424 active**) strictly in train |
| **Zero-Supervision Excluded** | N/A | N/A | **1,765 rows excluded** (all 4 task labels masked) |
| **Class Weighting Mode** | `none` ($w_c = 1.0$) | `sqrt_balanced` | `sqrt_balanced` |
| **Weight Derivation Source** | N/A | MultiBully train fold | **MultiBully train fold only** |
| **Max Weight Cap** | N/A | 4.0 | 4.0 |
| **Task Loss Weights** | `{sarcasm: 1, hate: 1, target: 1, severity: 1}` | `{sarcasm: 1, hate: 1, target: 1, severity: 1}` | `{sarcasm: 1, hate: 1, target: 1, severity: 1}` |
| **Focal Modulation ($\gamma$)**| 0.0 (Pure Cross-Entropy) | 0.0 (Pure Cross-Entropy) | 0.0 (Pure Cross-Entropy) |
| **Backbone Architecture** | `xlm-roberta-base` | `xlm-roberta-base` | `xlm-roberta-base` |
| **Batch Size** | 64 | 64 | 64 |
| **Steps per Epoch** | **61** | **61** | **61** |
| **Total Max Steps** | **488** | **488** | **488** |
| **Warmup Steps** | **48** (`warmup_ratio: 0.10`) | **48** (`warmup_ratio: 0.10`) | **48** (`warmup_ratio: 0.10`) |
| **Learning Rates** | Encoder: $1\times 10^{-5}$, Head: $5\times 10^{-4}$ | Encoder: $1\times 10^{-5}$, Head: $5\times 10^{-4}$ | Encoder: $1\times 10^{-5}$, Head: $5\times 10^{-4}$ |
| **Early Stopping / Patience** | **Disabled (`false`)** / 0 | **Disabled (`false`)** / 0 | **Disabled (`false`)** / 0 |
| **Epochs / Validation Checks**| 8 epochs (evaluates every 61 steps) | 8 epochs (evaluates every 61 steps) | 8 epochs (evaluates every 61 steps) |
| **Checkpoint Selection** | Best Val Mean Macro-F1 | Best Val Mean Macro-F1 | Best Val Mean Macro-F1 |
| **Fold Split Cache** | `data/processed/multibully_v2_cv_folds.csv` | `data/processed/multibully_v2_cv_folds.csv` | `data/processed/multibully_v2_cv_folds.csv` |

---

## 4. Hypotheses, Matched Budget Accounting & Dataset Exposure

### 4.1 Scientific Hypotheses
- **$H_1$ (Balancing Effect):** E1d will achieve higher Macro-F1 than E1c on Target and Severity due to improved recall on minority classes (`organization`, `society`, `low`, `high`), without degrading Hate or Sarcasm detection.
- **$H_2$ (Transfer Interaction):** When trained with identical MultiBully-derived class weights and identical optimizer step budgets, E6b (MultiBully + HOT gold) will demonstrate whether auxiliary hate text supervision improves hate F1 or continues to produce negative transfer on conditional heads.

### 4.2 Exclusion of Zero-Supervision Auxiliary Rows
The raw derived dataset `data/processed/hot_gold_only.csv` contains $N = 3,189$ rows. An audit of supervision across all four tasks (`hate`, `sarcasm`, `target`, `severity`) reveals:
- **Zero-Supervision Rows ($N = 1,765$):** Samples from HOT with `orig_hot_label == 2.0` (offensive language masked to `NaN` to preserve binary hate gold ground truth) where sarcasm, target, and severity are also unannotated (`NaN`). If included in the training loader, these samples contribute `IGNORE_INDEX` (-100) across all heads, generating zero loss and zero gradient while needlessly consuming forward/backward cycles.
- **Valid Human-Gold Hate Rows ($N = 1,424$):** Samples with active human-annotated binary hate labels (1,121 `non_hate`, 303 `hate`).
- **Loader Filtering Action:** The DataLoader dynamically filters out all 1,765 zero-supervision rows upon loading. The on-disk file `data/processed/hot_gold_only.csv` remains strictly untouched ($3,189$ rows preserved). Exactly 1,424 valid human-labelled HOT hate examples are retained in the training loader.

### 4.3 Matched Optimizer Budget, Learning-Rate Schedule & Disabled Early Stopping
To ensure the comparison between E1d and E6b is strictly controlled, we do not rely solely on identical class weights. Differences in total gradient steps or learning-rate decay schedules would confound the experimental findings. We enforce:

1. **Step Budget Calibration:**
   - MultiBully training split has $N_{\text{MB}} \approx 3,885 \text{--} 3,889$ samples per fold. With `batch_size = 64`, this yields $\lceil 3889 / 64 \rceil = 61$ batches per epoch.
   - For 8 epochs, the reference budget is exactly **$61 \times 8 = 488$ optimizer steps**.
   - In E6b, combining MultiBully ($3,889$) and retained HOT gold ($1,424$) creates a pool of $5,313$ samples. Unconstrained training would yield $\lceil 5313 / 64 \rceil = 84$ batches/epoch ($672$ steps total), giving E6b $38\%$ more gradient updates.
   - By fixing `steps_per_epoch: 61` and `max_steps: 488` in E1c, E1d, and E6b, all models undergo **identically 488 optimizer updates**.
2. **Schedule Invariance:**
   - All models use linear warmup decay with `warmup_ratio = 0.10` calibrated to `total_steps = 488`, yielding identically **48 warmup steps**.
   - For every step $t \in [1, 488]$, the learning rate is mathematically identical between E1d and E6b.
3. **Disabled Early Stopping with Best Checkpoint Preservation:**
   - `early_stopping: false` and `patience: 0` are explicitly configured for all fixed-budget runs.
   - The trainer will **never terminate early**. All 488 planned optimizer updates will execute to completion.
   - Validation performance is evaluated at every 61-step checkpoint (at steps 61, 122, 183, 244, 305, 366, 427, 488).
   - Checkpoint selection preserves the model state corresponding to the highest validation Mean Macro-F1 (`eval_checkpoint_type: best_val_macro_f1`), ensuring final evaluation on test and val sets uses the best checkpoint.
   - `total_steps`, `steps_budget`, `best_epoch`, `best_step`, and per-checkpoint evaluation metrics are permanently recorded in `history.csv` and `cv_metrics.json`.

### 4.4 Sampling Strategy vs. Effective Dataset Exposure
A common reporting error is to describe an equalized optimizer update budget as "equal dataset exposure." In reality, while the update count is identical (488 steps), the sampling exposure per dataset diverges as expected:

- **E1d (MultiBully Only):**
  - Shuffled DataLoader over $N_{\text{MB}} \approx 3,886$ training samples.
  - In each epoch, 61 batches ($3,886$ samples) are drawn.
  - Total samples seen across 488 steps: $31,088$ sample draws.
  - **MultiBully Effective Exposure:** $31,088 / 3,886 = \mathbf{8.00\times}$ (8 full passes).
- **E6b (MultiBully + Retained HOT Gold):**
  - Shuffled combined DataLoader over $N_{\text{comb}} \approx 5,310$ samples ($73.2\%$ MultiBully, $26.8\%$ HOT gold).
  - In each epoch, exactly 61 batches of 64 ($3,904$ samples) are drawn uniformly at random from the cycled iterator.
  - Total samples seen across 488 steps: $488 \times 64 = 31,232$ sample draws.
  - **Combined Pool Effective Exposure:** $31,232 / 5,310 = \mathbf{5.88\times}$ passes per sample.
  - **Expected MultiBully Exposure:** $31,232 \times 0.732 \approx 22,862$ draws ($\mathbf{5.88\times}$ passes over MultiBully).
  - **Expected HOT Gold Exposure:** $31,232 \times 0.268 \approx 8,370$ draws ($\mathbf{5.88\times}$ passes across all 1,424 valid HOT samples).
- **Scientific Implication:** E6b trades $2.12\times$ passes ($26.8\%$) of MultiBully multimodal data for $5.88\times$ passes of auxiliary hate text data under a constant parameter-update budget. Any performance divergence directly isolates the substitution value of auxiliary text supervision without confounding from training duration.
