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
| **Auxiliary Dataset** | None | None | HOT Gold-Only ($N = 3,189$) strictly in train |
| **Class Weighting Mode** | `none` ($w_c = 1.0$) | `sqrt_balanced` | `sqrt_balanced` |
| **Weight Derivation Source** | N/A | MultiBully train fold | **MultiBully train fold only** |
| **Max Weight Cap** | N/A | 4.0 | 4.0 |
| **Task Loss Weights** | `{sarcasm: 1, hate: 1, target: 1, severity: 1}` | `{sarcasm: 1, hate: 1, target: 1, severity: 1}` | `{sarcasm: 1, hate: 1, target: 1, severity: 1}` |
| **Focal Modulation ($\gamma$)**| 0.0 (Pure Cross-Entropy) | 0.0 (Pure Cross-Entropy) | 0.0 (Pure Cross-Entropy) |
| **Backbone Architecture** | `xlm-roberta-base` | `xlm-roberta-base` | `xlm-roberta-base` |
| **Batch Size** | 64 | 64 | 64 |
| **Learning Rates** | Encoder: $1\times 10^{-5}$, Head: $5\times 10^{-4}$ | Encoder: $1\times 10^{-5}$, Head: $5\times 10^{-4}$ | Encoder: $1\times 10^{-5}$, Head: $5\times 10^{-4}$ |
| **Patience / Epochs** | 3 / 8 | 3 / 8 | 3 / 8 |
| **Fold Split Cache** | `data/processed/multibully_v2_cv_folds.csv` | `data/processed/multibully_v2_cv_folds.csv` | `data/processed/multibully_v2_cv_folds.csv` |

---

## 4. Key Hypotheses & Optimizer Update Analysis

### 4.1 Scientific Hypotheses
- **$H_1$ (Balancing Effect):** E1d will achieve higher Macro-F1 than E1c on Target and Severity due to improved recall on minority classes (`organization`, `society`, `low`, `high`), without degrading Hate or Sarcasm detection.
- **$H_2$ (Transfer Interaction):** When trained with identical MultiBully-derived class weights, E6b (MultiBully + HOT gold) will demonstrate whether auxiliary hate text supervision improves hate F1 or continues to produce negative transfer on conditional heads.

### 4.2 Optimizer Step Accounting
Because HOT samples are injected exclusively into the training splits of E6b:
- **E1c & E1d Training Size:** ~4,003 samples per fold $\rightarrow \lceil 4003 / 64 \rceil = 63$ batches/epoch $\rightarrow 504$ total steps across 8 epochs.
- **E6b Training Size:** 4,003 + 3,189 = 7,192 samples per fold $\rightarrow \lceil 7192 / 64 \rceil = 113$ batches/epoch $\rightarrow 904$ total steps across 8 epochs.
- **Schedule Invariance:** Both configurations employ `get_linear_schedule_with_warmup` with `warmup_ratio = 0.10` calibrated dynamically to `steps = len(dl_tr) * epochs`, ensuring identical relative warmup profiles.
