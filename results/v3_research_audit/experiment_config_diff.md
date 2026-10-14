# Experiment Configuration Audit & Differential Analysis (E1–E7 & Proposed Extensions)

**Audit Date:** October 2026  
**Reference Git Commit:** `0c1025e64c30730556c6bb1fbbd1c7f088cfb176`  
**Repository Branch:** `main` (clean working tree verified)  
**Supercomputer Cluster:** NVIDIA DGX Pod (`/workspace/toxifusionnet`) vs Local Workspace

---

## 1. Executive Summary & Environment Verification

This audit examines the exact configuration parameters, architectural settings, task weighting, and dataset routing across all historical ToxiFusionNet experiments (E1, E3, E5, E6, E7) and establishes controlled counterpart configurations (E1b, E6a, E7a, and severity loss ablations).

### Git & Codebase Alignment
- **Active Git Commit:** `0c1025e` matches both local repository state and remote `origin/main`.
- **DGX Supercomputer Environment Parity:**
  - Python: 3.10+ / 3.13 venv
  - PyTorch: 2.6.0+cu124
  - Model Backbones: `distilbert-base-uncased` (text encoder, 66M params) and `openai/clip-vit-base-patch32` (vision encoder, 88M params, frozen).
  - All existing runs under `runs/` and evaluation outputs under `results/v2_evaluation/` are strictly preserved without modification or deletion.

---

## 2. Configuration Parameter Comparison Matrix

The table below contrasts the configurations across all historical runs and the new proposed controlled baselines.

| Parameter | E1 (Historical) | E1b (Controlled Baseline) | E3 (Historical) | E5 (Historical) | E6 (Historical) | E6a (Gold Hate Only) | E7 (Historical) | E7a (Multimodal Gold Only) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config File** | `exp1_multibully_text.yaml` | `exp1b_multibully_loss_equalized.yaml` | `exp3_multibully_clip.yaml` | `exp5_multibully_balanced_loss.yaml` | `exp6_combined_text.yaml` | `exp6a_combined_text_gold_only.yaml` | `exp7_combined_clip.yaml` | `exp7a_combined_clip_gold_only.yaml` |
| **Modalities Used** | Text only (`use_visual=False`) | Text only (`use_visual=False`) | Text + Image (`use_visual=True`) | Text + Image (`use_visual=True`) | Text only (`use_visual=False`) | Text only (`use_visual=False`) | Text + Image (`use_visual=True`) | Text + Image (`use_visual=True`) |
| **Dataset Source** | MultiBully ($N=5,718$) | MultiBully ($N=5,718$) | MultiBully ($N=5,718$) | MultiBully ($N=5,718$) | MultiBully ($5,718$) + HOT ($3,189$) | MultiBully ($5,718$) + HOT Gold ($3,189$) | MultiBully ($5,718$) + HOT ($3,189$) | MultiBully ($5,718$) + HOT Gold ($3,189$) |
| **Hate Supervision Source** | MultiBully `Text-Label` | MultiBully `Text-Label` | MultiBully `Text-Label` | MultiBully `Text-Label` | MultiBully `Text-Label` + HOT Human Gold | MultiBully `Text-Label` + HOT Human Gold | MultiBully `Text-Label` + HOT Human Gold | MultiBully `Text-Label` + HOT Human Gold |
| **Auxiliary Task Supervision** | MultiBully Human Gold | MultiBully Human Gold | MultiBully Human Gold | MultiBully Human Gold | MultiBully + Qwen Pseudo-labels | MultiBully Gold Only (HOT masked to NaN) | MultiBully + Qwen Pseudo-labels | MultiBully Gold Only (HOT masked to NaN) |
| **Task Loss Weights** $[w_{hate}, w_{sarc}, w_{targ}, w_{sev}]$ | $[1.0, 1.0, 0.5, 0.5]$ | $[1.0, 1.0, 1.0, 1.0]$ | $[1.0, 1.0, 0.5, 0.5]$ | Balanced $[1.0, 0.8, 1.2, 1.5]$ | $[1.0, 1.0, 1.0, 1.0]$ | $[1.0, 1.0, 1.0, 1.0]$ | $[1.0, 1.0, 1.0, 1.0]$ | $[1.0, 1.0, 1.0, 1.0]$ |
| **Hate Class Weighting** | Unweighted | Unweighted | Unweighted | Inverse Class Frequency | Unweighted | Unweighted | Unweighted | Unweighted |
| **Batch Size** | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| **Learning Rate** | $2 \times 10^{-5}$ | $2 \times 10^{-5}$ | $2 \times 10^{-5}$ | $2 \times 10^{-5}$ | $2 \times 10^{-5}$ | $2 \times 10^{-5}$ | $2 \times 10^{-5}$ | $2 \times 10^{-5}$ |
| **Max Epochs** | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| **Early Stopping** | Patience 2 on Val Loss | Patience 2 on Val Loss | Patience 2 on Val Loss | Patience 2 on Val Loss | Patience 2 on Val Loss | Patience 2 on Val Loss | Patience 2 on Val Loss | Patience 2 on Val Loss |
| **Cross-Validation Scheme** | 5-Fold Grouped CV (`cv_folds.csv`) | 5-Fold Grouped CV (`cv_folds.csv`) | 5-Fold Grouped CV (`cv_folds.csv`) | 5-Fold Grouped CV (`cv_folds.csv`) | 5-Fold Grouped CV (HOT in train only) | 5-Fold Grouped CV (HOT in train only) | 5-Fold Grouped CV (HOT in train only) | 5-Fold Grouped CV (HOT in train only) |
| **Out-of-Fold Evaluation Set** | MultiBully ($5,718$) | MultiBully ($5,718$) | MultiBully ($5,718$) | MultiBully ($5,718$) | MultiBully ($5,718$) | MultiBully ($5,718$) | MultiBully ($5,718$) | MultiBully ($5,718$) |
| **Cached CLIP Features** | N/A | N/A | False (On-the-fly) | False (On-the-fly) | N/A | N/A | False / Compatible | True (Opt-in Cache Store) |

---

## 3. Critical Analytical Findings & Confounding Variables

### 3.1 Confounding Variable: Loss Weights in E1 vs. E6
In historical E1 (`exp1_multibully_text.yaml`), task loss weights were configured as:
```yaml
task_weights:
  hate: 1.0
  sarcasm: 1.0
  target: 0.5
  severity: 0.5
```
In E6 (`exp6_combined_text.yaml`), task loss weights were equalized to:
```yaml
task_weights:
  hate: 1.0
  sarcasm: 1.0
  target: 1.0
  severity: 1.0
```
**Scientific Impact:** Comparing E6 directly against E1 introduces **two simultaneous variables**:
1. Addition of auxiliary HOT training samples ($+1,424$ hate supervision instances).
2. Doubling of the gradient backpropagation contribution from `target` and `severity` heads ($0.5 \to 1.0$).

Any performance shift observed between E1 and E6 cannot be causally attributed to the HOT dataset alone.  
**Resolution:** We created `exp1b_multibully_loss_equalized.yaml`. E1b serves as the true unconfounded baseline for E6 and E6a.

---

### 3.2 Target Label Discrepancy: `Text-Label` vs. `Img-Text-Label`
In all experiments E1–E7, the target label mapped for the primary binary task `hate` is:
```yaml
tasks:
  hate:
    from: Text-Label
    mapping:
      Bully: hate
      Nonbully: non_hate
```
As detailed in `label_definition_audit.md`, MultiBully includes both `Text-Label` and `Img-Text-Label`. In 1,725 memes (29.5% of the entire benchmark), `Text-Label` is `Nonbully` but `Img-Text-Label` is `Bully`.  
Evaluating multimodal fusion models (E3, E5, E7) against `Text-Label` penalizes the vision stream whenever the model correctly detects visual hate on a meme whose text is benign.

---

### 3.3 Pseudo-label Confounding in Auxiliary Tasks
In E6 and E7, auxiliary tasks (`sarcasm`, `target`, `severity`) for HOT samples were imputed using Qwen-based pseudo-labels. In contrast, in E6a and E7a, all auxiliary labels for HOT are masked (`NaN`), ensuring auxiliary heads are supervised exclusively by human ground truth.

---

## 4. Controlled Severity Loss Ablation Matrix

To address extreme class imbalance in severity without oversampling validation splits, four controlled configurations were created:

| Configuration File | Loss Objective | Class Weight Strategy | Ordinal Distance Penalty |
| :--- | :--- | :--- | :--- |
| `ablation_severity_unweighted_ce.yaml` | Standard Cross-Entropy | None ($w_c = 1.0$) | Disabled ($0.0$) |
| `ablation_severity_class_weighted_ce.yaml` | Class-Weighted Cross-Entropy | Inverse class frequency computed strictly on fold train partition | Disabled ($0.0$) |
| `ablation_severity_focal.yaml` | Focal Loss ($\gamma = 2.0, \alpha = 0.25$) | Implicit hard-sample down-weighting | Disabled ($0.0$) |
| `ablation_severity_ordinal.yaml` | Ordinal Distance-Penalized CE | Computed per fold | Active ($\lambda = 0.5$, penalizes squared grade error) |

All ablations maintain identical batch size (64), learning rate ($2 \times 10^{-5}$), 5-fold grouped partitions, and seed (42).
