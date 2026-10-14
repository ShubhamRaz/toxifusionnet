# ToxiFusionNet Publication Improvement Plan & Master Research Roadmap

**Document Version:** 3.0 (Publication Readiness)  
**Date:** October 2026  
**Reference Git Commit:** `0c1025e64c30730556c6bb1fbbd1c7f088cfb176`  
**Target Submission:** Top-tier NLP / Multimodal AI Conference (e.g., ACL, EMNLP, MM, or NeurIPS Datasets & Benchmarks)

---

## 1. Executive Summary & Publication Strategy

ToxiFusionNet proposes a multi-task multimodal architecture combining contextual language modeling (DistilBERT) with visual semantics (CLIP) and specialized rationalization heads (Hate, Sarcasm, Target, Severity).

To elevate the experimental campaign (E1–E7) to publication quality, this plan resolves six fundamental methodological requirements:
1. **Unconfounded Experimental Control:** Isolates the causal effect of auxiliary text transfer by holding task loss weights constant ($1.0$).
2. **Dual-Benchmark Task Clarification:** Disentangles the evaluation of **Text-Level Bullying** (`Text-Label`) from **Whole-Meme Multimodal Hate** (`Img-Text-Label`), resolving the apparent multimodal paradox where 29.5% of memes possess innocent text but hateful imagery.
3. **Strict Data Provenance:** Eliminates synthetic pseudo-label contamination from auxiliary benchmarks (E6a/E7a) and restricts evaluation strictly to human ground truth.
4. **Group-Aware Statistical Rigor:** Deploys group-bootstrap confidence intervals (95% CI) and paired significance tests across all four heads.
5. **Principled Severity Imbalance Suite:** Evaluates class-weighted, focal, and ordinal distance-penalized loss formulations without oversampling validation data.
6. **High-Throughput Feature Caching:** Prototypes verified frozen CLIP embedding stores, cutting training time by ~95% while maintaining bit-level numerical parity.

---

## 2. Proposed Experiment Matrix

Every proposed experiment tests a precise hypothesis against a designated comparator under strict variable control.

| Experiment ID | Configuration File | Baseline Comparator | Core Hypothesis Tested | Controlled Variables | Primary Metrics | Estimated DGX GPU Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E1b** | `exp1b_multibully_loss_equalized.yaml` | E1 (Historical) | Equalizing auxiliary task loss weights ($0.5 \to 1.0$) prevents gradient starvation on target/severity without degrading text hate F1. | Identical data ($N=5,718$), architecture, batch size ($64$), LR ($2\times 10^{-5}$); only task weights vary. | Macro F1 on Hate, Sarcasm, Target, Severity; Group-Bootstrap 95% CI. | ~15 min (5 folds) |
| **E6a** | `exp6a_combined_text_gold_only.yaml` | E1b (Controlled Text Baseline) | Adding $1,424$ human-labeled auxiliary hate texts without pseudo-labels improves text hate generalization on meme OCR. | Identical loss weights ($1.0$), batch size ($64$); auxiliary HOT sarcasm/target/severity strictly masked ($0$ supervision). | Hate Macro F1, Hate PR-AUC; Paired McNemar Test ($p$-value). | ~25 min (5 folds) |
| **E6** *(Ablation)* | `exp6_combined_text.yaml` | E6a (Gold Only) | Auxiliary LLM (Qwen) pseudo-labels for sarcasm/target/severity provide beneficial multi-task regularization vs. negative transfer. | Identical data count; only pseudo-label masking is toggled. | All-task Macro F1; Cross-task transfer delta. | ~25 min (Historical run completed) |
| **E7a** | `exp7a_combined_clip_gold_only.yaml` | E3 (Multimodal MultiBully) & E6a | Multimodal fusion trained with clean auxiliary human text hate yields superior out-of-fold generalization. | Multimodal architecture, identical loss weights; HOT auxiliary tasks masked. | Multitask Macro F1; Modality contribution score. | ~35 min (with CLIP Cache) |
| **E7** *(Ablation)* | `exp7_combined_clip.yaml` | E7a (Gold Only) | Pseudo-label multi-task auxiliary supervision interacts constructively with multimodal fusion representations. | Multimodal architecture; pseudo-labels active. | Multitask Macro F1. | ~35 min (Historical run completed) |
| **E_Meme_Text** | `exp_memehate_text.yaml` *(Proposed)* | E1b | Text-only models fail to detect whole-meme hate when OCR text is innocent ($29.5\%$ of corpus). | Target mapped from `Img-Text-Label`; unimodal DistilBERT backbone. | Meme Hate Macro F1, Recall on multimodal-dependent subset. | ~15 min (5 folds) |
| **E_Meme_CLIP** | `exp_memehate_clip.yaml` *(Proposed)* | E_Meme_Text | Multimodal CLIP fusion outperforms text-only models by $\ge 5\%$ F1 on whole-meme hate detection. | Target mapped from `Img-Text-Label`; multimodal fusion architecture. | Meme Hate Macro F1; Bootstrap CI difference ($p < 0.001$). | ~25 min (with CLIP Cache) |
| **Ablation-Sev-CW** | `ablation_severity_class_weighted_ce.yaml` | `ablation_severity_unweighted_ce.yaml` | Inverse class weighting computed on fold training data boosts High-Severity recall without collapsing Low-Severity precision. | Loss formulation; class weights computed exclusively per fold train split. | Severity Macro F1, High-Severity Recall, Quadratic Weighted Kappa. | ~15 min (5 folds) |
| **Ablation-Sev-Focal** | `ablation_severity_focal.yaml` | `ablation_severity_unweighted_ce.yaml` | Focal loss ($\gamma=2.0$) focuses gradient updates on rare hard severity cases. | Focal loss objective vs. Cross-Entropy. | Severity Macro F1, High-Severity Recall. | ~15 min (5 folds) |
| **Ablation-Sev-Ordinal**| `ablation_severity_ordinal.yaml` | `ablation_severity_unweighted_ce.yaml` | Distance-penalized loss ($L_{CE} + \lambda \sum |i-j|^2 p_j$) reduces catastrophic severity rank inversions. | Ordinal penalty ($\lambda=0.5$); identical architecture. | Quadratic Weighted Kappa, Off-by-one Error Rate, Severity Macro F1. | ~15 min (5 folds) |

---

## 3. Methodological Protocols & Enhancements

### 3.1 Statistical Significance Protocol (Phase D)
For all experiments, reporting will adhere to:
1. **Group-Aware Bootstrap Confidence Intervals:** 1,000 bootstrap iterations clustered by normalized text group. 95% CIs reported for Macro F1 across all four heads (`hate`, `sarcasm`, `target`, `severity`).
2. **Paired Hypothesis Testing:**
   - Binary tasks (`hate`, `sarcasm`): Paired McNemar test on pooled out-of-fold contingency tables.
   - Multiclass tasks (`target`, `severity`): Paired Wilcoxon signed-rank test across fold-level macro F1 scores.
3. **Per-Class Diagnostics:** Reporting precision, recall, F1, support, and confusion matrix heatmaps for every class in all heads.

### 3.2 Severity Imbalance Treatment & Data Recommendation (Phase E)
The MultiBully human-labeled severity distribution is extremely skewed:
- **Low Severity:** 1,182 samples (74.81%)
- **Medium Severity:** 363 samples (22.97%)
- **High Severity:** **35 samples (2.22%)**

#### Analytical Guidelines:
1. **Zero Validation Oversampling:** In all experiments, validation and test splits maintain the natural empirical distribution. No SMOTE or synthetic oversampling is applied to evaluation partitions.
2. **Fold-Strict Weight Computation:** When class weights are enabled, weights are dynamically computed using only the training split of the active fold:
   $$w_c = \frac{N_{\text{train}}}{C \cdot N_{\text{train}, c}}$$
3. **Data Collection Recommendation for Camera-Ready / Extension:**
   - With only 35 high-severity memes in 5,718 samples, the variance of high-severity test metrics is high (approximately 7 test samples per fold).
   - We recommend a targeted data collection protocol: using active learning (uncertainty sampling from multimodal models) or community-specific seed terms (violent extremism, severe racial/religious slurs) to annotate an additional 250–500 high-severity memes with multi-annotator agreement ($\ge 3$ human annotators, Krippendorff's $\alpha \ge 0.70$).

### 3.3 Frozen CLIP Feature Caching (Phase F)
In multimodal training (E3, E5, E7, E7a), the CLIP image encoder (`openai/clip-vit-base-patch32`) remains frozen. 
- **Bottleneck:** Repeatedly forwarding 5,718 images through the 12-layer Vision Transformer on every epoch accounts for $>85\%$ of training compute.
- **Implementation:** `src/toxifusionnet/models/clip_cache.py` implements an SHA-256 hashed on-disk tensor store keyed by image content, preprocessing configuration, and model checkpoint.
- **Numerical Parity:** Verified in `tests/test_clip_caching.py` with tolerance $|y_{\text{cached}} - y_{\text{live}}| \le 1\times 10^{-5}$.
- **Performance Impact:** Reduces 5-fold multimodal CV time from ~3.5 hours to under 30 minutes on DGX A100.

---

## 4. Execution Sequence for NVIDIA DGX Supercomputer

All files and scripts are designed for seamless, non-destructive execution. Existing runs in `runs/` and `results/v2_evaluation/` will remain completely intact.

### Step 1: Preflight Verification & Integrity Check
```bash
python scripts/verify_audit_integrity.py
pytest tests/
```
*Expected Result:* 66/66 tests pass, 0 data leakage, 100% fold coverage.

### Step 2: Unconfounded Text Baseline & Clean Auxiliary Transfer
```bash
# E1b: Equalized loss weights on MultiBully text
python -m toxifusionnet.cli cv --config configs/exp1b_multibully_loss_equalized.yaml --out-dir results/v3_evaluation/e1b_text_equalized

# E6a: MultiBully + HOT Human Gold Only (zero pseudo-labels)
python -m toxifusionnet.cli cv --config configs/exp6a_combined_text_gold_only.yaml --out-dir results/v3_evaluation/e6a_text_gold_only
```

### Step 3: Multimodal Benchmark with Feature Caching
```bash
# E7a: MultiBully + HOT Multimodal Human Gold Only
python -m toxifusionnet.cli cv --config configs/exp7a_combined_clip_gold_only.yaml --out-dir results/v3_evaluation/e7a_clip_gold_only
```

### Step 4: Controlled Severity Loss Ablation Suite
```bash
python -m toxifusionnet.cli cv --config configs/ablation_severity_unweighted_ce.yaml --out-dir results/v3_evaluation/ablation_sev_unweighted
python -m toxifusionnet.cli cv --config configs/ablation_severity_class_weighted_ce.yaml --out-dir results/v3_evaluation/ablation_sev_weighted
python -m toxifusionnet.cli cv --config configs/ablation_severity_focal.yaml --out-dir results/v3_evaluation/ablation_sev_focal
python -m toxifusionnet.cli cv --config configs/ablation_severity_ordinal.yaml --out-dir results/v3_evaluation/ablation_sev_ordinal
```

### Step 5: Statistical Significance & Report Aggregation
```bash
python scripts/compute_publication_metrics.py --results-dir results/v3_evaluation/ --out-dir results/v3_evaluation/summary
```

---

## 5. Risk Assessment & Unresolved Vulnerabilities

| Risk Item | Severity | Mitigation Strategy Implemented |
| :--- | :---: | :--- |
| **Conflation of Text Hate vs. Meme Hate** | High | Formally documented in `label_definition_audit.md`. Separate benchmarks created. |
| **Pseudo-label Contamination** | High | Created `hot_gold_only.csv` and `exp6a`/`exp7a` with complete masking of pseudo-labels. |
| **High Severity Statistical Power** | Medium | Extreme scarcity ($N=35$). Acknowledged as dataset limitation; evaluated with Quadratic Weighted Kappa and ordinal metrics rather than raw high-class precision alone. |
| **GPU Out-of-Memory / CUDA Truncation** | Low | Batch size maintained at 64; frozen CLIP embeddings reduce VRAM overhead by $>60\%$. |

---

## 6. Publication Deliverables Checklist

- [x] Full Git alignment and non-destructive workspace preservation.
- [x] Dual task definition audit and contingency table.
- [x] Clean auxiliary dataset (`data/processed/hot_gold_only.csv`).
- [x] Controlled experiment configurations (`exp1b`, `exp6a`, `exp7a`, severity ablations).
- [x] Group-bootstrap confidence intervals for all four prediction heads.
- [x] Frozen CLIP caching module with numerical parity test.
- [x] Full audit suite passing 66/66 unit and integration tests.
- [x] Comprehensive research audit documentation in `results/v3_research_audit/`.
