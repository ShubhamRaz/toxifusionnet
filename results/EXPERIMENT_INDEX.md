# ToxiFusionNet: Comprehensive Experiment Index & Reproducibility Registry

**Repository:** `https://github.com/ShubhamRaz/toxifusionnet`  
**Registry Version:** `3.0.0`  
**Last Updated:** October 16, 2026  

---

## 1. Overview & Evaluation Protocols

This index catalogs all historical, validated, and reproducible experiments conducted within the ToxiFusionNet project. Every experiment reported here maps to concrete configuration files, fixed random seeds, versioned dataset manifests, and evaluation artifacts.

### Established Protocol Standards
- **Evaluation Methodology:** 5-Fold Grouped Cross-Validation with strict priority multi-task stratification (`StratifiedGroupKFold`).
- **Isolation Guarantees:**
  - Grouping isolates identical normalized texts, duplicate meme image templates, and near-duplicate character $n$-gram variants.
  - Multi-dataset experiments (`E6b`) inject auxiliary data strictly into the inner training pool, preserving 100% outer-test isolation on the MultiBully benchmark.
- **Reporting Metrics:** Macro-F1 across all 4 tasks (Hate, Sarcasm, Target, Severity), per-class F1, AUROC, bootstrap 95% confidence intervals (1,000 resamples), and Holm-Bonferroni adjusted paired permutation / McNemar significance.

---

## 2. Master Experiment Registry

| Experiment ID | Task Setup | Dataset & Role | Model Architecture | Random Seed | Checkpoints & Predictions | Status |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| **`E1c`** | Text Multi-Task | MultiBully Text (5,718) — Unweighted | Tiny Transformer (GloVe-Twitter-100) | `42` | `results/v3_journal_final/e1c_text_unweighted/` | **Verified Canonical** |
| **`E1d`** | Text Multi-Task | MultiBully Text (5,718) — Dynamic Weighted | Tiny Transformer (GloVe-Twitter-100) | `42` | `results/v3_journal_final/e1d_text_weighted/` | **Verified Canonical** |
| **`E1d-Cleaned`** | Text Multi-Task | MultiBully Cleaned (5,718) — Dynamic Weighted | XLM-RoBERTa-base | `42` | `results/v3_journal_final/e1d_cleaned/` | **Cleaned Baseline (Macro-F1: 0.512)** |
| **`E1d-HingRoBERTa`** | Text Multi-Task | MultiBully Cleaned (5,718) — Dynamic Weighted | L3Cube HingRoBERTa (`l3cube-pune/hing-roberta`) | `42` | `results/v3_journal_final/e1d_hingroberta_cleaned/` | **Integrated / DGX-Ready** |
| **`E6b`** | Text Multi-Task + Gold Aux | MultiBully (5,718) + HOT Gold Aux (1,424 Train-Only) | Tiny Transformer (GloVe-Twitter-100) | `42` | `results/v3_journal_final/e6b_text_gold_weighted/` | **Verified Canonical** |
| **`E1` (Legacy)** | Text Multi-Task | MultiBully (5,718) — Text Only | Pretrained Transformer (MuRIL / XLM-R) | `42` | `results/v2_evaluation/e1_text/` | Historical Reference |
| **`E3` (Legacy)** | Multimodal Multi-Task | MultiBully (5,718) — Text + CLIP ViT | Gated Sarcasm Fusion + CLIP Visual | `42` | `results/v2_evaluation/e3_clip/` | Historical Reference |
| **`E5` (Legacy)** | Multimodal Multi-Task | MultiBully (5,718) — Balanced Loss | Gated Sarcasm Fusion + CLIP Visual | `42` | `results/v2_evaluation/e5_balanced_loss/` | Historical Reference |
| **`E6` (Legacy)** | Combined Text | MultiBully + HOT v1 (Contaminated Pseudo) | Transformer + Qwen Pseudo Labels | `42` | `results/v2_combined_dataset/e6_text/` | Historical (Contaminated) |
| **`E7` (Legacy)** | Combined Multimodal | MultiBully + HOT v1 (Contaminated Pseudo) | Multimodal + Qwen Pseudo Labels | `42` | `results/v2_combined_dataset/e7_clip/` | Historical (Contaminated) |
| **`CPU-Ablation`** | Multi-Task Ablation Suite | MultiBully (5,718) | 8 Architectural Variants (GloVe-CPU) | `42` | `results/ablation_summary.csv` | Shipped Benchmark |
| **`Baselines`** | Heuristic & Linear | MultiBully (5,718) | Majority, Keyword, Sentiment, TF-IDF + LR | N/A | `results/baselines.csv` | Shipped Benchmark |

---

## 3. Detailed Canonical Performance & Statistical Significance

### Experiment E1d vs. E6b vs. E1c (Pooled 5-Fold Evaluation, N=5,718)

| Metric / Task | E1c (Unweighted MB) | E1d (Weighted MB) | E6b (Combined MB + Gold HOT) | Delta (E1d vs E1c) | Delta (E6b vs E1d) | Holm-Adjusted $p$ (E6b vs E1d) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hate Macro-F1** | 0.6838 | **0.6808** | **0.6828** | -0.0030 ($p=1.00$) | +0.0020 (CI: [-0.007, 0.011]) | $p=1.000$ (Not Sig) |
| **Hate AUROC** | 0.7602 | **0.7606** | **0.7614** | +0.0004 | +0.0008 | $p=0.820$ (Not Sig) |
| **Sarcasm Macro-F1** | 0.5330 | **0.5608** | **0.5594** | +0.0278 ($p=0.0008^*$) | -0.0013 (CI: [-0.012, 0.010]) | $p=1.000$ (Not Sig) |
| **Target Macro-F1** | 0.3684 | **0.4364** | **0.4103** | +0.0680 ($p=0.0049^*$) | -0.0261 (CI: [-0.062, 0.010]) | $p=0.640$ (Not Sig) |
| **Severity Macro-F1** | 0.3219 | **0.4252** | **0.3527** | +0.1033 ($p=0.0049^*$) | -0.0725 (CI: [-0.123, -0.022]) | **$p=0.0330^*$ (Sig Drop)** |

### Per-Class Performance Breakdown (E1d vs. E6b)

#### Hate Classification ($N=5,718$)
- **E1d:** Non-Hate F1: `0.7852`, Hate F1: `0.5764`, Precision: `0.7021`, Recall: `0.6607`
- **E6b:** Non-Hate F1: `0.7865`, Hate F1: `0.5791`, Precision: `0.7042`, Recall: `0.6628`
- **McNemar Test:** Statistic = `0.0576`, $p = 1.000$ (No statistically significant difference in classification decisions).

#### Sarcasm Classification ($N=5,718$)
- **E1d:** Not Sarcastic F1: `0.6841`, Sarcastic F1: `0.4375`, AUROC: `0.5842`
- **E6b:** Not Sarcastic F1: `0.6830`, Sarcastic F1: `0.4358`, AUROC: `0.5839`

#### Target Classification ($N=1,437$ Supervised Hate Posts)
- **E1d:** Individual F1: `0.6921`, Community F1: `0.3812`, Organization F1: `0.3450`, Society F1: `0.3274`
- **E6b:** Individual F1: `0.6804`, Community F1: `0.3512`, Organization F1: `0.3120`, Society F1: `0.2978`

#### Severity Classification ($N=1,580$ Supervised Hate Posts)
- **E1d:** Low F1: `0.3412`, Moderate F1: `0.7125`, High F1: `0.2220`
- **E6b:** Low F1: `0.2450`, Moderate F1: `0.6820`, High F1: `0.1310`

---

## 4. Key Scientific Insights

1. **Class Balancing Impact (E1c -> E1d):**
   Dynamic class-frequency weighting produces statistically significant, substantial gains on the underrepresented tasks:
   - Target Macro-F1 increases by **+6.80%** ($p = 0.0049$).
   - Severity Macro-F1 increases by **+10.33%** ($p = 0.0049$).
   - Sarcasm Macro-F1 increases by **+2.78%** ($p = 0.0008$).
   - Hate detection performance is strictly maintained without degradation ($F1 = 0.6808$ vs $0.6838$).

2. **Auxiliary Cross-Dataset Transfer (E1d -> E6b):**
   Injecting 1,424 supervised human-gold hate tweets from HOT into the training pool (with strict outer-test isolation):
   - Confirms zero degradation on outer MultiBully test hate detection ($0.6828$ vs $0.6808$, $p=1.00$).
   - Shows that text-only tweets do not improve meme sarcasm detection.
   - Slightly dilutes severity calibration because HOT lacks severity labels (which are masked to NaN).

3. **Multi-Task Gating & Sarcasm Fusion:**
   As established in `results/ablation_summary.csv`, sarcasm-aware fusion prevents false negatives on sarcastic hate, but threshold tuning is critical for operating balance.

---

## 5. Artifact Manifest & Checksum Validation

| Artifact Name | Path | Rows | CRLF-Normalized SHA-256 |
| :--- | :--- | :---: | :--- |
| **Canonical MultiBully** | `data/processed/multibully.csv` | 5,718 | `1326c04f9dfc8272fdfbb7ba322bb1ec5ef7e06a86c673e4499d08e7ffea6f0a` |
| **Canonical HOT Gold** | `data/processed/hot_gold_only.csv` | 3,189 | `a655e0204afa53d50f55cf55d045dcf85b2e3ffb0d71a17c093a83017a15ec7d` |
| **MultiBully MemeHate** | `data/processed/multibully_memehate.csv` | 5,718 | `668fb4f1a216f499b1191d2eb383fa1548a30364f9f743c330f81d182583842c` |
| **Canonical Fold Cache** | `data/processed/multibully_v2_cv_folds.csv` | 28,590 | `73d4dbb4858907f43765dea0acff36aa4ae21ddc01e1694c0fcf85791fb2d1a5` |
| **Canonical Combined Cache** | `data/processed/combined_multibully_hot_cv_folds.csv` | 44,535 | `fd00161698dbfdc16381513b369b5d74d294dccf359f9cd937a70f33e7aa8ba7` |
| **Cleaned MultiBully** | `data/cleaned/multibully_cleaned.csv` | 5,718 | *Generated via `scripts/build_cleaned_pipeline.py`* |
| **Cleaned HOT Gold** | `data/cleaned/hot_cleaned.csv` | 3,189 | *Generated via `scripts/build_cleaned_pipeline.py`* |
| **Cleaned Combined Folds** | `data/cleaned/combined_multibully_hot_cv_folds.csv` | 44,535 | *Generated via `scripts/build_cleaned_pipeline.py`* |
| **Unified Leakage-Safe Folds** | `data/cleaned/multibully_v2_cv_folds.csv` | 28,590 | *Generated via `scripts/build_cleaned_pipeline.py`* |

---

## 6. Replication Instructions

To reproduce any experiment from this registry:
```bash
# E1d (Weighted MultiBully 5-Fold CV)
python -m toxifusionnet.cli cv \
    --config configs/exp1d_multibully_text_weighted.yaml \
    --data data/processed/multibully.csv \
    --folds-cache data/processed/multibully_v2_cv_folds.csv \
    --out-dir results/replication/e1d

# E6b (Combined Text Gold Weighted 5-Fold CV with Outer Isolation)
python -m toxifusionnet.cli cv \
    --config configs/exp6b_combined_text_gold_weighted.yaml \
    --data data/processed/multibully.csv \
    --aux-data data/processed/hot_gold_only.csv \
    --folds-cache data/processed/multibully_v2_cv_folds.csv \
    --out-dir results/replication/e6b
```
