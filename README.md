# ToxiFusionNet 🛡️

**Sarcasm-Aware Multi-Task Transformer for Hate Speech Detection in Hindi-English (Hinglish) Code-Mixed Text.**

ToxiFusionNet detects hate speech—specifically identifying nuanced and covert *sarcastic hate*—and simultaneously predicts the **target** and **severity** level of each detected attack with token-level rationales and explainability.

```
                          ┌────────────────────────┐
                          │ Code-Mixed Hinglish    │
                          └───────────┬────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        ┌──────────────────────┐              ┌──────────────────────┐
        │ XLM-RoBERTa Backbone │              │ Pragmatic Sarcasm    │
        │ & Sub-Word Tokenizer │              │ Surface Cues Vector  │
        └──────────┬───────────┘              └──────────┬───────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Sarcasm Branch Head     │
                        │   P(Sarcasm | Text)       │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Gated Sarcasm Cross-      │
                        │ Attention Fusion Module   │
                        └─────────────┬─────────────┘
                                      │
                   ┌──────────────────┼──────────────────┐
                   ▼                  ▼                  ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │   Hate Speech    │ │  Target Head     │ │  Severity Head   │
        │   Detection      │ │ (Individual /    │ │ (Low / Moderate /│
        │ (Non-Hate / Hate)│ │  Org/Comm/Society│ │  High)           │
        └──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## Benchmark Results (Stage 3 Tri-Dataset Benchmark)

ToxiFusionNet was trained and evaluated on the unified **Tri-Dataset Merge** ($N = 56,907$ rows), integrating gold human-annotated memes from **MultiBully** ($5,718$), real-world tweets from **HOT** ($3,189$), and balanced sarcasm/target coverage from the **48k corpus**.

### Official Evaluation Scorecard (Held-Out Test Set, $N = 8,513$ posts)

| Task / Objective | Macro-F1 | Accuracy | AUROC | Balanced Acc | Miss Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Multi-Task Average** | **`0.9714`** | — | — | — | — |
| **Hate Speech Detection (`hate`)** | **`0.9665`** | `0.9666` | **`0.9964`** | `0.9665` | `3.34%` |
| **Sarcasm Detection (`sarcasm`)** | **`0.9548`** | `0.9550` | **`0.9943`** | `0.9540` | `4.50%` |
| **Target Identification (`target`)** | **`0.9822`** | `0.9813` | — | `0.9820` | `1.87%` |
| **Severity Prediction (`severity`)** | **`0.9821`** | `0.9817` | — | `0.9821` | `1.83%` |
| **Hate Detection on Sarcastic Posts** | **`0.9699`** | — | — | — | **FNR: `3.84%`** |
| **Hate Detection on Non-Sarcastic Posts**| **`0.9628`** | — | — | — | **FNR: `2.87%`** |

*For complete training dynamics, cross-dataset split details, and ablation comparisons, see [STAGE_3_FINAL_REPORT.md](STAGE_3_FINAL_REPORT.md) and [TOXIFUSIONNET_FULL_PROJECT_REPORT.md](reports/TOXIFUSIONNET_FULL_PROJECT_REPORT.md).*

---

## Dataset Provenance & References

The unified **56,907** row corpus combines three curated sources:

1. **Hinglish Toxic Hate Dataset ($48,000$ rows):**
   - **Repository / URL:** [Kaggle: Hinglish Toxic Hate Dataset](https://www.kaggle.com/datasets/subamraz/hinglish-toxic-hate-dataset/data)
   - **Characteristics:** Synthetically balanced anchor corpus engineered for 1:1 balance across sarcasm ($P(\text{hate}\mid\text{sarcasm})=0.50$), all 4 target classes, and all 3 severity ratings to prevent majority-class collapse.

2. **MultiBully Dataset ($5,718$ rows):**
   - **Repository / URL:** [Kaggle: Multi-Bully Dataset](https://www.kaggle.com/datasets/pranee5/multi-bully)
   - **Reference Paper:** Maity et al., *"MultiBully: Multimodal Cyberbullying Detection in Code-Mixed Social Media Posts"*, ACM SIGIR 2022.
   - **Characteristics:** Real-world multimodal memes with gold human annotations for cyberbullying, sarcasm, victim target, and harm score.

3. **HOT (Hindi Offensive Tweet) Dataset ($3,189$ rows):**
   - **Repository / URL:** [GitHub: HOT_Dataset_modified.csv](https://github.com/ramitsawhney27/OffensiveHinglishTweetClassification/blob/master/HOT_Dataset_modified.csv)
   - **Reference Project:** Ramit Sawhney et al., *Offensive Hinglish Tweet Classification*.
   - **Characteristics:** Real-world code-mixed Twitter/X posts, enriched with Qwen multi-task annotations for target and severity.

---

## Quick Start & Inference

### 1. Installation

```powershell
# Create & activate environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install toxifusionnet package in editable mode
pip install -e .
```

### 2. Run Inference on Any Text

Test the production model using the CLI:

```powershell
python -m toxifusionnet.cli predict `
  --run-dir runs/exp_tri_dataset_xlmr `
  --text "Congrats bro, strict diet ke beech spicy golgappe ne prove kar diya ki hope pe world kayam hai" `
         "Saale ye lobby ke log itna disgusting aur pathetic harkat karta hai" `
         "Aloo paratha bahut tasty tha yaar"
```

The output returns JSON predictions containing:
- Binary sarcasm classification (`sarcastic` vs `not_sarcastic`)
- Hate speech probability (`non_hate` vs `hate`)
- Hate-conditional target prediction (`individual`, `organization`, `community`, `society`)
- Severity rating (`low`, `moderate`, `high`)
- Gated fusion attention weights and token-level attribution rationales

---

## Reproducing Training

To re-run training on the unified tri-dataset from scratch:

```powershell
python -m toxifusionnet.cli train --config configs/combined_all_3_datasets.yaml
```

The configuration automatically engages GPU acceleration (CUDA with FP16 AMP) and saves checkpoints and evaluation metrics to `runs/exp_tri_dataset_xlmr/`.

---

## Directory Structure

```
toxifusionnet/
├── configs/
│   └── combined_all_3_datasets.yaml    # Production training configuration
├── data/
│   └── processed/
│       ├── combined_all_3_datasets.csv # Canonical unified 56.9k dataset
│       └── combined_all_3_datasets.meta.json
├── runs/
│   └── exp_tri_dataset_xlmr/
│       ├── model.pt                    # 1.13 GB trained weights checkpoint
│       ├── metrics.json                # Complete evaluation metrics
│       ├── history.csv                 # Per-epoch loss and validation metrics
│       ├── preds_test.csv              # Test prediction dump
│       └── tokenizer/                  # XLM-RoBERTa tokenizer
├── src/
│   └── toxifusionnet/                  # Core package modules
│       ├── models/                     # ToxiFusionNet architecture & losses
│       ├── preprocessing/              # Normalization & language identification
│       └── training/                   # Trainer, class-balancing & metrics
├── STAGE_3_FINAL_REPORT.md             # Comprehensive scientific report
└── README.md                           # Master project documentation
```
