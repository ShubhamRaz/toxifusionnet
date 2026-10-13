# ToxiFusionNet E6/E7: MultiBully + HOT Label Provenance & Taxonomic Audit

**Document:** `results/v2_combined_dataset/label_provenance_and_audit.md`  
**Date:** October 13, 2026  
**Status:** VERIFIED PRE-TRAINING AUDIT  
**Datasets:** MultiBully ($N=5,718$) + Hindi Offensive Tweets (`HOT`, $N=3,189$)  
**Combined Total:** $N=8,907$ rows  

---

## 1. Executive Summary & Verification Findings

This audit establishes the exact lineage and taxonomic mapping of every target variable across the combined MultiBully and HOT corpora for Experiments **E6 (Text-Only)** and **E7 (Multimodal CLIP)**.

### Key Governance Verdicts:
1. **Defensible Hate Speech Boundary**:
   - In the raw HOT dataset (`dataset.csv`), `orig_hot_label` contains three classes: **`0.0` (Normal, $N=1,121$)**, **`1.0` (Hate Speech, $N=303$)**, and **`2.0` (Offensive Language, $N=1,765$)**.
   - In standard content moderation science (Davidson et al., 2017; Founta et al., 2018), **offensive profanity/swearing is distinct from hate speech** (which requires targeted animus against protected identities, group-based contempt, or incitement).
   - Per the experiment protocol instructions, **the 1,765 Offensive samples are NOT automatically mapped to Hate**. Because offensive language cannot be defensibly mapped into binary hate without corrupting the decision boundary and inducing massive false alarms, they are assigned **`hate = NaN` (`IGNORE_INDEX = -100`)**.
   - Normal tweets ($N=1,121$) are mapped to `non_hate` (Human gold).
   - Hate tweets ($N=303$) are mapped to `hate` (Human gold).
2. **Explicit Demarcation of Qwen3:8B Pseudo-Labels**:
   - The HOT dataset **NEVER possessed human ground-truth annotations for sarcasm, target, or severity**.
   - All annotations for sarcasm, target, and severity present in `dataset.csv` were generated off-line by **`qwen3:8b`**.
   - These are strictly designated as **pseudo-labels**. Under no circumstances are they reported or evaluated as ground truth.
   - Uncertain or unannotated samples (`qwen_sarcasm == "uncertain"`, $N=732$; `qwen_target == "not_applicable"`, $N=2,154$; `qwen_severity == "not_applicable"`, $N=2,169$) are mapped to `NaN` (`IGNORE_INDEX = -100`), receiving zero loss penalty during training.
3. **MultiBully Gold Integrity**:
   - All 5,718 MultiBully annotations remain 100% human-annotated gold standards derived from Maity et al. (SIGIR 2022).
   - MultiBully validation and test partitions remain 100% untouched and isolated from HOT data.

---

## 2. HOT Dataset Cross-Tabulation & Label Alignment Matrix

Cross-tabulation of original human labels (`orig_hot_label`) against Qwen3:8B annotations across all 3,189 HOT records:

| Original HOT Label | Meaning | Total Records | Qwen3:8B `non_hate` | Qwen3:8B `hate` | V2 Canonical Hate Mapping | Supervision Status in E6/E7 |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **`0.0`** | **Normal** | **1,121** | 1,095 (97.7%) | 26 (2.3%) | **`non_hate`** | **Active Gold Supervision ($N=1,121$)** |
| **`1.0`** | **Hate Speech** | **303** | 134 (44.2%) | 169 (55.8%) | **`hate`** | **Active Gold Supervision ($N=303$)** |
| **`2.0`** | **Offensive Language** | **1,765** | 956 (54.2%) | 809 (45.8%) | **`NaN` (`-100`)** | **Masked Out (Zero Hate Penalty)** |
| **Total** | — | **3,189** | 2,185 | 1,004 | 1,424 valid, 1,765 masked | 1,424 active / 1,765 masked |

### Provenance Analysis:
- Notice that Qwen disagreed with the human ground truth on 44.2% of `1.0` hate speech samples (classifying them as `non_hate`). Using human gold labels rather than raw Qwen labels ensures higher fidelity and avoids synthetic LLM bias.
- For `2.0` (Offensive), Qwen split them almost 50/50 (45.8% hate vs 54.2% non-hate), confirming that profane Hindi slang without demographic animus is inherently ambiguous for binary hate classification. Masking them out prevents degrading the precision of the hate detector on cyberbullying memes.

---

## 3. Comprehensive Task Class Support Across Datasets

Summary of class distributions across MultiBully ($N=5,718$), HOT ($N=3,189$), and the combined training pool ($N=8,907$):

| Task | Class | MultiBully Gold ($N=5,718$) | HOT ($N=3,189$) [Provenance] | Combined Pool ($N=8,907$) |
|:---|:---|:---:|:---:|:---:|
| **Hate** | `non_hate` | 4,138 (72.4%) | 1,121 (Human Gold) | 5,259 |
| | `hate` | 1,580 (27.6%) | 303 (Human Gold) | 1,883 |
| | *Masked (`NaN` / `-100`)* | *0* | *1,765 (Offensive excluded)* | *1,765* |
| **Sarcasm** | `not_sarcastic` | 3,589 (62.8%) | 1,675 (Qwen Pseudo) | 5,264 |
| | `sarcastic` | 2,129 (37.2%) | 782 (Qwen Pseudo) | 2,911 |
| | *Masked (`NaN` / `-100`)* | *0* | *732 (Qwen Uncertain)* | *732* |
| **Target** | `individual` | 1,037 (72.7% of hate) | 692 (Qwen Pseudo) | 1,729 |
| | `community` | 251 (17.6% of hate) | 330 (Qwen Pseudo) | 581 |
| | `organization` | 81 (5.7% of hate) | 6 (Qwen Pseudo) | 87 |
| | `society` | 68 (4.8% of hate) | 7 (Qwen Pseudo) | 75 |
| | *Masked (`NaN` / `-100`)* | *4,281 (Non-hate conditional)*| *2,154 (Not applicable)* | *6,435* |
| **Severity**| `low` | 141 (8.9% of hate) | 0 (Qwen Pseudo) | 141 |
| | `moderate` | 1,409 (89.2% of hate) | 307 (Qwen Pseudo) | 1,716 |
| | `high` | 30 (1.9% of hate) | 713 (Qwen Pseudo) | 743 |
| | *Masked (`NaN` / `-100`)* | *4,138 (Non-hate conditional)*| *2,169 (Not applicable)* | *6,307* |

---

## 4. Class Imbalance Mitigation Strategy

1. **Loss Function Masking**:
   - For all tasks, loss is calculated using `valid = (y != -100) & (y.notna())`.
   - Samples with masked labels contribute zero loss and zero gradient to that task head.
2. **Dynamic Fold-Specific Class Weighting**:
   - `class_weights` for balanced focal loss are computed strictly on the inner training set for each fold.
   - Any class with zero support or masked indices is computed as $w_c = \frac{N_{\text{valid}}}{K \cdot \max(N_c, 1.0)}$.
3. **Task-Balanced Loss Weights**:
   - Following the successful finding of E5, task weights are set to $\{sarcasm: 1.0, hate: 1.0, target: 1.0, severity: 1.0\}$.
   - This prevents the target and severity heads from suffering gradient starvation when trained on the larger combined corpus.
