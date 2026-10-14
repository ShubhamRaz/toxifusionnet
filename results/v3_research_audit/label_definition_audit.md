# MultiBully Label Definition Audit: Text-Label vs. Img-Text-Label

**Audit Date:** October 2026  
**Reference Git Commit:** `0c1025e64c30730556c6bb1fbbd1c7f088cfb176`  
**Dataset Analyzed:** `data/raw/multibully.csv` ($N = 5,865$, 5,854 valid records) and `data/processed/multibully.csv` ($N = 5,718$)

---

## 1. Executive Summary & Root Scientific Issue

In the original MultiBully annotation protocol, annotators provided two distinct binary bullying judgments for each meme:
1. **`Text-Label`**: Evaluates whether the extracted OCR text **in isolation** constitutes bullying/hate.
2. **`Img-Text-Label`**: Evaluates whether the **combined multimodal meme** (the image in conjunction with the OCR text) constitutes bullying/hate.

### The Problem in Prior Experiments (E1–E7)
In all existing ToxiFusionNet experiments (E1, E3, E5, E6, E7), the primary binary classification task (`hate`) was mapped exclusively from `Text-Label`:
```yaml
tasks:
  hate:
    from: Text-Label
    mapping:
      Bully: hate
      Nonbully: non_hate
```
**Consequence:** Models were trained and evaluated on whether the **OCR text alone** was bullying, **not** whether the meme as a whole was toxic.  
When multimodal models (E3, E5, E7) leveraged visual signals from hateful imagery (e.g., swastikas, offensive visual tropes, physical threats depicted in the image) on memes with innocent text captions (e.g., "Look at this guy", "Dinner time"), the ground truth label `Text-Label` marked them as `Nonbully`. Consequently, the multimodal model was **penalized for being correct** regarding the meme's true toxicity.

---

## 2. Contingency Analysis & Empirical Label Distributions

From our analysis of the complete raw MultiBully corpus ($N = 5,854$ non-null records):

| Annotation Field | Nonbully Count | Nonbully % | Bully Count | Bully % | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Text-Label`** (Text in isolation) | 4,246 | 72.53% | 1,608 | 27.47% | 5,854 |
| **`Img-Text-Label`** (Whole meme) | 2,632 | 44.96% | 3,222 | 55.04% | 5,854 |

### 2x2 Contingency Table: `Text-Label` vs. `Img-Text-Label`

| `Text-Label` $\downarrow$ \ `Img-Text-Label` $\rightarrow$ | `Nonbully` (Image+Text Benign) | `Bully` (Image+Text Bullying) | Total `Text-Label` |
| :--- | :--- | :--- | :--- |
| **`Nonbully`** (Text Benign) | **2,521** (43.06%) *(Concordant Benign)* | **1,725** (29.47%) *(Multimodal-Dependent Hate)* | 4,246 (72.53%) |
| **`Bully`** (Text Bullying) | **111** (1.90%) *(Text Bully / Meme Context Dampened)* | **1,497** (25.57%) *(Concordant Bully)* | 1,608 (27.47%) |
| **Total `Img-Text-Label`** | 2,632 (44.96%) | 3,222 (55.04%) | 5,854 (100.0%) |

---

## 3. Key Observations & Failure Mode Taxonomy

### 3.1 The 1,725 Multimodal-Dependent Memes (29.47%)
Almost **one-third (29.5%) of the entire MultiBully corpus** consists of memes where the OCR text is innocuous or neutral, but the visual meme context is hateful or bullying.
- **Example Scenario A:** Image shows a grotesque racist caricature; OCR text says *"Good morning to everyone except them"*.
  - `Text-Label`: `Nonbully` (neutral text).
  - `Img-Text-Label`: `Bully` (blatantly hateful meme).
- **Evaluation Anomaly:** In E1, E3, E5, E6, E7, the target label was `Nonbully`. If CLIP features caused the model to predict `hate`, the model was marked as committing a false positive error!

### 3.2 The 111 Text-Bullying / Meme-Benign Samples (1.90%)
A small subset (1.9%) where harsh slang or aggressive words in the text were judged non-bullying within the humorous or self-deprecating visual context of the meme.

### 3.3 The Concordant Subsets (68.63%)
- **Concordant Benign:** 2,521 samples (43.1%) where both text and image are completely safe.
- **Concordant Bully:** 1,497 samples (25.6%) where the text alone is abusive and the meme image confirms the abuse.

---

## 4. Scientific Resolution & Formal Benchmark Separation

To maintain strict scientific validity and ensure reproducible comparisons:

### Principle 1: Retain Historical E1–E7 as the "Text-Level Hate" Benchmark
- Do not overwrite or modify existing E1, E3, E5, E6, or E7 configurations.
- Clearly document in the manuscript that E1–E7 evaluated **Text-Level Bullying** (`Text-Label`).
- This accurately reports the historical baseline while clarifying why unimodal text models were strong competitors on this specific target.

### Principle 2: Establish the "Multimodal Whole-Meme Hate" Benchmark
- Create dedicated experiment configurations that target `Img-Text-Label`:
  - `exp_memehate_text.yaml`: DistilBERT text-only evaluated against `Img-Text-Label`.
  - `exp_memehate_clip.yaml`: DistilBERT + Frozen CLIP evaluated against `Img-Text-Label`.
- **Hypothesis:** On the whole-meme benchmark, multimodal fusion will demonstrate a statistically significant performance advantage over unimodal text models ($p < 0.001$, group-bootstrap CI), because the visual stream is essential to classify the 1,725 multimodal-dependent instances.

### Proposed Configuration Definition for Meme-Level Hate:
```yaml
tasks:
  hate:
    from: Img-Text-Label
    mapping:
      Bully: hate
      Nonbully: non_hate
  sarcasm:
    from: Sarcasm-Label
    mapping:
      Sarcastic: sarcastic
      Nonsarcastic: non_sarcastic
  target:
    from: Target-Label
    mapping:
      Individual: individual
      Organization: organization
      Community: community
  severity:
    from: Severity-Label
    mapping:
      Low: low
      Medium: medium
      High: high
```

---

## 5. Summary Matrix of Target Consistency

| Benchmark Suite | Target Label Source | What is being evaluated? | Expected Multimodal Advantage |
| :--- | :--- | :--- | :--- |
| **Historical Benchmark (E1, E3, E5, E6, E7, E1b, E6a, E7a)** | `Text-Label` | Toxicity of extracted OCR text in isolation | Low / Neutral (visual features confound text-only label) |
| **Meme-Level Benchmark (`exp_memehate_*`)** | `Img-Text-Label` | Toxicity of full multimodal meme (image + text) | High / Decisive (essential for 29.5% multimodal-dependent cases) |
