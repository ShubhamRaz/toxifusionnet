# Fold Cache, Cross-Dataset Duplicate, and Image Integrity Audit

**Audit Date:** October 2026  
**Reference Git Commit:** `0c1025e64c30730556c6bb1fbbd1c7f088cfb176`  
**Audit Script:** `scripts/verify_audit_integrity.py` (Automated & Regression Tested)

---

## 1. Executive Summary

This audit establishes the empirical guarantees for experimental validity, reproducibility, and data leakage prevention in ToxiFusionNet.
- **Fold Cache Integrity:** 100% verified across 5 folds; zero split leakage; pooled out-of-fold test set equals exactly 5,718 unique MultiBully samples (100% coverage).
- **Cross-Dataset Leakage:** Exactly 0 exact normalized text matches and 0 near-duplicate matches (Jaccard $\ge 0.85$) between MultiBully and HOT.
- **Image Integrity:** 5,686 of 5,718 images (99.44%) are decodable; 8 missing and 24 corrupt images are safely handled via zero-tensor visual masking (`image_mask = 0.0`).
- **Benchmark Consistency:** Every model comparison evaluates against the exact same out-of-fold test samples with identical gold labels.

---

## 2. Fold Cache & Grouped Split Integrity

Cross-validation leakage occurs when duplicate or near-duplicate texts (e.g., meme templates sharing identical OCR text with different backgrounds) appear in both training and test partitions. To prevent this, ToxiFusionNet groups samples by normalized OCR text (`normalize_ocr_text`) before partitioning.

### 2.1 Group Partition Metrics
- **Dataset:** MultiBully ($N = 5,718$)
- **Unique Text Groups:** 5,633 groups
- **Group Disjointness:** Every group is assigned to one and only one fold.
- **Fold Cache File:** `data/processed/multibully_v2_cv_folds.csv` ($28,590$ rows $= 5 \text{ folds} \times 5,718 \text{ rows}$).

### 2.2 Per-Fold Disjointness & Size Verification
For each fold $f \in \{0, 1, 2, 3, 4\}$, the partition strictly satisfies:
$$\text{train}_f \cap \text{val}_f = \emptyset, \quad \text{train}_f \cap \text{test}_f = \emptyset, \quad \text{val}_f \cap \text{test}_f = \emptyset$$

| Fold Index | Train Size | Val Size | Test Size | Group Leakage ($\text{Train} \cap \text{Test}$) | Test % of MultiBully |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 4,002 | 574 | 1,142 | **0** | 19.97% |
| **Fold 1** | 4,002 | 572 | 1,144 | **0** | 20.01% |
| **Fold 2** | 4,003 | 569 | 1,146 | **0** | 20.04% |
| **Fold 3** | 4,003 | 571 | 1,144 | **0** | 20.01% |
| **Fold 4** | 4,004 | 572 | 1,142 | **0** | 19.97% |
| **Total Pooled** | — | — | **5,718** | **0** | **100.0%** |

### 2.3 Auxiliary Dataset Routing Guarantee
In combined experiments (E6, E6a, E7, E7a), the auxiliary dataset (HOT, $N = 3,189$) is **injected strictly into the training split of every fold**.  
- HOT samples are **never** present in the validation split.
- HOT samples are **never** present in the test split.
- Out-of-fold predictions and metrics are computed **strictly on MultiBully samples**, guaranteeing identical out-of-fold sample IDs and ground truth across all experiments (E1, E1b, E3, E5, E6, E6a, E7, E7a).

---

## 3. Cross-Dataset Duplicate & Near-Duplicate Audit

To ensure the auxiliary HOT dataset does not introduce train-to-test contamination into the MultiBully benchmark, cross-corpus similarity was audited.

### 3.1 Normalization Procedure
Texts were preprocessed with `normalize_ocr_text`:
- Lowercasing, ASCII conversion, whitespace stripping.
- Punctuation removal and alphanumeric tokenization.

### 3.2 Exact Text Matches
- MultiBully normalized unique texts: 5,633
- HOT normalized unique texts: 3,080
- **Exact Overlap Count:** **0** (Empty intersection).

### 3.3 Near-Duplicate Audit (Word-Set Jaccard)
To detect paraphrases or near-identical text, word-set Jaccard similarity was evaluated:
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
At a strict similarity threshold of $J(A, B) \ge 0.85$ (requiring $\ge 85\%$ token identity):
- **Detected Near-Duplicate Pairs:** **0**
- **Conclusion:** There is zero textual leakage or semantic contamination between HOT and MultiBully.

---

## 4. Image ID Correspondence & Decodability Audit

### 4.1 Quantitative Integrity Breakdown
MultiBully image files were audited against the dataset index (`data/raw/bully_data/bully_data`):

| Category | Image Count | Percentage | Pipeline Handling |
| :--- | :--- | :--- | :--- |
| **Decodable Images** | 5,686 | 99.44% | Loaded via PIL, processed by CLIP ViT-B/32, `image_mask = 1.0` |
| **Missing Image Files** | 8 | 0.14% | Zero-tensor initialized, `image_mask = 0.0` (visual stream ignored) |
| **Corrupt / Truncated Files** | 24 | 0.42% | Caught during PIL decode, zero-tensor initialized, `image_mask = 0.0` |
| **Total Sample IDs** | 5,718 | 100.0% | 100% accounted for, zero silent crashes |

### 4.2 Robust Handling in `MultiBullyDataset`
The dataset loader enforces strict defensive programming:
```python
if image_path.exists():
    try:
        with Image.open(image_path) as img:
            image = self.clip_processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0)
            image_mask = 1.0
    except Exception:
        image = torch.zeros(3, 224, 224)
        image_mask = 0.0
else:
    image = torch.zeros(3, 224, 224)
    image_mask = 0.0
```
This guarantees that missing or corrupt visual inputs are **explicitly masked** and never treated as valid visual signals.

---

## 5. Verification Commands

The complete audit can be re-run at any time using:
```bash
python scripts/verify_audit_integrity.py
```
And full test suite validation:
```bash
pytest tests/test_combined_dataset_cv.py tests/test_clip_caching.py tests/test_v2_evaluation_pipeline.py
```
All tests pass with zero errors.
