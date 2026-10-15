"""Automated unit and regression tests for HOT / MultiBully data integrity audit.

Tests:
1. HOT label mapping rules (Normal -> non_hate, Hate -> hate, Offensive -> ignored / NaN).
2. Ignored categories (needs_review / uncertain -> NaN, never silently converted to negative).
3. Generic build_canonical() fails explicitly on unknown classes and does not discard multi-task rows.
4. Task-wise supervision masking and zero-supervision row filtering.
5. Cross-source duplicate detection and zero-leakage guarantee across all folds.
6. MultiBully Text-Label vs Img-Text-Label definition consistency.
7. Audit report deliverables exist, match computed data, and declare PASS verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from toxifusionnet.data.grouped_splits import normalize_ocr_text
from toxifusionnet.data.schema import build_canonical, load_schema
from toxifusionnet.labels import IGNORE_INDEX, label_to_id
from toxifusionnet.preprocessing.normalize import TextNormalizer

DATA_DIR = Path("data/processed")


def test_hot_label_mapping_and_rules():
    """Verify HOT label mapping: 0.0 -> non_hate, 1.0 -> hate, 2.0 -> NaN (ignored)."""
    hot_gold = pd.read_csv(DATA_DIR / "hot_gold_only.csv")
    assert len(hot_gold) == 3189, "HOT gold-only must retain all 3,189 rows"

    # Category 0.0 (Normal) -> non_hate
    normal_mask = (hot_gold["orig_hot_label"] == 0.0)
    assert int(normal_mask.sum()) == 1121
    assert (hot_gold.loc[normal_mask, "hate"] == "non_hate").all()
    # Canonical label ID must be 0
    normal_ids = [label_to_id("hate", v) for v in hot_gold.loc[normal_mask, "hate"]]
    assert set(normal_ids) == {0}

    # Category 1.0 (Hate Speech) -> hate
    hate_mask = (hot_gold["orig_hot_label"] == 1.0)
    assert int(hate_mask.sum()) == 303
    assert (hot_gold.loc[hate_mask, "hate"] == "hate").all()
    # Canonical label ID must be 1
    hate_ids = [label_to_id("hate", v) for v in hot_gold.loc[hate_mask, "hate"]]
    assert set(hate_ids) == {1}

    # Category 2.0 (Offensive Language) -> NaN (ignored for binary hate supervision)
    off_mask = (hot_gold["orig_hot_label"] == 2.0)
    assert int(off_mask.sum()) == 1765
    assert hot_gold.loc[off_mask, "hate"].isna().all()
    # Canonical label ID must be -100 (IGNORE_INDEX)
    off_ids = [label_to_id("hate", v) for v in hot_gold.loc[off_mask, "hate"]]
    assert set(off_ids) == {IGNORE_INDEX}


def test_ignored_categories_never_converted_to_negative():
    """Verify needs_review and missing pseudo-labels are ignored (NaN), never converted to negative."""
    hot_clean = pd.read_csv(DATA_DIR / "hot_cleaned.csv")
    hot_gold = pd.read_csv(DATA_DIR / "hot_gold_only.csv")

    # In raw HOT dataset, 732 rows have status 'needs_review' or sarcasm 'uncertain'
    needs_review_count = int((hot_clean["annotation_status"] == "needs_review").sum())
    assert needs_review_count == 732

    # In hot_gold_only.csv, all 3,189 rows have sarcasm, target, severity masked to NaN
    for task in ("sarcasm", "target", "severity"):
        assert hot_gold[task].isna().all(), f"All rows in hot_gold_only must have {task} as NaN"
        ids = [label_to_id(task, v) for v in hot_gold[task]]
        assert set(ids) == {IGNORE_INDEX}, f"All {task} IDs in hot_gold_only must be IGNORE_INDEX"


def test_build_canonical_fails_explicitly_on_unknown_classes():
    """Verify build_canonical fails explicitly on unknown classes rather than silently mapping to hate."""
    # Unknown class in hate derivation
    df_unknown = pd.DataFrame({
        "tweet": ["hello world", "test tweet"],
        "label": ["random_unknown_class_xyz", "not_cyberbullying"]
    })
    schema_derive = {
        "text_col": "tweet",
        "derive": {"hate": {"from": "label", "non_hate_values": ["not_cyberbullying"]}}
    }
    with pytest.raises(ValueError, match="Unknown class.*Cannot silently map to hate"):
        build_canonical(df_unknown, schema_derive)

    # Unknown class in direct column
    schema_col = {
        "text_col": "tweet",
        "columns": {"hate": "label"}
    }
    with pytest.raises(ValueError, match="Unknown class.*Cannot silently map to hate"):
        build_canonical(df_unknown, schema_col)


def test_build_canonical_preserves_multitask_rows_when_hate_is_missing():
    """Verify build_canonical does not drop rows merely because hate is missing if other task labels exist."""
    df_multi = pd.DataFrame({
        "text": ["some text 1", "some text 2", "some text 3"],
        "hate": ["hate", np.nan, np.nan],
        "is_sarcastic": ["yes", "no", "yes"],
        "target": ["individual", "gender", np.nan],
    })
    schema = {
        "text_col": "text",
        "columns": {"hate": "hate", "sarcasm": "is_sarcastic", "target": "target"}
    }
    canonical = build_canonical(df_multi, schema, weak_fill=False)
    # Rows 1 and 2 must NOT be dropped, because is_sarcastic and target are present!
    assert len(canonical) == 3
    assert pd.isna(canonical.iloc[1]["hate"])
    assert canonical.iloc[1]["sarcasm"] == "not_sarcastic"
    assert canonical.iloc[1]["target"] == "gender"


def test_task_wise_supervision_and_zero_supervision_auxiliary_filtering():
    """Verify task supervision counts and zero-supervision row filtering in cv.py."""
    mb = pd.read_csv(DATA_DIR / "multibully.csv")
    hot_gold = pd.read_csv(DATA_DIR / "hot_gold_only.csv")

    # MultiBully has supervision for hate (5,718), sarcasm (5,718), target (1,437), severity (1,580)
    assert mb["hate"].notna().sum() == 5718
    assert mb["sarcasm"].notna().sum() == 5718
    assert mb["target"].notna().sum() == 1437
    assert mb["severity"].notna().sum() == 1580

    # Non-hate rows in MultiBully have target and severity conditional-masked to NaN
    non_hate_mask = (mb["hate"] == "non_hate")
    assert int(non_hate_mask.sum()) == 4138
    assert mb.loc[non_hate_mask, "target"].isna().all()
    assert mb.loc[non_hate_mask, "severity"].isna().all()

    # Zero supervision row check in hot_gold_only.csv
    # Rows with label == 2.0 (Offensive) have hate=NaN, sarcasm=NaN, target=NaN, severity=NaN
    all_masked = (
        hot_gold["hate"].isna()
        & hot_gold["sarcasm"].isna()
        & hot_gold["target"].isna()
        & hot_gold["severity"].isna()
    )
    assert int(all_masked.sum()) == 1765, "Exactly 1,765 rows must have zero supervision across all tasks"
    retained_supervised = hot_gold[~all_masked]
    assert len(retained_supervised) == 1424, "Exactly 1,424 rows must have valid supervision"
    assert (retained_supervised["hate"].notna()).all(), "All retained auxiliary rows must have valid hate labels"


def test_cross_source_duplicate_detection_and_zero_leakage():
    """Verify zero text matches and zero leakage between HOT and MultiBully across all folds."""
    mb = pd.read_csv(DATA_DIR / "multibully.csv")
    hot_gold = pd.read_csv(DATA_DIR / "hot_gold_only.csv")
    folds = pd.read_csv(DATA_DIR / "multibully_v2_cv_folds.csv")

    mb_norm = normalize_ocr_text(mb["text"])
    hot_norm = normalize_ocr_text(hot_gold["text"])

    # 1. Exact string matches
    exact_matches = set(mb_norm[mb_norm != ""]).intersection(set(hot_norm[hot_norm != ""]))
    assert len(exact_matches) == 0, f"Found exact cross-dataset text matches: {exact_matches}"

    # 2. Duplicate IDs
    id_overlap = set(mb["id"]).intersection(set(hot_gold["id"]))
    assert len(id_overlap) == 0, f"Found duplicate IDs across datasets: {id_overlap}"

    # 3. Overlap against validation and test sets per fold
    mb_clean = mb.drop(columns=["split"], errors="ignore").merge(folds, on="id")
    hot_set = set(hot_norm[hot_norm != ""])
    for f in range(5):
        val_ids = set(mb_clean[(mb_clean["fold"] == f) & (mb_clean["split"] == "val")]["id"])
        test_ids = set(mb_clean[(mb_clean["fold"] == f) & (mb_clean["split"] == "test")]["id"])

        val_texts = set(mb_norm[mb["id"].isin(val_ids)])
        test_texts = set(mb_norm[mb["id"].isin(test_ids)])

        assert len(val_texts.intersection(hot_set)) == 0, f"Fold {f}: validation overlap detected!"
        assert len(test_texts.intersection(hot_set)) == 0, f"Fold {f}: test overlap detected!"

    # 4. Near-duplicate overlap (Character 3-5 n-gram TF-IDF cosine similarity)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    vec.fit(list(mb_norm) + list(hot_norm))
    X_mb = vec.transform(mb_norm)
    X_hot = vec.transform(hot_norm)
    sim = X_hot.dot(X_mb.T)

    max_sim = float(sim.max())
    assert max_sim < 0.85, f"Pairwise cosine similarity exceeded 0.85 (observed: {max_sim:.4f})"
    assert int((sim >= 0.85).sum()) == 0


def test_multibully_text_label_vs_memehate_label_consistency():
    """Verify MultiBully Text-Label is distinct from Img-Text-Label and reconcile 1,559 vs 1,725."""
    mb_text = pd.read_csv(DATA_DIR / "multibully.csv")
    mb_meme = pd.read_csv(DATA_DIR / "multibully_memehate.csv")
    raw_mb = pd.read_csv(Path("data/raw/multibully.csv"))

    # Text-only hate target: 1,580 hate, 4,138 non-hate (N = 5,718)
    assert mb_text["hate"].value_counts().to_dict() == {"non_hate": 4138, "hate": 1580}

    # Whole-meme hate target: 3,139 hate, 2,579 non-hate (N = 5,718)
    assert mb_meme["hate"].value_counts().to_dict() == {"hate": 3139, "non_hate": 2579}

    # Processed divergence:
    proc_vis_only = (mb_text["hate"] == "non_hate") & (mb_meme["hate"] == "hate")
    proc_rev_disc = (mb_text["hate"] == "hate") & (mb_meme["hate"] == "non_hate")
    assert int(proc_vis_only.sum()) == 1670, "Processed dataset has exactly 1,670 visual-only bullying memes (29.21%)"
    assert int(proc_rev_disc.sum()) == 111, "Processed dataset has exactly 111 text-only hate memes (1.94%)"

    # Net marginal difference: exactly 1,559
    assert (3139 - 1580) == 1559
    assert (1670 - 111) == 1559

    # Raw divergence:
    raw_vis_only = (raw_mb["Text-Label"] == "Nonbully") & (raw_mb["Img-Text-Label"] == "Bully")
    raw_rev_disc = (raw_mb["Text-Label"] == "Bully") & (raw_mb["Img-Text-Label"] == "Nonbully")
    assert int(raw_vis_only.sum()) == 1725, "Raw dataset has exactly 1,725 visual-only bullying memes (29.41%)"
    assert int(raw_rev_disc.sum()) == 111, "Raw dataset has exactly 111 text-only bullying memes (1.89%)"


def test_audit_deliverables_exist_and_verdict_pass():
    """Verify audit deliverables exist in both root and results/ and confirm overall PASS verdict."""
    audit_dir = Path("results/v3_journal_final/data_integrity_audit")
    for base_p in [audit_dir, Path(".")]:
        md_file = base_p / "data_integrity_report.md"
        json_file = base_p / "data_integrity_summary.json"
        csv_file = base_p / "data_integrity_summary.csv"

        assert md_file.exists(), f"{md_file} must exist"
        assert json_file.exists(), f"{json_file} must exist"
        assert csv_file.exists(), f"{csv_file} must exist"

    with open(audit_dir / "data_integrity_summary.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["overall_verdict"] == "PASS"
    assert data["cross_dataset_leakage_audit"]["leakage_verdict"] == "PASS"
    assert data["e6b_pseudolabel_check"]["has_pseudolabel_supervision"] is False
    assert data["label_definition_consistency"]["processed_text_vs_meme_divergence_count"] == 1670
    assert data["label_definition_consistency"]["raw_text_vs_meme_divergence_count"] == 1725
    assert data["label_definition_consistency"]["net_marginal_difference_processed"] == 1559
