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
    """Verify MultiBully Text-Label is distinct from Img-Text-Label via ID-safe 1:1 join and reconcile 1,559 vs 1,725."""
    mb_text = pd.read_csv(DATA_DIR / "multibully.csv")
    mb_meme = pd.read_csv(DATA_DIR / "multibully_memehate.csv")
    raw_path = Path("data/raw/multibully.csv")
    if not raw_path.exists():
        raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "multibully.csv"
    if not raw_path.exists():
        pytest.skip("data/raw/multibully.csv not present in test environment")
    raw_mb = pd.read_csv(raw_path)

    # Validate ID uniqueness in both processed files
    assert mb_text["id"].is_unique, "MultiBully text dataset must have unique IDs"
    assert mb_meme["id"].is_unique, "MultiBully meme dataset must have unique IDs"

    # ID-safe one-to-one merge on shared identifier 'id'
    merged = pd.merge(
        mb_text[["id", "hate"]],
        mb_meme[["id", "hate"]],
        on="id",
        suffixes=("_text", "_meme"),
        how="inner",
        validate="1:1"
    )
    assert len(merged) == 5718, "One-to-one merge must match exactly 5,718 rows"
    assert len(merged) == len(mb_text) == len(mb_meme), "Zero records may be dropped during join"

    # Marginal hate totals in processed data
    assert (merged["hate_text"] == "hate").sum() == 1580
    assert (merged["hate_text"] == "non_hate").sum() == 4138
    assert (merged["hate_meme"] == "hate").sum() == 3139
    assert (merged["hate_meme"] == "non_hate").sum() == 2579

    # Pairwise discordance computed strictly from joined rows
    proc_vis_only = (merged["hate_text"] == "non_hate") & (merged["hate_meme"] == "hate")
    proc_rev_disc = (merged["hate_text"] == "hate") & (merged["hate_meme"] == "non_hate")
    assert int(proc_vis_only.sum()) == 1670, "Processed dataset has exactly 1,670 visual-only bullying memes (29.21%)"
    assert int(proc_rev_disc.sum()) == 111, "Processed dataset has exactly 111 text-only hate memes (1.94%)"

    # Net marginal difference: exactly 1,559
    assert (3139 - 1580) == 1559
    assert (1670 - 111) == 1559

    # Raw counts computed directly from raw source
    raw_vis_only = (raw_mb["Text-Label"] == "Nonbully") & (raw_mb["Img-Text-Label"] == "Bully")
    raw_rev_disc = (raw_mb["Text-Label"] == "Bully") & (raw_mb["Img-Text-Label"] == "Nonbully")
    assert int(raw_vis_only.sum()) == 1725, "Raw dataset has exactly 1,725 visual-only bullying memes (29.41%)"
    assert int(raw_rev_disc.sum()) == 111, "Raw dataset has exactly 111 text-only bullying memes (1.89%)"


def test_per_fold_sample_counts_from_canonical_cache():
    """Verify per-fold train, val, and test row counts are computed from canonical fold cache without conflation."""
    folds_df = pd.read_csv(DATA_DIR / "multibully_v2_cv_folds.csv")
    mb_df = pd.read_csv(DATA_DIR / "multibully.csv")
    hot_gold = pd.read_csv(DATA_DIR / "hot_gold_only.csv")

    assert len(mb_df) == 5718
    assert len(hot_gold) == 3189

    # Retained supervised HOT auxiliary rows
    sup_mask = pd.Series(False, index=hot_gold.index)
    for t in ["hate", "sarcasm", "target", "severity"]:
        if t in hot_gold.columns:
            sup_mask = sup_mask | hot_gold[t].apply(lambda v: label_to_id(t, v) != IGNORE_INDEX)
    n_hot_supervised = int(sup_mask.sum())
    assert n_hot_supervised == 1424, "Exactly 1,424 supervised HOT rows retained"

    mb_merged = mb_df.drop(columns=["split"], errors="ignore").merge(folds_df, on="id")

    total_test_rows = 0
    for f in range(5):
        f_mb = mb_merged[mb_merged["fold"] == f]
        n_train = len(f_mb[f_mb["split"] == "train"])
        n_val = len(f_mb[f_mb["split"] == "val"])
        n_test = len(f_mb[f_mb["split"] == "test"])

        # Disjoint partition sums
        assert n_train + n_val + n_test == 5718, f"Fold {f} must sum to 5,718 rows"
        total_test_rows += n_test

        # MultiBully train partition range
        assert 3881 <= n_train <= 3889, f"Fold {f} train rows {n_train} outside expected range [3881, 3889]"
        assert 684 <= n_val <= 693, f"Fold {f} val rows {n_val} outside expected range [684, 693]"
        assert 1142 <= n_test <= 1146, f"Fold {f} test rows {n_test} outside expected range [1142, 1146]"

        # E6b actual training rows: MultiBully train + 1,424 HOT
        e6b_train = n_train + n_hot_supervised
        assert 5305 <= e6b_train <= 5313, f"Fold {f} E6b train rows {e6b_train} outside expected range [5305, 5313]"

        # Confirm validation is never labeled as training
        assert n_train != (n_train + n_val), "Train partition must not conflate validation rows"

    assert total_test_rows == 5718, "All 5 outer test folds must sum to exactly 5,718 MultiBully rows"


def test_data_driven_audit_verdict_rejects_failures():
    """Verify compute_audit_verdicts dynamically derives verdicts and rejects deliberate failures."""
    import copy
    import sys
    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_data_integrity_audit import compute_audit_verdicts

    json_path = Path("results/v3_journal_final/data_integrity_audit/data_integrity_summary.json")
    assert json_path.exists(), "Summary JSON must exist"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Real workspace data: raw_hot_dataset is absent on disk -> Section A is WARNING, overall is WARNING
    base_verdict = compute_audit_verdicts(
        provenance_info=data["provenance_audit"],
        e6b_pseudolabel_check=data["e6b_pseudolabel_check"],
        category_audit=data["hot_category_audit"],
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=data["cross_dataset_leakage_audit"],
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=data["per_fold_splits"],
    )
    assert base_verdict["section_verdicts"]["A"] == "WARNING"
    assert base_verdict["overall_verdict"] == "WARNING"
    assert base_verdict["section_verdicts"]["B"] == "PASS"
    assert base_verdict["section_verdicts"]["C"] == "PASS"
    assert base_verdict["section_verdicts"]["D"] == "PASS"
    assert base_verdict["section_verdicts"]["E"] == "PASS"

    # 2. When upstream raw_hot_dataset exists, Section A becomes PASS and overall becomes PASS
    mock_prov = copy.deepcopy(data["provenance_audit"])
    mock_prov["raw_hot_dataset"]["exists"] = True
    v_full = compute_audit_verdicts(
        provenance_info=mock_prov,
        e6b_pseudolabel_check=data["e6b_pseudolabel_check"],
        category_audit=data["hot_category_audit"],
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=data["cross_dataset_leakage_audit"],
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=data["per_fold_splits"],
    )
    assert v_full["section_verdicts"]["A"] == "PASS"
    assert v_full["overall_verdict"] == "PASS"

    # 3. Deliberate failure: missing critical file must produce FAIL
    corrupt_prov = copy.deepcopy(data["provenance_audit"])
    corrupt_prov["processed_multibully"]["exists"] = False
    v_missing_file = compute_audit_verdicts(
        provenance_info=corrupt_prov,
        e6b_pseudolabel_check=data["e6b_pseudolabel_check"],
        category_audit=data["hot_category_audit"],
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=data["cross_dataset_leakage_audit"],
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=data["per_fold_splits"],
    )
    assert v_missing_file["section_verdicts"]["A"] == "FAIL"
    assert v_missing_file["overall_verdict"] == "FAIL", "Missing critical file cannot produce overall PASS or WARNING"

    # 4. Deliberate failure: pseudo-label in E6b must produce FAIL
    corrupt_pseudo = copy.deepcopy(data["e6b_pseudolabel_check"])
    corrupt_pseudo["has_pseudolabel_supervision"] = True
    v_pseudo = compute_audit_verdicts(
        provenance_info=data["provenance_audit"],
        e6b_pseudolabel_check=corrupt_pseudo,
        category_audit=data["hot_category_audit"],
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=data["cross_dataset_leakage_audit"],
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=data["per_fold_splits"],
    )
    assert v_pseudo["section_verdicts"]["A"] == "FAIL"
    assert v_pseudo["overall_verdict"] == "FAIL", "E6b pseudo-label cannot produce overall PASS or WARNING"

    # 5. Deliberate failure: cross-dataset leakage must produce FAIL
    corrupt_leak = copy.deepcopy(data["cross_dataset_leakage_audit"])
    corrupt_leak["exact_ocr_matches"] = 2
    v_leak = compute_audit_verdicts(
        provenance_info=data["provenance_audit"],
        e6b_pseudolabel_check=data["e6b_pseudolabel_check"],
        category_audit=data["hot_category_audit"],
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=corrupt_leak,
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=data["per_fold_splits"],
    )
    assert v_leak["section_verdicts"]["D"] == "FAIL"
    assert v_leak["overall_verdict"] == "FAIL", "Cross-dataset leakage cannot produce overall PASS or WARNING"


def test_category_audit_rejects_corrupted_observed_label_distribution():
    """Verify that compute_audit_verdicts inspects actual gold_only_label_dist values and fails on corruption."""
    import copy
    import sys
    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_data_integrity_audit import compute_audit_verdicts

    json_path = Path("results/v3_journal_final/data_integrity_audit/data_integrity_summary.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Corrupt Normal (code 0.0): preserve total_count = 1121, but corrupt distribution with 1 hate row
    corrupt_cats = copy.deepcopy(data["hot_category_audit"])
    corrupt_cats[0]["gold_only_label_dist"] = {"non_hate": 1120, "hate": 1}
    assert corrupt_cats[0]["total_count"] == 1121, "Row count is preserved"
    v1 = compute_audit_verdicts(
        provenance_info=data["provenance_audit"],
        e6b_pseudolabel_check=data["e6b_pseudolabel_check"],
        category_audit=corrupt_cats,
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=data["cross_dataset_leakage_audit"],
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=data["per_fold_splits"],
    )
    assert v1["section_verdicts"]["B"] == "FAIL"
    assert v1["overall_verdict"] == "FAIL"

    # 2. Corrupt Hate Speech (code 1.0): preserve total_count = 303, but change 1 row to non_hate
    corrupt_cats2 = copy.deepcopy(data["hot_category_audit"])
    corrupt_cats2[1]["gold_only_label_dist"] = {"hate": 302, "non_hate": 1}
    assert corrupt_cats2[1]["total_count"] == 303, "Row count is preserved"
    v2 = compute_audit_verdicts(
        provenance_info=data["provenance_audit"],
        e6b_pseudolabel_check=data["e6b_pseudolabel_check"],
        category_audit=corrupt_cats2,
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=data["cross_dataset_leakage_audit"],
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=data["per_fold_splits"],
    )
    assert v2["section_verdicts"]["B"] == "FAIL"
    assert v2["overall_verdict"] == "FAIL"

    # 3. Corrupt Offensive Language (code 2.0): preserve total_count = 1765, but unmask 1 row to non_hate
    corrupt_cats3 = copy.deepcopy(data["hot_category_audit"])
    corrupt_cats3[2]["gold_only_label_dist"] = {"NaN": 1764, "non_hate": 1}
    assert corrupt_cats3[2]["total_count"] == 1765, "Row count is preserved"
    v3 = compute_audit_verdicts(
        provenance_info=data["provenance_audit"],
        e6b_pseudolabel_check=data["e6b_pseudolabel_check"],
        category_audit=corrupt_cats3,
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=data["cross_dataset_leakage_audit"],
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=data["per_fold_splits"],
    )
    assert v3["section_verdicts"]["B"] == "FAIL"
    assert v3["overall_verdict"] == "FAIL"


def test_fold_integrity_verdict_rejects_overlapping_ids():
    """Verify that compute_audit_verdicts fails when partition counts remain unchanged but IDs overlap."""
    import copy
    import sys
    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_data_integrity_audit import compute_audit_verdicts

    json_path = Path("results/v3_journal_final/data_integrity_audit/data_integrity_summary.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Within-fold train/validation ID overlap with counts unchanged
    corrupt_splits_tv = copy.deepcopy(data["per_fold_splits"])
    corrupt_splits_tv["fold_0"]["train_val_overlap_count"] = 1
    # Note: partition counts are completely unchanged
    assert corrupt_splits_tv["fold_0"]["multibully_train_rows"] == 3889
    assert corrupt_splits_tv["fold_0"]["multibully_val_rows"] == 687
    v_tv = compute_audit_verdicts(
        provenance_info=data["provenance_audit"],
        e6b_pseudolabel_check=data["e6b_pseudolabel_check"],
        category_audit=data["hot_category_audit"],
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=data["cross_dataset_leakage_audit"],
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=corrupt_splits_tv,
    )
    assert v_tv["section_verdicts"]["C"] == "FAIL"
    assert v_tv["overall_verdict"] != "PASS"

    # 2. Cross-fold outer test ID overlap: fold 0 and fold 1 duplicate a test ID, counts unchanged
    corrupt_splits_cross = copy.deepcopy(data["per_fold_splits"])
    corrupt_splits_cross["fold_1"]["test_ids"][0] = corrupt_splits_cross["fold_0"]["test_ids"][0]
    # Counts remain 1142 and 1144
    assert len(corrupt_splits_cross["fold_0"]["test_ids"]) == 1142
    assert len(corrupt_splits_cross["fold_1"]["test_ids"]) == 1144
    v_cross = compute_audit_verdicts(
        provenance_info=data["provenance_audit"],
        e6b_pseudolabel_check=data["e6b_pseudolabel_check"],
        category_audit=data["hot_category_audit"],
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=data["cross_dataset_leakage_audit"],
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=corrupt_splits_cross,
    )
    assert v_cross["section_verdicts"]["C"] == "FAIL"
    assert v_cross["overall_verdict"] != "PASS"

    # 3. Explicit train_ids and val_ids overlapping
    corrupt_splits_explicit = copy.deepcopy(data["per_fold_splits"])
    corrupt_splits_explicit["fold_0"]["train_ids"] = [10, 20, 30]
    corrupt_splits_explicit["fold_0"]["val_ids"] = [30, 40, 50]  # ID 30 is overlapping
    v_explicit = compute_audit_verdicts(
        provenance_info=data["provenance_audit"],
        e6b_pseudolabel_check=data["e6b_pseudolabel_check"],
        category_audit=data["hot_category_audit"],
        task_supervision_rows=data["task_supervision_audit"],
        leakage_summary=data["cross_dataset_leakage_audit"],
        confounding_analysis=data["label_definition_consistency"],
        per_fold_splits=corrupt_splits_explicit,
    )
    assert v_explicit["section_verdicts"]["C"] == "FAIL"
    assert v_explicit["overall_verdict"] != "PASS"


def test_near_duplicate_jaccard_and_char_scope_documentation():
    """Verify near-duplicate documentation records character n-gram full scope and Jaccard candidate screening."""
    json_path = Path("results/v3_journal_final/data_integrity_audit/data_integrity_summary.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    leak = data["cross_dataset_leakage_audit"]

    # Character n-gram scope
    char_scope = leak["char_ngram_full_corpus"]
    assert char_scope["total_pairs_evaluated"] == 18234702
    assert char_scope["char_pairs_ge_85"] == 0
    assert char_scope["char_pairs_ge_80"] == 0
    assert char_scope["max_char_cosine_similarity"] < 0.85

    # Word-set Jaccard candidate scope
    jaccard_scope = leak["word_jaccard_candidate_scope"]
    assert jaccard_scope["multibully_texts_ge3_tokens"] == 5605
    assert jaccard_scope["multibully_texts_lt3_tokens_excluded"] == 113
    assert jaccard_scope["hot_texts_ge3_tokens"] == 3130
    assert jaccard_scope["hot_texts_lt3_tokens_excluded"] == 59
    assert jaccard_scope["candidate_pairs_evaluated"] == 2846571
    assert jaccard_scope["jaccard_pairs_ge_80"] == 0

    # Explicit methodological limitation documented
    assert "methodological_limitation" in leak
    assert "cannot establish semantic equivalence" in leak["methodological_limitation"].lower()


def test_audit_deliverables_exist_and_verdict_warning_on_absent_raw_hot():
    """Verify audit deliverables exist in both root and results/ and confirm WARNING verdict due to absent raw HOT source."""
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

    assert data["overall_verdict"] == "WARNING"
    assert data["section_verdicts"]["A"] == "WARNING"
    assert data["section_verdicts"]["B"] == "PASS"
    assert data["section_verdicts"]["C"] == "PASS"
    assert data["section_verdicts"]["D"] == "PASS"
    assert data["section_verdicts"]["E"] == "PASS"
    assert data["e6b_pseudolabel_check"]["has_pseudolabel_supervision"] is False
    assert data["label_definition_consistency"]["processed_text_vs_meme_divergence_count"] == 1670
    assert data["label_definition_consistency"]["raw_text_vs_meme_divergence_count"] == 1725
    assert data["label_definition_consistency"]["net_marginal_difference_processed"] == 1559

    # Verify per-fold disjointness and test_ids
    for f in range(5):
        fstat = data["per_fold_splits"][f"fold_{f}"]
        assert fstat["train_val_overlap_count"] == 0
        assert fstat["train_test_overlap_count"] == 0
        assert fstat["val_test_overlap_count"] == 0
        assert len(fstat["test_ids"]) in (1142, 1144, 1146)
