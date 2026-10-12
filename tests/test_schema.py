import numpy as np
import pandas as pd

from toxifusionnet.data.schema import build_canonical, guess_schema
from toxifusionnet.data.splits import make_splits


def raw():
    return pd.DataFrame({
        "tweet": ["a b c", "tum log bekaar ho", "yeh sab theek hai", "wah kya baat hai lol", "religion wale ghatiya", "x y z"] * 10,
        "label": ["not_cyberbullying", "gender", "not_cyberbullying", "ethnicity", "religion", "other_cyberbullying"] * 10})


SCHEMA = {"text_col": "tweet", "derive": {
    "hate": {"from": "label", "non_hate_values": ["not_cyberbullying"]},
    "target": {"from": "label", "map": {"gender": "gender", "ethnicity": "demography", "religion": "religion", "other_cyberbullying": "individual"}},
    "sarcasm": {"weak": True}, "severity": {"weak": True}}}


def test_derives_hate_and_target_and_masks_non_hate():
    df = build_canonical(raw(), SCHEMA)
    assert set(df.hate.unique()) == {"hate", "non_hate"}
    assert df.loc[df.hate == "non_hate", "target"].isna().all()
    assert set(df.loc[df.hate == "hate", "target"]) == {"gender", "demography", "religion", "individual"}


def test_weak_labels_are_flagged_and_severity_only_for_hate():
    df = build_canonical(raw(), SCHEMA)
    assert '"sarcasm": "weak"' in df.label_source.iloc[0]
    assert df.loc[df.hate == "non_hate", "severity"].isna().all()
    assert df.loc[df.hate == "hate", "severity"].notna().all()
    assert set(df.sarcasm.unique()) <= {"sarcastic", "not_sarcastic"}


def test_no_weak_leaves_columns_empty():
    df = build_canonical(raw(), SCHEMA, weak_fill=False)
    assert df.sarcasm.isna().all() and df.severity.isna().all()


def test_gold_columns_with_value_maps():
    r = pd.DataFrame({"text": ["a", "b", "c", "d"], "is_sarcastic": ["yes", "no", "Yes", "n"], "hate": [1, 0, 1, 0]})
    df = build_canonical(r, {"text_col": "text", "columns": {"sarcasm": "is_sarcastic", "hate": "hate"}})
    assert df.sarcasm.tolist() == ["sarcastic", "not_sarcastic", "sarcastic", "not_sarcastic"]
    assert df.hate.tolist() == ["hate", "non_hate", "hate", "non_hate"]


def test_guess_schema_and_splits():
    sch = guess_schema(raw())
    assert sch["text_col"] == "tweet" and "hate" in sch["derive"]
    df = make_splits(build_canonical(raw(), SCHEMA), seed=1)
    assert set(df.split) == {"train", "val", "test"} and (df.split == "train").mean() > 0.7
