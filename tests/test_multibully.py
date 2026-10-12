"""The real-data path: MultiBully column layout, label mapping, 4-class target taxonomy."""
import numpy as np
import pandas as pd
import pytest

from toxifusionnet.analysis.describe import describe_dataset
from toxifusionnet.config import load_config
from toxifusionnet.data.schema import build_canonical, load_schema, read_table, resolve_column
from toxifusionnet.data.splits import make_splits
from toxifusionnet.data.synthetic import make_multibully_layout
from toxifusionnet.inference.pipeline import ToxiFusionPipeline
from toxifusionnet.labels import LABEL_SPACES, label_to_id, num_classes
from toxifusionnet.training.trainer import train_run

SCHEMA = "configs/schema_multibully.yaml"


@pytest.fixture(scope="module")
def raw():
    return make_multibully_layout(400, seed=5)


def test_real_layout_is_mapped_with_gold_sources(raw):
    df = build_canonical(raw, load_schema(SCHEMA))
    assert set(df.hate.dropna()) == {"hate", "non_hate"}
    src = df.label_source.iloc[0]
    for t in ("sarcasm", "target", "severity"):
        assert f'"{t}": "gold"' in src
    assert '"hate": "gold-derived"' in src
    assert num_classes("target") == 4 and LABEL_SPACES["target"] == ["individual", "organization", "community", "society"]


def test_hate_comes_from_text_label_not_meme_label(raw):
    df = build_canonical(raw, load_schema(SCHEMA))
    expected = np.where(raw["Text-Label"].values[: len(df)] == "Bully", "hate", "non_hate")
    assert (df.hate.values == expected).all()
    assert (raw["Img-Text-Label"] == "Bully").mean() > (raw["Text-Label"] == "Bully").mean()   # meme label differs


def test_severity_and_target_mapping_and_hate_conditioning(raw):
    df = build_canonical(raw, load_schema(SCHEMA))
    h = df[df.hate == "hate"]
    assert set(h.severity.dropna()) <= {"low", "moderate", "high"} and h.severity.notna().all()
    assert df.loc[df.hate == "non_hate", ["target", "severity"]].isna().all().all()
    r = raw.set_index("Img-Name")
    row = h.iloc[0]
    assert {"Harmless": "low", "Partially-Harmful": "moderate", "Very-Harmful": "high"}[r.loc[row.id, "Harmful-Score"]] == row.severity


def test_column_names_are_matched_loosely(raw):
    renamed = raw.rename(columns={"Img-Text": "img_text", "Text-Label": "TEXT label", "Harmful-Score": "harmful score"})
    assert resolve_column(renamed, "Img-Text") == "img_text"
    df = build_canonical(renamed, load_schema(SCHEMA))
    assert df.hate.notna().all() and df.severity.notna().any()


def test_targets_outside_active_taxonomy_are_dropped():
    sch = {"text_col": "t", "columns": {"target": "tg"}, "derive": {"hate": {"from": "h", "non_hate_values": ["no"]}}}
    df = build_canonical(pd.DataFrame({"t": ["a", "b"], "tg": ["Organization", "religion"], "h": ["yes", "yes"]}), sch)
    assert pd.isna(df.target.iloc[0])   # 'organization' does not exist in the 5-class default taxonomy
    assert df.target.iloc[1] == "religion"


def test_exclude_doubtful_rows(raw):
    r = raw.copy()
    r.loc[r.index[:10], "Doubtful"] = "yes"
    sch = load_schema(SCHEMA)
    sch["exclude"] = {"column": "Doubtful", "when_not_empty": True}
    assert len(build_canonical(r, sch)) == len(r) - 10


def test_xlsx_roundtrip(raw, tmp_path):
    raw.to_excel(tmp_path / "mb.xlsx", index=False)
    df = build_canonical(read_table(tmp_path / "mb.xlsx"), load_schema(SCHEMA))
    assert len(df) == len(raw) and df.sarcasm.notna().all()


def test_describe_reports_sarcastic_hate_and_warns(raw):
    df = make_splits(build_canonical(raw, load_schema(SCHEMA)))
    d = describe_dataset(df)
    assert d["hate_x_sarcasm_by_split"]["test"]["sarcastic_hate"] >= 0
    assert any("sarcastic-hate" in w for w in d["warnings"])      # small test split -> power warning


def test_train_and_predict_with_four_class_target(raw, tmp_path):
    df = make_splits(build_canonical(raw, load_schema(SCHEMA)))
    cfg = load_config("configs/smoke_multibully.yaml", ["train.epochs=2"])
    res = train_run(cfg, df, tmp_path / "run")
    assert len(res["test"]["target"]["classes"]) == 4
    pipe = ToxiFusionPipeline.from_run(tmp_path / "run")
    out = pipe.predict("Wah wah, us jaati ke log to genius hain, ekdum bekaar hain lol")
    if out["hate"]["label"] == "hate":
        assert set(out["target"]["probs"]) == {"individual", "organization", "community", "society"}
    assert label_to_id("target", "society") == 3
