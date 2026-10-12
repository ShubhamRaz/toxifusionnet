import json

import pandas as pd
import pytest

from toxifusionnet.baselines.classical import run_baselines
from toxifusionnet.inference.pipeline import ToxiFusionPipeline
from toxifusionnet.robustness.evaluate import evaluate_robustness
from toxifusionnet.robustness.perturbations import PERTURBATIONS


def test_training_artifacts(tiny_run):
    for f in ("model.pt", "metrics.json", "history.csv", "preds_test.csv", "config.yaml", "tokenizer", "encoder_config"):
        assert (tiny_run / f).exists(), f
    m = json.loads((tiny_run / "metrics.json").read_text())
    assert m["test"]["hate"]["n"] > 0 and "sarcasm_subset" in m["test"]


def test_pipeline_output_follows_flow_chart(tiny_run):
    pipe = ToxiFusionPipeline.from_run(tiny_run)
    r = pipe.predict("Wah wah, us dharm ke log to genius hain, ekdum bekaar hain lol 🙄")
    assert {"text", "normalized_text", "language", "sarcasm", "hate", "target", "severity", "rationale"} <= set(r)
    r2 = pipe.predict(["aaj ka match bahut accha tha yaar", "yeh restaurant theek-thaak hai"])
    assert len(r2) == 2
    for x in r2:  # target / severity only exist when hate is detected
        assert (x["hate"]["label"] == "hate") == (x["target"] is not None)


def test_pipeline_batch_equals_single(tiny_run):
    pipe = ToxiFusionPipeline.from_run(tiny_run)
    t = ["bijli ka bill kaafi badhiya hai", "Great job, aurtein to mahaan hain lol"]
    a = pipe.predict_proba(t)["hate"][:, 1]
    b = [pipe.predict_proba([x])["hate"][0, 1] for x in t]
    assert max(abs(a - b)) < 1e-4


def test_robustness_and_perturbations(tiny_run, synth_df):
    import random
    for name, fn in PERTURBATIONS.items():
        assert isinstance(fn("Wah bahut accha hai yaar lol", random.Random(0)), str), name
    pipe = ToxiFusionPipeline.from_run(tiny_run)
    res = evaluate_robustness(pipe, synth_df[synth_df.split == "test"].head(60))
    assert len(res) == len(PERTURBATIONS) + 1 and res.loc[0, "flip_rate"] == 0


def test_baselines(synth_df):
    out = run_baselines(synth_df)
    assert set(out["metrics"]) == {"majority", "keyword", "sentiment", "tfidf_lr"}
    assert "target_macro_f1" in out["metrics"]["tfidf_lr"] and "sarcasm_macro_f1" in out["metrics"]["tfidf_lr"]
    assert out["metrics"]["tfidf_lr"]["hate_macro_f1"] > 0.5


def test_api(tiny_run, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("TOXIFUSION_RUN_DIR", str(tiny_run))
    from toxifusionnet.app import api
    api.get_pipeline.cache_clear()
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/predict", json={"texts": ["tum log bekaar ho lol"], "explain": False})
    assert r.status_code == 200 and "hate" in r.json()["results"][0]
    assert c.post("/predict", json={"texts": []}).status_code == 422


def test_from_pretrained_directory_path(tmp_path, synth_df):
    """The real-encoder route (AutoTokenizer/AutoModel.from_pretrained) works with a locally saved checkpoint."""
    from toxifusionnet.config import Config, load_config
    from toxifusionnet.models.encoder import build_encoder_and_tokenizer
    from toxifusionnet.preprocessing.normalize import TextNormalizer
    from toxifusionnet.training.trainer import train_run
    norm = TextNormalizer()
    enc, tok = build_encoder_and_tokenizer(Config.wrap({"encoder_name": "tiny", "tiny_hidden": 32, "tiny_layers": 1}),
                                           [norm(t).words for t in synth_df.text])
    enc.save_pretrained(tmp_path / "ckpt"); tok.save_pretrained(tmp_path / "ckpt")
    cfg = load_config("configs/smoke_test.yaml", [f"model.encoder_name={tmp_path / 'ckpt'}", "train.epochs=1"])
    res = train_run(cfg, synth_df, tmp_path / "run")
    assert res["test"]["hate"]["n"] > 0
    assert ToxiFusionPipeline.from_run(tmp_path / "run").predict("tum log bekaar ho")["hate"]["label"] in ("hate", "non_hate")


def test_config_extends_chain(tmp_path):
    from toxifusionnet.config import load_config
    (tmp_path / "default.yaml").write_text("a: 1\nb: {x: 1, y: 2}\n")
    (tmp_path / "mid.yaml").write_text("b: {y: 20}\nc: 3\n")
    (tmp_path / "top.yaml").write_text("extends: mid.yaml\nc: 30\n")
    cfg = load_config(tmp_path / "top.yaml", ["b.x=9"])
    assert cfg.a == 1 and cfg.b.y == 20 and cfg.b.x == 9 and cfg.c == 30
    real = load_config("configs/multibully_cpu.yaml")
    assert real.model.encoder_name == "tiny" and real.data.processed_path.endswith("multibully.csv") and list(real.labels.target)[-1] == "society"
