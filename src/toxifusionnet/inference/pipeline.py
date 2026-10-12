"""End-to-end inference mirroring the project flow chart:

Input -> Preprocessing (LID, normalisation, features) -> Sarcasm detection -> Hate detection (sarcasm-aware)
      -> if hate: Target classification + Severity classification -> Output (+ token rationales)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config import Config
from ..data.dataset import encode_words
from ..labels import LABEL_SPACES, apply_label_config
from ..models.encoder import rebuild_encoder_from_dir
from ..models.toxifusionnet import ToxiFusionNet
from ..preprocessing.features import cue_vector
from ..preprocessing.normalize import TextNormalizer


class ToxiFusionPipeline:
    def __init__(self, model: ToxiFusionNet, tokenizer, normalizer: TextNormalizer, max_length: int = 128,
                 device="cpu", hate_threshold: float = 0.5, sarcasm_threshold: float = 0.5):
        self.model = model.to(device).eval()
        self.tokenizer, self.normalizer, self.max_length = tokenizer, normalizer, max_length
        self.device = torch.device(device)
        self.hate_threshold, self.sarcasm_threshold = hate_threshold, sarcasm_threshold

    @classmethod
    def from_run(cls, run_dir: str | Path, device: str = "cpu", **kw) -> "ToxiFusionPipeline":
        run_dir = Path(run_dir)
        ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
        cfg = Config.wrap(ckpt["cfg"])
        apply_label_config(cfg.get("labels"))
        encoder, tok = rebuild_encoder_from_dir(run_dir)
        model = ToxiFusionNet(encoder, cfg.model)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, tok, TextNormalizer.from_config(cfg.preprocess), int(cfg.data.max_length), device, **kw)

    # ------------------------------------------------------------------ low level
    @torch.no_grad()
    def _forward(self, texts: list[str], batch_size: int = 64):
        results = []
        for s in range(0, len(texts), batch_size):
            chunk = texts[s:s + batch_size]
            nts = [self.normalizer(t) for t in chunk]
            words = [nt.words if nt.words else ["<empty>"] for nt in nts]
            enc = encode_words(self.tokenizer, words, self.max_length)
            T = max(len(x) for x in enc["input_ids"])
            ids = torch.full((len(chunk), T), self.tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), T), dtype=torch.long)
            for i, x in enumerate(enc["input_ids"]):
                ids[i, :len(x)] = torch.tensor(x); mask[i, :len(x)] = 1
            cue = torch.from_numpy(np.stack([cue_vector(nt) for nt in nts]))
            out = self.model(ids.to(self.device), mask.to(self.device), cue.to(self.device))
            probs = {t: torch.softmax(l.float(), -1).cpu().numpy() for t, l in out["logits"].items()}
            attn = {k: v.float().cpu().numpy() for k, v in out["attn"].items()}
            for i in range(len(chunk)):
                results.append({"nt": nts[i], "words": words[i], "word_ids": enc.word_ids(i),
                                "probs": {t: p[i] for t, p in probs.items()},
                                "attn": {k: a[i, :len(enc["input_ids"][i])] for k, a in attn.items()},
                                "gate": None if out["gate"] is None else float(out["gate"][i].mean())})
        return results

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> dict[str, np.ndarray]:
        r = self._forward(list(texts), batch_size)
        return {t: np.stack([x["probs"][t] for x in r]) for t in r[0]["probs"]} if r else {}

    def hate_probability(self, texts: list[str]) -> np.ndarray:
        return self.predict_proba(texts)["hate"][:, 1]

    @staticmethod
    def _word_scores(attn: np.ndarray, word_ids, n_words: int) -> np.ndarray:
        s = np.zeros(n_words, dtype=np.float32)
        for a, w in zip(attn, word_ids):
            if w is not None and w < n_words:
                s[w] += a
        return s

    # ------------------------------------------------------------------ public API
    def predict(self, texts, explain: bool = True, top_k: int = 5, batch_size: int = 64) -> list[dict]:
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        outs = []
        for r in self._forward(texts, batch_size):
            nt, P = r["nt"], r["probs"]
            res: dict = {
                "text": nt.raw, "normalized_text": nt.text,
                "language": {"tags": [f"{w}/{g}" for w, g in zip(nt.words, nt.lang_tags)],
                             "code_mixing_index": round(nt.cmi, 1), "switches": nt.switches,
                             "dominant": Counter(g for g in nt.lang_tags if g in ("hi", "en")).most_common(1)[0][0] if any(g in ("hi", "en") for g in nt.lang_tags) else "unk"},
            }
            if "sarcasm" in P:
                p = float(P["sarcasm"][1])
                res["sarcasm"] = {"label": LABEL_SPACES["sarcasm"][int(p >= self.sarcasm_threshold)], "prob": round(p, 4)}
            ph = float(P["hate"][1])
            is_hate = ph >= self.hate_threshold
            res["hate"] = {"label": LABEL_SPACES["hate"][int(is_hate)], "prob": round(ph, 4)}
            for t in ("target", "severity"):  # flow chart: only classified when hate speech is detected
                if t in P and is_hate:
                    k = int(P[t].argmax())
                    res[t] = {"label": LABEL_SPACES[t][k], "prob": round(float(P[t][k]), 4),
                              "probs": {c: round(float(v), 4) for c, v in zip(LABEL_SPACES[t], P[t])}}
                else:
                    res[t] = None
            if r["gate"] is not None:
                res["sarcasm_fusion_gate"] = round(r["gate"], 4)
            if explain:
                rat = {}
                for name, key in (("hate_tokens", "hate"), ("sarcasm_tokens", "sarcasm")):
                    if key in r["attn"]:
                        ws = self._word_scores(r["attn"][key], r["word_ids"], len(r["words"]))
                        order = np.argsort(-ws)[:top_k]
                        rat[name] = [[r["words"][i], round(float(ws[i]), 4)] for i in order if ws[i] > 0]
                res["rationale"] = rat
            outs.append(res)
        return outs[0] if single else outs
