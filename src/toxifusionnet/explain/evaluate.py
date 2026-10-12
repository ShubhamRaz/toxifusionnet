"""Rationale evaluation on a labelled dataframe: plausibility (needs gold rationales) + faithfulness (does not)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import faithfulness, plausibility, top_k_mask
from .rationale import aggregate_annotators, to_binary_masks


def evaluate_rationales(pipeline, df: pd.DataFrame, n: int = 200, k: int = 5, seed: int = 0) -> dict:
    d = df[df["hate"] == "hate"].sample(min(n, int((df["hate"] == "hate").sum())), random_state=seed)
    scores, golds, comp, suff = [], [], [], []
    rows = pipeline._forward(d["text"].tolist())
    for (_, row), r in zip(d.iterrows(), rows):
        words = r["words"]
        ws = pipeline._word_scores(r["attn"]["hate"], r["word_ids"], len(words))
        pm = top_k_mask(ws, min(k, len(words)))
        gold_masks = to_binary_masks(words, row.get("hate_rationale"))
        if gold_masks:
            scores.append(ws); golds.append((aggregate_annotators(gold_masks, len(words)) > 0).astype(np.float32))
        p_full = lambda ws_: float(pipeline.hate_probability([" ".join(ws_)])[0])  # noqa: E731
        f = faithfulness(p_full, words, pm)
        comp.append(f["comprehensiveness"]); suff.append(f["sufficiency"])
    out = {"faithfulness": {"comprehensiveness": float(np.mean(comp)), "sufficiency": float(np.mean(suff)), "n": len(comp)}}
    out["plausibility"] = plausibility(scores, golds, k) if golds else "no gold rationales in data"
    return out


def export_human_eval_sheet(pipeline, df: pd.DataFrame, path: str, n: int = 100, seed: int = 0):
    """CSV for human assessment of rationale quality (1-5 Likert on plausibility / helpfulness)."""
    d = df.sample(min(n, len(df)), random_state=seed)
    preds = pipeline.predict(d["text"].tolist(), explain=True)
    rows = [{"id": i, "text": p["text"], "pred_sarcasm": p.get("sarcasm", {}).get("label"), "pred_hate": p["hate"]["label"],
             "pred_target": (p["target"] or {}).get("label"), "pred_severity": (p["severity"] or {}).get("label"),
             "model_rationale_hate": " ".join(t for t, _ in p["rationale"].get("hate_tokens", [])),
             "model_rationale_sarcasm": " ".join(t for t, _ in p["rationale"].get("sarcasm_tokens", [])),
             "rating_plausibility_1to5": "", "rating_helpfulness_1to5": "", "annotator_id": "", "comments": ""}
            for i, p in zip(d["id"], preds)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
