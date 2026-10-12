"""Dataset description: label distributions, sarcastic-hate counts per split (statistical power!), language mix."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ..preprocessing.normalize import TextNormalizer

MIN_SUBSET = 30  # a 95% CI for a proportion at n=30 is about +-0.18 wide; below this, subset claims are underpowered


def _dist(s: pd.Series) -> dict:
    vc = s.fillna("(missing)").value_counts()
    return {k: {"n": int(v), "pct": round(100 * v / max(len(s), 1), 1)} for k, v in vc.items()}


def describe_dataset(df: pd.DataFrame) -> dict:
    out: dict = {"n_rows": int(len(df)), "label_source": df["label_source"].iloc[0] if "label_source" in df and len(df) else None}
    out["splits"] = {k: int(v) for k, v in df["split"].value_counts().items()} if "split" in df else {}
    for t in ("sarcasm", "hate", "target", "severity"):
        out[t] = _dist(df[t])
    # joint hate x sarcasm per split -> the sarcastic-hate subset is what the central hypothesis is tested on
    joint = {}
    for sp in sorted(df["split"].unique()) if "split" in df else ["all"]:
        d = df[df.split == sp] if "split" in df else df
        n_sh = int(((d.hate == "hate") & (d.sarcasm == "sarcastic")).sum())
        joint[sp] = {"n": int(len(d)), "hate": int((d.hate == "hate").sum()), "sarcastic": int((d.sarcasm == "sarcastic").sum()),
                     "sarcastic_hate": n_sh, "sarcastic_non_hate": int(((d.hate == "non_hate") & (d.sarcasm == "sarcastic")).sum())}
    out["hate_x_sarcasm_by_split"] = joint
    h = df[df.hate == "hate"]
    out["target_x_severity_among_hate"] = {str(k): v for k, v in pd.crosstab(h.target.fillna("(missing)"), h.severity.fillna("(missing)")).to_dict("index").items()}
    # language mix
    norm = TextNormalizer()
    cmi, n_words, hi_share, dev = [], [], [], []
    for t in df["text"].astype(str):
        r = norm(t)
        cmi.append(r.cmi); n_words.append(len(r.words))
        lang = [g for g in r.lang_tags if g in ("hi", "en")]
        hi_share.append(sum(g == "hi" for g in lang) / max(len(lang), 1))
        dev.append(bool(re.search(r"[\u0900-\u097F]", t)))
    cmi, hi_share, n_words = np.array(cmi), np.array(hi_share), np.array(n_words)
    out["language"] = {
        "mean_cmi": round(float(cmi.mean()), 2), "pct_posts_cmi_ge_10": round(100 * float((cmi >= 10).mean()), 1),
        "pct_posts_with_any_hindi_token": round(100 * float((hi_share > 0).mean()), 1),
        "pct_posts_majority_hindi": round(100 * float((hi_share > 0.5).mean()), 1),
        "pct_posts_with_devanagari": round(100 * float(np.mean(dev)), 1),
        "note": "token-level language ID is heuristic (seed lexicons); treat as an estimate"}
    out["length_words"] = {"mean": round(float(n_words.mean()), 1), "p50": int(np.percentile(n_words, 50)), "p95": int(np.percentile(n_words, 95)), "max": int(n_words.max())}
    warn = []
    for sp in ("val", "test"):
        if sp in joint and joint[sp]["sarcastic_hate"] < MIN_SUBSET:
            warn.append(f"{sp}: only {joint[sp]['sarcastic_hate']} sarcastic-hate posts (< {MIN_SUBSET}); false-negative-rate and "
                        f"paired-bootstrap results on this subset will have very wide confidence intervals.")
    if out["language"]["pct_posts_with_any_hindi_token"] < 50:
        warn.append(f"Only {out['language']['pct_posts_with_any_hindi_token']}% of posts contain a detected Hindi token: the corpus is "
                    f"predominantly English; describe it as 'code-mixed' only for its Hinglish portion.")
    out["warnings"] = warn
    return out


def to_markdown(d: dict) -> str:
    L = [f"# Dataset report\n", f"Rows: **{d['n_rows']}**  |  splits: {d['splits']}  |  label sources: `{d['label_source']}`\n"]
    for t in ("sarcasm", "hate", "target", "severity"):
        L.append(f"**{t}**: " + ", ".join(f"{k} {v['n']} ({v['pct']}%)" for k, v in d[t].items()) + "\n")
    L.append("\n## Hate x sarcasm by split\n\n| split | posts | hate | sarcastic | sarcastic hate | sarcastic non-hate |\n|---|---|---|---|---|---|")
    for sp, v in d["hate_x_sarcasm_by_split"].items():
        L.append(f"| {sp} | {v['n']} | {v['hate']} | {v['sarcastic']} | **{v['sarcastic_hate']}** | {v['sarcastic_non_hate']} |")
    L.append("\n## Language mix (heuristic)\n\n" + "\n".join(f"- {k}: {v}" for k, v in d["language"].items()))
    L.append(f"\n## Length (words)\n\n{d['length_words']}\n")
    if d["warnings"]:
        L.append("## Warnings\n\n" + "\n".join(f"- {w}" for w in d["warnings"]))
    return "\n".join(L)
