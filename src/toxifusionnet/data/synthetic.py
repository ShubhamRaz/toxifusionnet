"""Synthetic Hinglish data generator -- ONLY for unit tests / smoke tests of the pipeline.

The generated sentences are template-based, deliberately mild, and carry labels by construction. Metrics on this
data say nothing about real-world performance. Rows are flagged ``synthetic=True`` in the output.
"""
from __future__ import annotations

import json
import random

import pandas as pd

TOPICS = ["aaj ka match", "yeh movie", "metro ki service", "college ka canteen", "traffic", "exam ka result",
          "naya phone", "office ka wifi", "bijli ka bill", "weekend ka plan", "yeh restaurant", "bus ka time"]
NEUTRAL_PRED = ["bahut accha tha yaar", "kaafi badhiya hai", "theek-thaak hai", "dekhne layak hai", "mast tha honestly",
                "is really good", "was quite fine tbh", "ke baare mein kal baat karte hain"]
BAD_SITUATION = ["bilkul bekaar chal raha hai", "phir se late ho gaya", "kuch kaam nahi kar raha", "ne sab kharab kar diya",
                 "again stuck, nothing works"]
SARC_PRAISE = ["Wah wah", "Kya baat hai", "Great job", "Bahut khoob", "Oh really", "Genius", "Shukriya"]
SARC_TAIL = ["lol", "😂", "🙄", "😏", "#sarcasm", "haha", ""]
TARGETS = {
    "individual": ["yeh banda", "tum", "tera dost", "us insaan"],
    "caste": ["us jaati ke log", "unki jaati wale", "woh jaati wale"],
    "religion": ["us dharm ke log", "unke mazhab wale", "woh dharm wale"],
    "gender": ["ladkiyan", "aurtein", "mard log"],
    "demography": ["bahar ke log", "gaon wale", "shehar wale", "dusre state ke log"],
}
INSULTS = ["bekaar", "ghatiya", "nikamme", "bewakoof", "kamzor", "faltu"]
SEVERITY_CUES = {  # mild placeholder cues so that severity is learnable
    "low": ["thoda ajeeb hain", "kabhi kabhi irritate karte hain"],
    "moderate": ["kaafi {insult} hain", "inse door raho"],
    "high": ["ekdum {insult} hain, inhe yahan se nikaal do", "sabse {insult}, bilkul bardasht nahi"],
}
NARRATIVE_PREFIX = ["WhatsApp forward: sach hai!", "Breaking: media nahi dikhayega.", "Forwarded as received:", "Ye sach hai, share karo."]
FILLERS = ["yaar", "bhai", "honestly", "seriously", "tbh", ""]
DEV = {"kya": "क्या", "hai": "है", "nahi": "नहीं", "bahut": "बहुत"}


def _noise(s: str, rng: random.Random) -> str:
    out = []
    for w in s.split():
        r = rng.random()
        lw = w.lower()
        if lw in DEV and r < 0.08:
            w = DEV[lw]
        elif lw == "nahi" and r < 0.2:
            w = "nhi"
        elif lw.isalpha() and len(lw) > 3 and r < 0.04:
            w = w + w[-1] * 2
        out.append(w)
    return " ".join(out)


def make_synthetic(n: int = 3000, seed: int = 0, hate_rate: float = 0.45, sarcasm_rate: float = 0.4,
                   narrative_rate: float = 0.2) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        hate = rng.random() < hate_rate
        sarc = rng.random() < sarcasm_rate
        narrative = rng.random() < narrative_rate
        topic = rng.choice(TOPICS)
        target = sev = None
        sarc_rat: list[str] = []
        hate_rat: list[str] = []
        if hate:
            target = rng.choice(list(TARGETS))
            phrase = rng.choice(TARGETS[target])
            sev = rng.choice(["low", "moderate", "high"])
            insult = rng.choice(INSULTS)
            cue = rng.choice(SEVERITY_CUES[sev]).format(insult=insult)
            if sarc:
                praise = rng.choice(SARC_PRAISE)
                tail = rng.choice([t for t in SARC_TAIL if t])
                # sarcastic hate: positive-sounding surface, insult only sometimes explicit
                core = f"{praise}, {phrase} to {rng.choice(['mahaan', 'genius', 'kitne samajhdar'])} hain, {cue} {tail}"
                sarc_rat = praise.lower().split() + [tail.lower()]
                hate_rat = phrase.split() + cue.split()
            else:
                core = f"{phrase} {cue}"
                hate_rat = phrase.split() + cue.split()
        else:
            if sarc:
                praise = rng.choice(SARC_PRAISE)
                tail = rng.choice([t for t in SARC_TAIL if t])
                core = f"{praise}, {topic} {rng.choice(BAD_SITUATION)} {tail}"
                sarc_rat = praise.lower().split() + [tail.lower()]
            else:
                core = f"{topic} {rng.choice(NEUTRAL_PRED)}"
        filler = rng.choice(FILLERS)
        text = f"{core} {filler}".strip()
        if narrative:
            text = f"{rng.choice(NARRATIVE_PREFIX)} {text}"
        text = _noise(text, rng)
        rows.append({
            "id": f"syn{i}", "text": text,
            "sarcasm": "sarcastic" if sarc else "not_sarcastic",
            "hate": "hate" if hate else "non_hate",
            "target": target, "severity": sev,
            "sarcasm_rationale": json.dumps([w for w in sarc_rat if w]) if sarc else None,
            "hate_rationale": json.dumps(hate_rat) if hate else None,
            "narrative": int(narrative), "label_source": json.dumps({"all": "synthetic"}), "synthetic": True,
        })
    return pd.DataFrame(rows)


def make_multibully_layout(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC rows in the exact column layout / value vocabulary of the MultiBully annotation sheet, so that
    ``configs/schema_multibully.yaml`` and the whole real-data code path can be tested offline. Not real data."""
    base = make_synthetic(n, seed)
    rng = random.Random(seed + 1)
    tgt = {"individual": "Individual", "caste": "Community", "religion": "Community", "gender": "Society",
           "demography": "Society"}
    sev = {"low": "Harmless", "moderate": "Partially-Harmful", "high": "Very-Harmful"}
    rows = []
    for r in base.itertuples():
        hate = r.hate == "hate"
        text_label = "Bully" if hate else "Nonbully"
        img_text = "Bully" if (hate or rng.random() < 0.25) else "Nonbully"      # meme-level label is noisier
        rows.append({
            "Img-Name": f"{r.Index}.jpg", "Img-Text": r.text, "Img-Text-Label": img_text,
            "Img-Label": rng.choice(["Bully", "Nonbully"]), "Text-Label": text_label,
            "Sentiment": rng.choice(["Negative", "Neutral", "Positive"]),
            "Emotion": rng.choice(["Disgust", "Happiness", "Sadness", "Angry", "Other"]),
            "Sarcasm": "Yes" if r.sarcasm == "sarcastic" else "No",
            "Harmful-Score": sev[r.severity] if hate else rng.choice(["Harmless", "Harmless", "Partially-Harmful"]),
            "Target": tgt[r.target] if hate else "None", "Unnamed: 10": None,
            "Doubtful": None})
    return pd.DataFrame(rows)
