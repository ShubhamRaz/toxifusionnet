"""Text perturbations for robustness testing (flow of the evaluation plan: *robustness to fake narratives*, code-mixing
noise, transliteration variation). Each perturbation takes ``(text, rng)`` and returns the perturbed text."""
from __future__ import annotations

import random
import re

VARIANTS = {"kya": "kia", "nahi": "nhi", "bahut": "bohot", "accha": "acha", "hai": "h", "hain": "hn", "log": "logg",
            "yaar": "yar", "kyun": "kyu", "mujhe": "muje", "tumhe": "tmhe", "ekdum": "ekdm", "bilkul": "bilkl"}
HI2EN = {"bahut": "very", "accha": "good", "log": "people", "dost": "friend", "kaam": "work", "bekaar": "useless",
         "sab": "all", "kal": "tomorrow", "aaj": "today", "ghar": "home"}
NARRATIVE_FRAMES = ["WhatsApp forward: sach hai, share karo!", "Breaking: media nahi dikhayega ye sach.",
                    "Forwarded as received, pata nahi sach hai ya nahi but", "Ye khabar sach hai, zyada se zyada share karo.",
                    "Fake news channels ye nahi batayenge:", "Insider source ke mutabik,"]
NEUTRAL_EMOJI = ["🙂", "👍", "😂", "🙏", "🔥"]
SARC_EMOJI = ["🙄", "😏", "🤡"]


def spelling_noise(text: str, rng: random.Random, p: float = 0.25) -> str:
    """Texting-style vowel dropping on long words."""
    def f(m):
        w = m.group(0)
        return re.sub(r"(?<=[a-z])[aeiou](?=[a-z])", "", w, count=1) if len(w) > 4 and rng.random() < p else w
    return re.sub(r"[A-Za-z]+", f, text)


def elongation(text: str, rng: random.Random, p: float = 0.2) -> str:
    def f(m):
        w = m.group(0)
        if len(w) > 2 and rng.random() < p:
            vs = [i for i, c in enumerate(w) if c.lower() in "aeiou"]
            if vs:
                i = rng.choice(vs)
                return w[:i] + w[i] * rng.randint(2, 4) + w[i + 1:]
        return w
    return re.sub(r"[A-Za-z]+", f, text)


def transliteration_variants(text: str, rng: random.Random, p: float = 0.7) -> str:
    return re.sub(r"[A-Za-z]+", lambda m: VARIANTS[m.group(0).lower()] if m.group(0).lower() in VARIANTS and rng.random() < p else m.group(0), text)


def code_switch_shift(text: str, rng: random.Random, p: float = 0.6) -> str:
    return re.sub(r"[A-Za-z]+", lambda m: HI2EN[m.group(0).lower()] if m.group(0).lower() in HI2EN and rng.random() < p else m.group(0), text)


def neutral_emoji_noise(text: str, rng: random.Random) -> str:
    return f"{text} {rng.choice(NEUTRAL_EMOJI)}"


def sarcasm_cue_injection(text: str, rng: random.Random) -> str:
    """Adds sarcasm-looking cues to a post: tests over-reliance on surface cues (hurts precision if model keys on them)."""
    return f"{text} {rng.choice(SARC_EMOJI)} lol"


def narrative_framing(text: str, rng: random.Random) -> str:
    """Wraps a post in misinformation-style framing (forwarded-message / conspiracy cues) without changing its meaning."""
    f = rng.choice(NARRATIVE_FRAMES)
    return f"{f} {text}" if rng.random() < 0.7 else f"{text} {f}"


def case_noise(text: str, rng: random.Random, p: float = 0.3) -> str:
    return "".join(c.upper() if (c.isalpha() and rng.random() < p) else c for c in text)


PERTURBATIONS = {
    "spelling_noise": spelling_noise, "elongation": elongation, "transliteration_variants": transliteration_variants,
    "code_switch_shift": code_switch_shift, "neutral_emoji": neutral_emoji_noise,
    "sarcasm_cue_injection": sarcasm_cue_injection, "narrative_framing": narrative_framing, "case_noise": case_noise,
}
