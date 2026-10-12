"""Text normalisation for noisy Hindi-English code-mixed social-media text (flow chart: *Preprocessing*)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .language_id import TOKEN_RE, analyze, identify_languages
from .lexicons import (LAUGH_EMOJI, LAUGH_RE, NEG_EMOJI, POS_EMOJI, SARC_EMOJI, SARCASM_MARKERS)
from .transliteration import transliterate_devanagari

try:  # optional dependency, graceful fallback
    import emoji as _emoji
except Exception:  # pragma: no cover
    _emoji = None

_FALLBACK_EMOJI = {"😂": "laughing", "🤣": "laughing", "😒": "unamused", "🙄": "rolling eyes", "😏": "smirking",
                   "👏": "clapping", "🤡": "clown", "😡": "angry", "🤬": "cursing", "👍": "thumbs up",
                   "👎": "thumbs down", "😍": "heart eyes", "😊": "smiling", "🙏": "folded hands", "💩": "poop"}

# Romanised-Hindi spelling variants -> canonical form
SPELLING = {
    "kyaa": "kya", "kia": "kya", "kiya": "kiya", "nhi": "nahi", "nahin": "nahi", "nhin": "nahi", "nai": "nahi",
    "bohot": "bahut", "bahot": "bahut", "bhot": "bahut", "bohut": "bahut", "acha": "accha", "achha": "accha",
    "achchha": "accha", "kyu": "kyun", "kyon": "kyun", "tm": "tum", "yar": "yaar", "yaaar": "yaar", "mujhe": "mujhe",
    "muje": "mujhe", "tumhein": "tumhe", "hein": "hain", "hain": "hain", "bhi": "bhi", "bi": "bhi", "wo": "woh",
    "vo": "woh", "jyada": "zyada", "bekar": "bekaar", "wahh": "wah", "waah": "wah", "waahh": "wah",
}
# English chat abbreviations -> expansion (profanity abbreviations such as "bc"/"mc" are deliberately NOT expanded)
SLANG = {
    "u": "you", "ur": "your", "r": "are", "plz": "please", "pls": "please", "thx": "thanks", "tnx": "thanks",
    "omg": "oh my god", "idk": "i do not know", "btw": "by the way", "tbh": "to be honest", "imo": "in my opinion",
    "gr8": "great", "b4": "before", "luv": "love", "bro": "bro", "ppl": "people", "govt": "government",
    "msg": "message", "dont": "do not", "cant": "can not", "wont": "will not",
}
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_WORD_TOKEN_RE = re.compile(r"<url>|<user>|emoji_[a-z0-9_]+|" + TOKEN_RE.pattern, re.UNICODE)
_SARC_HASHTAGS = {"sarcasm", "sarcastic", "not", "irony", "ironic", "jk", "justkidding"}


def _demojize(text: str) -> str:
    if _emoji is not None:
        # one token per emoji, e.g. "emoji_face_with_rolling_eyes" (language-neutral for LID / CMI statistics)
        return _emoji.demojize(text, delimiters=(" emoji_", " ")).replace("emoji_ ", "emoji_")
    return "".join(f" emoji_{_FALLBACK_EMOJI[c].replace(' ', '_')} " if c in _FALLBACK_EMOJI else c for c in text)


def _split_hashtag(tag: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", tag).replace("_", " ")


def raw_cue_stats(text: str) -> dict:
    """Surface/pragmatic cues counted on the *raw* text (before they are destroyed by normalisation)."""
    letters = [c for c in text if c.isalpha()]
    upper_words = [w for w in re.findall(r"\b[A-Z]{3,}\b", text)]
    lowered = text.lower()
    tags = {t.lower() for t in _HASHTAG_RE.findall(text)}
    return {
        "n_emoji_pos": sum(c in POS_EMOJI for c in text),
        "n_emoji_neg": sum(c in NEG_EMOJI for c in text),
        "n_emoji_laugh": sum(c in LAUGH_EMOJI for c in text),
        "n_emoji_sarc": sum(c in SARC_EMOJI for c in text),
        "n_laugh": len(re.findall(LAUGH_RE, lowered)),
        "n_exclaim": text.count("!"),
        "n_question": text.count("?"),
        "has_ellipsis": int("..." in text or "…" in text),
        "n_quotes": text.count('"') // 2 + text.count("“"),
        "caps_ratio": (sum(len(w) for w in upper_words) / max(len(letters), 1)),
        "hashtag_sarcasm": int(bool(tags & _SARC_HASHTAGS)),
        "n_markers": sum(m in lowered for m in SARCASM_MARKERS),
    }


@dataclass
class NormalizedText:
    raw: str
    text: str
    words: list[str]
    lang_tags: list[str]
    cmi: float
    switches: int
    script_mixed: bool
    stats: dict = field(default_factory=dict)


class TextNormalizer:
    """Configurable normaliser. Every step can be switched off for ablations."""

    def __init__(self, normalize: bool = True, transliterate_devanagari: bool = True, demojize: bool = True,
                 expand_slang: bool = True, collapse_repeats: bool = True, canonical_spelling: bool = True,
                 lowercase: bool = True):
        self.enabled = normalize
        self.translit = transliterate_devanagari
        self.demojize = demojize
        self.slang = expand_slang
        self.repeats = collapse_repeats
        self.spelling = canonical_spelling
        self.lower = lowercase

    @classmethod
    def from_config(cls, cfg) -> "TextNormalizer":
        return cls(**{k: v for k, v in dict(cfg).items() if k in cls.__init__.__code__.co_varnames})

    def __call__(self, raw: str) -> NormalizedText:
        raw = "" if raw is None else str(raw)
        stats = raw_cue_stats(raw)
        if not self.enabled:
            words = raw.split()
            tags = identify_languages(words)
            lid = analyze(raw)
            return NormalizedText(raw, raw, words, tags, lid.cmi, lid.switches, lid.script_mixed, stats)

        t = unicodedata.normalize("NFKC", raw)
        t = _URL_RE.sub(" <url> ", t)
        t = _MENTION_RE.sub(" <user> ", t)
        t = _HASHTAG_RE.sub(lambda m: " " + _split_hashtag(m.group(1)) + " ", t)
        if self.demojize:
            t = _demojize(t)
        if self.translit:
            t = transliterate_devanagari(t)
        if self.lower:
            t = t.lower()
        if self.repeats:
            t = _REPEAT_RE.sub(r"\1\1", t)
        words = _WORD_TOKEN_RE.findall(t)
        out: list[str] = []
        for w in words:
            if self.spelling and w in SPELLING:
                w = SPELLING[w]
            if self.slang and w in SLANG:
                out.extend(SLANG[w].split())
            else:
                out.append(w)
        tags = identify_languages(out)
        from .language_id import code_mixing_index, switch_count
        has_dev = any("\u0900" <= c <= "\u097f" for c in raw)
        has_lat = any(c.isascii() and c.isalpha() for c in raw)
        return NormalizedText(raw, " ".join(out), out, tags, code_mixing_index(tags), switch_count(tags),
                              has_dev and has_lat, stats)
