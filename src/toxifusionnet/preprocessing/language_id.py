"""Token-level language identification for Hindi-English code-mixed text (heuristic, lexicon + context).

Tags: ``hi`` (Hindi, romanized or Devanagari), ``en`` (English), ``univ`` (emoji / numbers / URLs / punctuation),
``unk`` (unresolved). Replace with a trained LID model (e.g. LinCE baselines) when labelled data is available.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .lexicons import AMBIGUOUS, ENGLISH_COMMON, ENGLISH_SUFFIXES, HINDI_SUFFIXES, ROMAN_HINDI
from .transliteration import has_devanagari

TOKEN_RE = re.compile(
    r"https?://\S+|www\.\S+|@\w+|#\w+|[\u0900-\u097F]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]|_",
    re.UNICODE,
)
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B50\u2B06\u2764]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _initial_tag(tok: str) -> str:
    if has_devanagari(tok):
        return "hi"
    if (tok.startswith("<") and tok.endswith(">")) or tok.startswith("emoji_"):
        return "univ"
    if tok.startswith(("http", "www", "@")) or re.fullmatch(r"[\d.,]+", tok) or not re.search(r"[A-Za-z]", tok):
        return "univ"
    if tok.startswith("#"):
        return "unk"
    t = tok.lower()
    in_hi, in_en = t in ROMAN_HINDI, t in ENGLISH_COMMON
    if t in AMBIGUOUS or (in_hi and in_en):
        return "amb"
    if in_hi:
        return "hi"
    if in_en:
        return "en"
    if t.endswith(ENGLISH_SUFFIXES) and len(t) > 4:
        return "en"
    if t.endswith(HINDI_SUFFIXES):
        return "hi"
    return "unk"


def identify_languages(tokens: list[str]) -> list[str]:
    """Return one tag per token; ambiguous / unknown tokens are resolved by neighbouring words."""
    tags = [_initial_tag(t) for t in tokens]
    word_idx = [i for i, g in enumerate(tags) if g != "univ"]
    for _ in range(2):  # two passes: neighbours resolved in pass 1 help pass 2
        for pos, i in enumerate(word_idx):
            if tags[i] not in ("amb", "unk"):
                continue
            neigh = []
            for j in (pos - 1, pos + 1):
                if 0 <= j < len(word_idx) and tags[word_idx[j]] in ("hi", "en"):
                    neigh.append(tags[word_idx[j]])
            if neigh:
                tags[i] = Counter(neigh).most_common(1)[0][0] if len(set(neigh)) == 1 else neigh[0]
    # whatever is left: ambiguous defaults to Hindi function-word reading, unknown stays unk
    return ["hi" if g == "amb" else g for g in tags]


def code_mixing_index(tags: list[str]) -> float:
    """Das & Gambäck CMI in [0, 100]: 100 * (1 - max_lang_count / (n - u)); 0 for monolingual text."""
    langs = [g for g in tags if g in ("hi", "en")]
    n_univ = sum(g == "univ" for g in tags)
    n = len(tags)
    if n - n_univ <= 0 or not langs:
        return 0.0
    return 100.0 * (1.0 - max(Counter(langs).values()) / (n - n_univ))


def switch_count(tags: list[str]) -> int:
    langs = [g for g in tags if g in ("hi", "en")]
    return sum(1 for a, b in zip(langs, langs[1:]) if a != b)


@dataclass
class LIDResult:
    tokens: list[str]
    tags: list[str]
    cmi: float
    switches: int
    script_mixed: bool


def analyze(text: str) -> LIDResult:
    toks = tokenize(text)
    tags = identify_languages(toks)
    has_dev = any(has_devanagari(t) for t in toks)
    has_lat = any(re.search(r"[A-Za-z]", t) for t in toks)
    return LIDResult(toks, tags, code_mixing_index(tags), switch_count(tags), has_dev and has_lat)
