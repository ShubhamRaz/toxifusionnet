"""Heuristic Devanagari -> Roman transliteration (ITRANS-lite) so mixed-script posts share one script.

This is intentionally simple: it handles consonant+matra composition, virama, anusvara/chandrabindu, nukta
and applies schwa deletion at word end. Swap in ``indic-transliteration`` for production-grade output.
"""
from __future__ import annotations

import re

_INDEP_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au"}
_MATRAS = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au"}
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}
_NUKTA_COMBO = {"क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f"}
_VIRAMA, _ANUSVARA, _CHANDRABINDU, _VISARGA, _NUKTA = "्", "ं", "ँ", "ः", "़"
_DIGITS = {c: str(i) for i, c in enumerate("०१२३४५६७८९")}
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text))


def _translit_word(word: str) -> str:
    import unicodedata
    word = unicodedata.normalize("NFC", word)
    out: list[str] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in _INDEP_VOWELS:
            out.append(_INDEP_VOWELS[ch])
            i += 1
        elif ch in _CONSONANTS or (i + 1 < n and word[i + 1] == _NUKTA):
            cons = _CONSONANTS.get(ch, ch)
            if i + 1 < n and word[i + 1] == _NUKTA:
                cons = _NUKTA_COMBO.get(ch + _NUKTA, cons)
                i += 1
            out.append(cons)
            nxt = word[i + 1] if i + 1 < n else ""
            if nxt in _MATRAS:
                out.append(_MATRAS[nxt])
                i += 2
            elif nxt == _VIRAMA:
                i += 2
            else:
                # inherent schwa; deleted at the end of the word and before another consonant cluster end
                at_end = i + 1 >= n or word[i + 1] in (_ANUSVARA, _CHANDRABINDU, _VISARGA) and i + 2 >= n
                if not at_end:
                    out.append("a")
                i += 1
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
            i += 1
        elif ch in (_ANUSVARA, _CHANDRABINDU):
            out.append("n")
            i += 1
        elif ch == _VISARGA:
            out.append("h")
            i += 1
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
            i += 1
        elif ch == "।":
            out.append(".")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def transliterate_devanagari(text: str) -> str:
    """Transliterate only the Devanagari runs of ``text``; leave everything else untouched."""
    if not has_devanagari(text):
        return text
    return re.sub(r"[\u0900-\u097F]+", lambda m: _translit_word(m.group(0)), text)
