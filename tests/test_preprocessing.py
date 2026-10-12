import numpy as np

from toxifusionnet.preprocessing.features import CUE_DIM, CUE_NAMES, cue_vector
from toxifusionnet.preprocessing.language_id import analyze, code_mixing_index
from toxifusionnet.preprocessing.normalize import TextNormalizer
from toxifusionnet.preprocessing.transliteration import transliterate_devanagari

N = TextNormalizer()


def test_masks_urls_mentions_and_splits_hashtags():
    r = N("Check https://t.co/abc @someone #IndiaFirst")
    assert "<url>" in r.words and "<user>" in r.words
    assert "india" in r.words and "first" in r.words


def test_devanagari_transliteration_and_spelling_canonicalisation():
    assert transliterate_devanagari("क्या") in ("kyaa", "kya")
    assert N("क्या है").words == ["kya", "hai"]
    assert N("ये लोग बहुत अच्छे हैं").script_mixed is False
    assert N("kya चल रहा hai").script_mixed is True


def test_elongation_and_slang():
    assert N("yaaaaar").words == ["yaar"]
    assert N("plz u r great").words[:3] == ["please", "you", "are"]


def test_emoji_is_language_neutral_token():
    r = N("wah 🙄")
    assert any(w.startswith("emoji_") for w in r.words)
    assert r.lang_tags[-1] == "univ"


def test_language_id_and_cmi():
    lid = analyze("tum logon ka future bahut bright hai")
    assert "hi" in lid.tags and "en" in lid.tags
    assert 0 < lid.cmi < 100
    assert code_mixing_index(["en", "en", "en"]) == 0.0


def test_cue_vector_shape_and_contrast():
    assert len(CUE_NAMES) == CUE_DIM
    v = cue_vector(N("Wah great job 🙄 #sarcasm"))
    assert v.shape == (CUE_DIM,) and v.dtype == np.float32
    idx = {n: i for i, n in enumerate(CUE_NAMES)}
    assert v[idx["polarity_contrast"]] == 1.0 and v[idx["hashtag_sarcasm"]] == 1.0


def test_normalizer_disabled_keeps_raw_tokens():
    r = TextNormalizer(normalize=False)("Yaaaar KYA hai")
    assert r.words == ["Yaaaar", "KYA", "hai"]
