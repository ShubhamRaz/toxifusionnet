"""Pre-trained static word vectors (GloVe / word2vec text format) to initialise the embedding table of a small
from-scratch encoder -- the fallback when no pre-trained transformer can be used (CPU only / offline).

Vocabulary policy (no labels involved): every word that occurs >= ``min_freq`` times in the TRAINING texts, plus any word
of any split that has a pre-trained vector (so unseen validation/test words keep their pre-trained meaning).
"""
from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from transformers import PreTrainedTokenizerFast

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def load_vector_subset(path: str | Path, wanted: set[str]) -> tuple[dict[str, np.ndarray], int]:
    """Stream a (gzipped) word2vec-text file and keep only the vectors of ``wanted`` words. Returns (vectors, dim).
    The extracted subset is cached next to the file (keyed by a hash of ``wanted``), so repeated runs skip the slow scan."""
    import hashlib
    path = Path(path)
    key = hashlib.sha1("\n".join(sorted(wanted)).encode("utf-8")).hexdigest()[:12]
    cache = path.with_name(f"{path.name}.subset-{key}.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {w: v for w, v in zip(z["words"].tolist(), z["vecs"])}, int(z["vecs"].shape[1])
    op = gzip.open if path.suffix == ".gz" else open
    vecs: dict[str, np.ndarray] = {}
    dim = 0
    with op(path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split(" ")
            if i == 0 and len(parts) == 2 and all(p.isdigit() for p in parts):
                dim = int(parts[1]); continue                      # word2vec header "n_words dim"
            w = parts[0]
            if w in wanted and w not in vecs:
                v = np.asarray(parts[1:], dtype=np.float32)
                dim = dim or len(v)
                if len(v) == dim:
                    vecs[w] = v
    if vecs:
        tmp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, words=np.array(list(vecs), dtype=str), vecs=np.stack(list(vecs.values())))
        tmp.replace(cache)
    return vecs, dim


def build_word_tokenizer(vocab_words: list[str]) -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    vocab = {t: i for i, t in enumerate(SPECIALS + [w for w in vocab_words if w not in SPECIALS])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()           # words are already tokenised by our normaliser
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")


def build_glove_vocab_and_tokenizer(train_words: list[list[str]], other_words: list[list[str]], path: str | Path,
                                    min_freq: int = 2, max_vocab: int = 30000):
    """Returns (tokenizer, vectors dict, dim, stats)."""
    freq = Counter(w.lower() for ws in train_words for w in ws)
    everything = set(freq) | {w.lower() for ws in other_words for w in ws}
    vecs, dim = load_vector_subset(path, everything)
    vocab = [w for w, c in freq.most_common() if c >= min_freq or w in vecs]
    vocab += sorted(w for w in vecs if w not in freq)           # unseen-in-train words that have a pre-trained vector
    vocab = vocab[:max_vocab]
    stats = {"vocab": len(vocab), "with_pretrained_vector": sum(w in vecs for w in vocab), "dim": dim,
             "train_token_coverage": float(sum(c for w, c in freq.items() if w in set(vocab)) / max(sum(freq.values()), 1))}
    return build_word_tokenizer(vocab), vecs, dim, stats


def init_embedding_table(encoder, tokenizer, vecs: dict[str, np.ndarray], scale: float = 0.1, seed: int = 0):
    """Copy pre-trained vectors into ``encoder.embeddings.word_embeddings`` (rescaled to a BERT-like magnitude)."""
    import torch
    emb = encoder.embeddings.word_embeddings.weight
    arr = np.stack(list(vecs.values()))
    mult = scale / max(float(arr.std()), 1e-6)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for w, i in tokenizer.get_vocab().items():
            if w in vecs:
                emb[i] = torch.from_numpy(vecs[w] * mult)
            elif i >= len(SPECIALS):
                emb[i] = torch.randn(emb.shape[1], generator=g) * scale * 0.5
