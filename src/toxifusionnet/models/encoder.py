"""Encoder + tokenizer construction.

* Real runs: any Hugging Face encoder, e.g. ``google/muril-base-cased`` (default), ``xlm-roberta-base`` or
  ``l3cube-pune/hing-bert`` (Hindi-English code-mixed).
* Smoke tests: ``encoder_name: tiny`` builds a small randomly-initialised BERT and trains a WordPiece tokenizer from the
  training texts, so the whole pipeline can be exercised offline in seconds (no pre-trained weights, no network).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel, PreTrainedTokenizerFast

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def train_wordpiece_tokenizer(word_lists: Iterable[list[str]], vocab_size: int = 2000) -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=1)
    tk.train_from_iterator((" ".join(w) for w in word_lists), trainer)
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tk.token_to_id("[CLS]")), ("[SEP]", tk.token_to_id("[SEP]"))])
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")


def _heads(h: int) -> int:
    return next(n for n in (4, 5, 2, 1) if h % n == 0)


def build_encoder_and_tokenizer(model_cfg, corpus_words: list[list[str]] | None = None, other_words: list[list[str]] | None = None):
    """Return ``(encoder, tokenizer)`` according to ``model_cfg.encoder_name``.

    ``tiny`` + ``model.glove_path``: word-level vocabulary whose embedding table is initialised from pre-trained
    vectors (hidden size = vector dimension). ``other_words`` are the (unlabelled) words of validation/test texts."""
    name = model_cfg.encoder_name
    if name == "tiny" and model_cfg.get("glove_path"):
        from .embeddings import build_glove_vocab_and_tokenizer, init_embedding_table
        tok, vecs, dim, stats = build_glove_vocab_and_tokenizer(corpus_words, other_words or [], model_cfg.glove_path,
                                                                int(model_cfg.get("min_word_freq", 2)))
        cfg = BertConfig(vocab_size=len(tok), hidden_size=dim, num_hidden_layers=int(model_cfg.get("tiny_layers", 2)),
                         num_attention_heads=_heads(dim), intermediate_size=2 * dim, max_position_embeddings=256,
                         pad_token_id=tok.pad_token_id)
        enc = BertModel(cfg)
        init_embedding_table(enc, tok, vecs, float(model_cfg.get("embedding_scale", 0.1)))
        enc.pretrained_vocab_stats = stats
        return enc, tok
    if name == "tiny":
        if corpus_words is None:
            raise ValueError("encoder_name='tiny' needs the training corpus to build a tokenizer")
        tok = train_wordpiece_tokenizer(corpus_words, vocab_size=int(model_cfg.get("tiny_vocab", 2000)))
        h = int(model_cfg.get("tiny_hidden", 64))
        cfg = BertConfig(vocab_size=len(tok), hidden_size=h, num_hidden_layers=int(model_cfg.get("tiny_layers", 2)),
                         num_attention_heads=4, intermediate_size=2 * h, max_position_embeddings=256,
                         pad_token_id=tok.pad_token_id)
        return BertModel(cfg), tok
    tok = AutoTokenizer.from_pretrained(name)
    enc = AutoModel.from_pretrained(name)
    return enc, tok


def rebuild_encoder_from_dir(run_dir: str | Path):
    """Rebuild the (randomly initialised) encoder architecture + tokenizer saved by the trainer (no hub access needed)."""
    run_dir = Path(run_dir)
    cfg = AutoConfig.from_pretrained(run_dir / "encoder_config")
    tok = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    return AutoModel.from_config(cfg), tok
