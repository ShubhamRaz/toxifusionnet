"""Torch dataset / collator. Text is normalised to words, then tokenised with ``is_split_into_words=True`` so that
word-level rationales can be mapped onto sub-words via ``word_ids``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..explain.rationale import ground_truth_attention
from ..labels import IGNORE_INDEX, TASKS, label_to_id
from ..preprocessing.features import CUE_DIM, cue_vector
from ..preprocessing.normalize import TextNormalizer


def encode_words(tokenizer, words_list: list[list[str]], max_length: int):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A *fast* tokenizer is required (word_ids are used to align word-level rationales to sub-words). "
                         "Use an encoder whose Hugging Face tokenizer has a fast implementation.")
    return tokenizer(words_list, is_split_into_words=True, truncation=True, max_length=max_length, padding=False)


class HateSarcasmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, normalizer: TextNormalizer, max_length: int = 128,
                 tasks=TASKS, rationale_cfg=None, image_dir=None,
                 image_processor=None, use_clip: bool = False, clip_store=None):
        self.df = df.reset_index(drop=True)
        self.tasks = list(tasks)
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.use_clip = use_clip
        self.clip_store = clip_store
        nts = [normalizer(t) for t in self.df["text"]]
        words = [nt.words if nt.words else ["<empty>"] for nt in nts]
        enc = encode_words(tokenizer, words, max_length)
        use_rat = bool(rationale_cfg and rationale_cfg.get("enabled", False))
        self.items = []
        for i, row in enumerate(self.df.itertuples(index=False)):
            labels = {t: label_to_id(t, getattr(row, t, None)) for t in TASKS}  # gold for ALL tasks (loss uses model tasks)
            wid = enc.word_ids(i)
            ht = st = None
            if use_rat:
                ht, st = ground_truth_attention(
                    words[i], wid, getattr(row, "hate_rationale", None), getattr(row, "sarcasm_rationale", None),
                    tau=float(rationale_cfg.get("temperature", 0.5)), mode=rationale_cfg.get("mode", "text"),
                    use_sarcasm=bool(rationale_cfg.get("use_sarcasm_rationale", True)))

            # Image loading with alt extension fallback or cached feature store
            img_tensor = None
            vis_feat = None
            has_image = 0.0
            source_ds = getattr(row, "source_dataset", "multibully")
            if self.use_clip:
                img_name = getattr(row, "id", None)
                if self.clip_store is not None and img_name and source_ds == "multibully":
                    vis_feat, has_image = self.clip_store.get(str(img_name))
                elif self.image_dir and img_name and source_ds == "multibully":
                    from pathlib import Path
                    img_path = Path(self.image_dir) / str(img_name)
                    if not img_path.exists():
                        stem = img_path.stem
                        alt_ext = ".png" if img_path.suffix.lower() == ".jpg" else ".jpg"
                        alt_path = img_path.parent / f"{stem}{alt_ext}"
                        if alt_path.exists():
                            img_path = alt_path
                    if img_path.exists():
                        try:
                            from PIL import Image
                            pil_img = Image.open(img_path).convert("RGB")
                            if self.image_processor is not None:
                                proc_res = self.image_processor(images=pil_img, return_tensors="pt")
                                img_tensor = proc_res["pixel_values"].squeeze(0)
                            else:
                                import torchvision.transforms as T
                                tr = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                img_tensor = tr(pil_img)
                            has_image = 1.0
                        except Exception:
                            has_image = 0.0
                            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
                    else:
                        has_image = 0.0
                        img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
                else:
                    has_image = 0.0
                    if self.clip_store is not None:
                        vis_feat = torch.zeros(768, dtype=torch.float32)
                    else:
                        img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

            self.items.append({
                "input_ids": enc["input_ids"][i], "labels": labels, "cue": cue_vector(nts[i]),
                "hate_t": ht, "sarc_t": st, "idx": i,
                "pixel_values": img_tensor, "vis_feat": vis_feat, "has_image": has_image, "source_dataset": source_ds
            })
        self.normalized = nts

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id: int):
    def collate(batch):
        B, T = len(batch), max(len(b["input_ids"]) for b in batch)
        ids = torch.full((B, T), pad_id, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.long)
        ht = torch.zeros((B, T)); st = torch.zeros((B, T))
        hv = torch.zeros(B, dtype=torch.bool); sv = torch.zeros(B, dtype=torch.bool)
        has_pix = any(b.get("pixel_values") is not None for b in batch)
        has_vis = any(b.get("vis_feat") is not None for b in batch)
        pix = torch.stack([b["pixel_values"] for b in batch]) if has_pix else None
        vis = torch.stack([b["vis_feat"] for b in batch]) if has_vis else None
        img_mask = torch.tensor([[b["has_image"]] for b in batch], dtype=torch.float32) if (has_pix or has_vis) else None
        sources = [b.get("source_dataset", "multibully") for b in batch]
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"]); mask[i, :n] = 1
            if b["hate_t"] is not None:
                ht[i, :n] = torch.from_numpy(b["hate_t"]); hv[i] = True
            if b["sarc_t"] is not None:
                st[i, :n] = torch.from_numpy(b["sarc_t"]); sv[i] = True
        return {
            "input_ids": ids, "attention_mask": mask,
            "cue": torch.from_numpy(np.stack([b["cue"] for b in batch])),
            "labels": {t: torch.tensor([b["labels"][t] for b in batch], dtype=torch.long) for t in TASKS},
            "hate_attn_target": ht, "hate_attn_valid": hv, "sarc_attn_target": st, "sarc_attn_valid": sv,
            "idx": torch.tensor([b["idx"] for b in batch]),
            "pixel_values": pix, "vis_feat": vis, "image_mask": img_mask, "source_dataset": sources,
        }
    return collate
