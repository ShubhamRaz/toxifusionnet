#!/usr/bin/env python3
"""Download pretrained SOTA models (L3Cube HingRoBERTa & OpenAI CLIP ViT) locally to disk.
"""
import sys
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

def download_models():
    base_dir = Path("models/pretrained_weights")
    hing_dir = base_dir / "hing_roberta"
    clip_dir = base_dir / "clip_vit"

    hing_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("1/2 Downloading L3Cube HingRoBERTa text model...")
    print("==================================================")
    tok = AutoTokenizer.from_pretrained("l3cube-pune/hing-roberta")
    tok.save_pretrained(hing_dir)
    print("-> Tokenizer saved to:", hing_dir)

    model = AutoModel.from_pretrained("l3cube-pune/hing-roberta")
    model.save_pretrained(hing_dir)
    print("-> HingRoBERTa weights saved to:", hing_dir)

    print("\n==================================================")
    print("2/2 Downloading OpenAI CLIP ViT vision model...")
    print("==================================================")
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    proc.save_pretrained(clip_dir)
    print("-> CLIP image processor saved to:", clip_dir)

    vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    vision.save_pretrained(clip_dir)
    print("-> CLIP vision weights saved to:", clip_dir)

    print("\n[SUCCESS] All SOTA model weights downloaded and saved locally in models/pretrained_weights/!")

if __name__ == "__main__":
    download_models()
