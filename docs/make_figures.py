"""Regenerates the report figures:  python docs/make_figures.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fc, ec="#333", fs=8.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, color="#333", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def architecture():
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(-0.2, 7); ax.axis("off")
    box(ax, 0.1, 5.3, 1.7, 0.9, "Hinglish post\n(Roman / Devanagari,\nemoji, slang)", "#e8e0fb")
    box(ax, 2.2, 5.0, 2.2, 1.5, "Preprocessing\n• token language ID\n• normalise / transliterate\n• word tokens + CMI\n• sarcasm cue vector (16-d)", "#d6e8fb", fs=7.8)
    box(ax, 4.9, 5.2, 2.3, 1.1, "Transformer encoder\nMuRIL · XLM-R · HingBERT\n(sub-word states H)", "#cfe3f7")
    arrow(ax, (1.8, 5.75), (2.2, 5.75)); arrow(ax, (4.4, 5.75), (4.9, 5.75))
    arrow(ax, (3.3, 5.0), (4.0, 3.9), color="#2a7a1f", ls="--")          # cue vector -> sarcasm branch
    ax.text(2.0, 4.35, "cue vector", fontsize=7.5, color="#2a7a1f", style="italic")
    box(ax, 4.0, 3.45, 2.6, 0.9, "Sarcasm branch\nattention pooling ⊕ cue-MLP", "#d9f2d0")
    box(ax, 7.3, 3.45, 2.4, 0.9, "Hate branch\nattention pooling", "#fde3c8")
    arrow(ax, (5.5, 5.2), (5.3, 4.35)); arrow(ax, (6.6, 5.2), (8.5, 4.35))
    box(ax, 4.0, 2.05, 2.6, 0.8, "Sarcasm head (2)", "#bfe6b3", bold=True)
    arrow(ax, (5.3, 3.45), (5.3, 2.85))
    box(ax, 7.0, 1.95, 3.0, 1.0, "Sarcasm-aware gated fusion\nLN( h + g ⊙ tanh(W s) )\ng = σ(W[h; s; P(sarcasm)])", "#fbd3a8", fs=7.8, bold=True)
    arrow(ax, (8.5, 3.45), (8.5, 2.95))
    arrow(ax, (6.6, 3.7), (7.2, 2.95), color="#2a7a1f")                   # sarcasm repr s, P(sarc) -> fusion
    ax.text(5.95, 3.12, "s, P(sarc)", fontsize=7.5, color="#2a7a1f", style="italic")
    heads = [(5.2, "Hate head (2)"), (7.6, "Target head (K)\nonly if hate"), (10.0, "Severity head (3)\nonly if hate")]
    for x, t in heads:
        box(ax, x, 0.65, 1.9, 0.7, t, "#f6c28b", bold=True, fs=8)
        arrow(ax, (8.5, 1.95), (x + 0.95, 1.35))
        arrow(ax, (x + 0.95, 0.65), (x + 0.95, 0.38))
    box(ax, 4.0, -0.1, 7.9, 0.48, "Output: sarcasm · hate · target · severity · token rationales (attention / LIME) · language info", "#e3f1fb", fs=8, bold=True)
    arrow(ax, (4.5, 2.05), (4.5, 0.38), color="#2a7a1f")
    box(ax, 10.1, 5.2, 1.8, 1.25, "Loss\nΣ w_t·CE_t (masked)\n+ λ·CE(attn, rationale)", "#f3f3f3", ec="#777", fs=7.4, ls="--")
    box(ax, 10.1, 3.35, 1.8, 1.2, "Hate + sarcasm\nrationales → Alg. 1\n→ softmax(τ) → supervise\nboth attention poolings", "#fff7c2", ec="#a89300", fs=7.4, ls="--")
    arrow(ax, (11.0, 4.55), (11.0, 5.2), color="#a89300", ls="--")
    arrow(ax, (10.1, 3.95), (9.7, 3.95), color="#a89300", ls="--")        # supervise hate attention
    ax.text(0.2, 0.2, "Labels missing for a task are masked (−100),\nso partially annotated corpora can be used.", fontsize=7.6, style="italic", color="#555")
    fig.tight_layout(); fig.savefig(OUT / "architecture.png", dpi=170); plt.close(fig)


def rationale_figure():
    words = ["wah", "wah", "us", "dharm", "ke", "log", "to", "genius", "hain", "lol"]
    h = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0], float)
    s = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 1], float)
    both = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 0], float)
    h2, s2 = np.maximum(h, both), np.maximum(s, both)
    r = np.where(s2 > h2, (s2 + h2) / 2, np.where(s2 < h2, h2, h2 + s2 / 2))
    tau = 0.5
    z = np.exp(r / tau); p = z / z.sum()
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 4.6), sharex=True)
    rows = [("hate rationale  h", h2, "#e8833a"), ("sarcasm rationale  s", s2, "#4caf50"), ("combined  r (Alg. 1)", r, "#7e57c2"),
            (f"target attention  softmax(r/τ), τ={tau}", p, "#1e88e5")]
    for ax, (name, v, c) in zip(axes, rows):
        ax.bar(range(len(words)), v, color=c); ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_yticks([]); [ax.spines[k].set_visible(False) for k in ("top", "right", "left")]
        for i, val in enumerate(v):
            if val > 0:
                ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, max(v.max() * 1.35, 0.1))
    axes[-1].set_xticks(range(len(words))); axes[-1].set_xticklabels(words, fontsize=9)
    fig.suptitle("Ground-truth attention from hate + sarcasm rationales (illustrative example)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "rationale_example.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    architecture(); rationale_figure(); print("figures written to", OUT)
