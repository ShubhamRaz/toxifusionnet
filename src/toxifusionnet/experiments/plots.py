"""Figures from a finished run directory: training curves, confusion matrices, sarcastic-subset FNR."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_run(run_dir: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    run_dir = Path(run_dir); out_dir = Path(out_dir or run_dir / "figures"); out_dir.mkdir(parents=True, exist_ok=True)
    res = json.loads((run_dir / "metrics.json").read_text())
    paths = []
    h = pd.read_csv(run_dir / "history.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h.epoch, h.train_loss, marker="o"); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(h.epoch, h.val_mean_macro_f1, marker="o", color="tab:green"); ax[1].set_title("Validation mean macro-F1"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); p = out_dir / "training_curves.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(p)

    tasks = [t for t in ("sarcasm", "hate", "target", "severity") if t in res["test"]]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.4))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tasks):
        cm = np.array(res["test"][t]["confusion_matrix"]); cls = res["test"][t]["classes"]
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cls))); ax.set_yticks(range(len(cls)))
        ax.set_xticklabels(cls, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(cls, fontsize=7)
        for i in range(len(cls)):
            for j in range(len(cls)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > .5 else "black")
        ax.set_title(f"{t} (test)", fontsize=9); ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("gold", fontsize=8)
    fig.tight_layout(); p = out_dir / "confusion_matrices.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(p)

    ss = res["test"].get("sarcasm_subset")
    if ss:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ks = ["sarcastic", "non_sarcastic"]
        ax.bar(ks, [ss[k]["false_negative_rate"] for k in ks], color=["tab:red", "tab:gray"])
        ax.set_ylabel("hate false-negative rate"); ax.set_title("Hate FNR by sarcasm", fontsize=10)
        fig.tight_layout(); p = out_dir / "fnr_by_sarcasm.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(p)
    return paths


def plot_ablation(summary_csv: str | Path, out_path: str | Path, metric: str = "hate_macro_f1"):
    s = pd.read_csv(summary_csv, index_col=0)
    mean, std = s[f"{metric}_mean"], s.get(f"{metric}_std", pd.Series(0, index=s.index)).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(mean.index[::-1], mean.values[::-1], xerr=std.values[::-1], color="tab:blue")
    ax.set_xlabel(metric); ax.set_title("Ablation study (test)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
