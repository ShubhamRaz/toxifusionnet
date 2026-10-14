"""Training / evaluation loops for ToxiFusionNet."""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..data.dataset import HateSarcasmDataset, make_collate
from ..labels import IGNORE_INDEX, LABEL_SPACES, TASKS, apply_label_config
from ..models.encoder import build_encoder_and_tokenizer
from ..models.losses import MultiTaskLoss
from ..models.toxifusionnet import ToxiFusionNet
from ..preprocessing.normalize import TextNormalizer
from .metrics import evaluate_predictions, flatten_metrics

log = logging.getLogger("toxifusionnet")


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    return torch.device(pref)


def class_weights(df: pd.DataFrame, task: str, mode: str) -> list[float] | None:
    if mode != "balanced" or task not in df:
        return None
    from ..labels import label_to_id
    y = np.array([label_to_id(task, v) for v in df[task]])
    y = y[y != IGNORE_INDEX]
    k = len(LABEL_SPACES[task])
    if len(y) == 0:
        return None
    counts = np.bincount(y, minlength=k).astype(float)
    w = len(y) / (k * np.maximum(counts, 1.0))
    return w.tolist()


@torch.no_grad()
def predict_loader(model: ToxiFusionNet, loader: DataLoader, device) -> dict:
    model.eval()
    probs: dict[str, list] = {t: [] for t in model.tasks}
    idxs, gates = [], []
    for b in loader:
        pix = b["pixel_values"].to(device) if b.get("pixel_values") is not None else None
        vis = b["vis_feat"].to(device) if b.get("vis_feat") is not None else None
        im_mask = b["image_mask"].to(device) if b.get("image_mask") is not None else None
        out = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["cue"].to(device),
                    pixel_values=pix, image_mask=im_mask, vis_feat=vis)
        for t, lg in out["logits"].items():
            probs[t].append(torch.softmax(lg.float(), -1).cpu().numpy())
        idxs.append(b["idx"].numpy())
        if out["gate"] is not None:
            gates.append(out["gate"].mean(-1).cpu().numpy())
    return {"probs": {t: np.concatenate(v) for t, v in probs.items() if v}, "idx": np.concatenate(idxs),
            "gate": np.concatenate(gates) if gates else None}


def predictions_frame(df: pd.DataFrame, ds: HateSarcasmDataset, pred: dict) -> pd.DataFrame:
    order = np.argsort(pred["idx"])
    rows = pd.DataFrame({"id": df["id"].to_numpy(), "text": df["text"].to_numpy()})
    if "source_dataset" in df:
        rows["source_dataset"] = df["source_dataset"].to_numpy()
    if "narrative" in df:
        rows["narrative"] = df["narrative"].to_numpy()
    for t in TASKS:
        rows[f"gold_{t}"] = [it["labels"][t] for it in ds.items]
    for t, p in pred["probs"].items():
        p = p[order]
        rows[f"pred_{t}"] = p.argmax(-1)
        rows[f"conf_{t}"] = p.max(-1)
        if t in ("sarcasm", "hate"):
            rows[f"p_{t}"] = p[:, 1]
    if pred["gate"] is not None:
        rows["fusion_gate"] = pred["gate"][order]
    return rows


def train_run(cfg, df: pd.DataFrame, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    apply_label_config(cfg.get("labels"))   # target taxonomy must be fixed BEFORE the model / datasets are built
    set_seed(int(cfg.seed))
    device = pick_device(cfg.train.get("device", "auto"))
    tasks = list(cfg.model.tasks)
    normalizer = TextNormalizer.from_config(cfg.preprocess)
    tr_df, va_df, te_df = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    log.info("train/val/test = %d/%d/%d on %s", len(tr_df), len(va_df), len(te_df), device)

    corpus = [normalizer(t).words for t in tr_df["text"]] if cfg.model.encoder_name == "tiny" else None
    other = [normalizer(t).words for t in pd.concat([va_df["text"], te_df["text"]])] if cfg.model.encoder_name == "tiny" else None
    encoder, tokenizer = build_encoder_and_tokenizer(cfg.model, corpus, other)
    if getattr(encoder, "pretrained_vocab_stats", None):
        log.info("pre-trained vectors: %s", encoder.pretrained_vocab_stats)
    tokenizer.save_pretrained(run_dir / "tokenizer")
    encoder.config.save_pretrained(run_dir / "encoder_config")

    model = ToxiFusionNet(encoder, cfg.model).to(device)
    use_clip = bool(getattr(model, "use_clip", False))
    image_proc = getattr(model, "clip_processor", None) if use_clip else None
    image_dir = cfg.data.get("image_dir", "data/raw/bully_data/bully_data")

    clip_store = None
    if use_clip and cfg.train.get("use_clip_cache", True) and getattr(model, "clip_vision", None) is not None:
        try:
            from ..models.clip_cache import build_or_load_clip_cache
            all_df = pd.concat([tr_df, va_df, te_df], ignore_index=True)
            clip_store = build_or_load_clip_cache(
                all_df, image_dir=image_dir, vision_model=model.clip_vision,
                image_processor=image_proc, cache_dir=cfg.data.get("clip_cache_dir", "data/cache/clip_features"),
                model_name=cfg.model.get("clip_model_name", "openai/clip-vit-base-patch32"),
                device=device
            )
        except Exception as e:
            log.warning("Could not initialize CLIP cache (%s); proceeding with live visual extraction", e)

    mk = lambda d: HateSarcasmDataset(d, tokenizer, normalizer, int(cfg.data.max_length), tasks,
                                     cfg.get("rationale"), image_dir=image_dir,
                                     image_processor=image_proc, use_clip=use_clip, clip_store=clip_store)  # noqa: E731
    ds_tr, ds_va, ds_te = mk(tr_df), mk(va_df), mk(te_df)
    collate = make_collate(tokenizer.pad_token_id)
    bs = int(cfg.train.batch_size)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, collate_fn=collate, num_workers=int(cfg.train.get("num_workers", 0)))
    dl_va = DataLoader(ds_va, batch_size=bs * 2, shuffle=False, collate_fn=collate)
    dl_te = DataLoader(ds_te, batch_size=bs * 2, shuffle=False, collate_fn=collate)

    lcfg = cfg.loss
    cw = {t: class_weights(tr_df, t, lcfg.get("class_weighting", "none")) for t in model.tasks}
    cw = {t: w for t, w in cw.items() if w is not None}
    att_lambda = float(lcfg.get("attention_lambda", 0.0)) if cfg.get("rationale", {}).get("enabled", False) else 0.0
    criterion = MultiTaskLoss(model.tasks, lcfg.get("weighting", "fixed"), dict(lcfg.get("weights", {})), cw,
                              float(lcfg.get("focal_gamma", 0.0)), att_lambda,
                              severity_loss_type=lcfg.get("severity_loss_type", "ce"),
                              ordinal_penalty=float(lcfg.get("ordinal_penalty_weight", 0.5))).to(device)
    groups = model.param_groups(float(cfg.train.encoder_lr), float(cfg.train.head_lr), float(cfg.train.weight_decay),
                                extra=criterion.log_vars.parameters() if lcfg.get("weighting") == "uncertainty" else (),
                                emb_lr_mult=float(cfg.train.get("embedding_lr_mult", 1.0)))
    opt = torch.optim.AdamW(groups)
    steps = max(1, len(dl_tr) * int(cfg.train.epochs))
    sched = get_linear_schedule_with_warmup(opt, int(float(cfg.train.warmup_ratio) * steps), steps)
    use_amp = device.type == "cuda" and bool(cfg.train.get("fp16", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_epoch, bad, history = -1.0, -1, 0, []
    for epoch in range(int(cfg.train.epochs)):
        model.train(); t0 = time.time(); run_loss = 0.0
        for b in dl_tr:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items() if k != "labels"}
            batch["labels"] = {t: v.to(device) for t, v in b["labels"].items()}
            pix = batch.get("pixel_values")
            vis = batch.get("vis_feat")
            im_mask = batch.get("image_mask")
            with torch.autocast(device.type, enabled=use_amp):
                out = model(batch["input_ids"], batch["attention_mask"], batch["cue"],
                            pixel_values=pix, image_mask=im_mask, vis_feat=vis)
                loss, parts = criterion(out, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scale_before <= scaler.get_scale():
                sched.step()
            run_loss += float(loss.detach())
        pred = predict_loader(model, dl_va, device)
        va_res = evaluate_predictions(predictions_frame(va_df, ds_va, pred), model.tasks)
        score = va_res["mean_macro_f1"]
        history.append({"epoch": epoch + 1, "train_loss": run_loss / len(dl_tr), "val_mean_macro_f1": score,
                        **{f"val_{k}": v for k, v in flatten_metrics(va_res).items() if k != "mean_macro_f1"},
                        "seconds": time.time() - t0})
        log.info("epoch %d  loss %.4f  val mean macro-F1 %.4f  (%.1fs)", epoch + 1, history[-1]["train_loss"], score, history[-1]["seconds"])
        if score > best:
            best, best_epoch, bad = score, epoch + 1, 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.to_dict()}, run_dir / "model.pt")
        else:
            bad += 1
            if bad >= int(cfg.train.patience):
                log.info("early stopping at epoch %d", epoch + 1)
                break
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device)["state_dict"])
    results = {"best_epoch": best_epoch, "pretrained_vocab": getattr(encoder, "pretrained_vocab_stats", None), "device": str(device), "n_params": sum(p.numel() for p in model.parameters()),
               "label_source": df["label_source"].iloc[0] if "label_source" in df and len(df) else None}
    for name, d, ds, dl in (("val", va_df, ds_va, dl_va), ("test", te_df, ds_te, dl_te)):
        pf = predictions_frame(d, ds, predict_loader(model, dl, device))
        pf.to_csv(run_dir / f"preds_{name}.csv", index=False)
        results[name] = evaluate_predictions(pf, model.tasks)
        results[f"{name}_flat"] = flatten_metrics(results[name])
        # Compute source-specific metrics if multiple sources exist
        if "source_dataset" in pf and pf["source_dataset"].nunique() > 1:
            for src in pf["source_dataset"].unique():
                sub_pf = pf[pf["source_dataset"] == src]
                if len(sub_pf) > 0:
                    results[f"{name}_{src}"] = evaluate_predictions(sub_pf, model.tasks)
                    results[f"{name}_{src}_flat"] = flatten_metrics(results[f"{name}_{src}"])
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return results


def train_or_resume(cfg, df: pd.DataFrame, run_dir: str | Path, resume: bool = True) -> dict:
    """Like ``train_run`` but re-uses a finished run directory whose saved configuration is identical (resumable studies)."""
    run_dir = Path(run_dir)
    m, c = run_dir / "metrics.json", run_dir / "config.yaml"
    if resume and m.exists() and c.exists() and (run_dir / "model.pt").exists():
        if yaml.safe_load(c.read_text()) == cfg.to_dict():
            log.info("resuming: re-using finished run %s", run_dir)
            return json.loads(m.read_text())
    return train_run(cfg, df, run_dir)
