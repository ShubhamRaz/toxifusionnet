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
import transformers

from ..data.dataset import HateSarcasmDataset, make_collate
from ..labels import IGNORE_INDEX, LABEL_SPACES, TASKS, apply_label_config
from ..models.encoder import build_encoder_and_tokenizer
from ..models.losses import MultiTaskLoss
from ..models.toxifusionnet import ToxiFusionNet
from ..preprocessing.normalize import TextNormalizer
from .balancing import compute_fold_class_weights, compute_task_class_weights
from .metrics import evaluate_predictions, flatten_metrics

log = logging.getLogger("toxifusionnet")


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    return torch.device(pref)


def class_weights(df: pd.DataFrame, task: str, mode: str) -> list[float] | None:
    res = compute_task_class_weights(df, task, mode=mode)
    return res.get("weights")


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
        if p.shape[-1] == 2:
            rows[f"p_{t}"] = p[:, 1]
        elif t in ("sarcasm", "hate"):
            rows[f"p_{t}"] = p[:, 1]
    if pred["gate"] is not None:
        rows["fusion_gate"] = pred["gate"][order]
    return rows


def train_run(cfg, df: pd.DataFrame, run_dir: str | Path, full_data: bool = False) -> dict:
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    apply_label_config(cfg.get("labels"))   # target taxonomy must be fixed BEFORE the model / datasets are built
    set_seed(int(cfg.seed))
    device = pick_device(cfg.train.get("device", "auto"))
    tasks = list(cfg.model.tasks)
    normalizer = TextNormalizer.from_config(cfg.preprocess)
    if full_data:
        # Every row participates in gradient updates. Empty evaluation partitions
        # are deliberate: this mode must not claim an independent test score.
        tr_df = df.reset_index(drop=True).copy()
        va_df = df.iloc[0:0].copy().reset_index(drop=True)
        te_df = df.iloc[0:0].copy().reset_index(drop=True)
    else:
        tr_df, va_df, te_df = (
            df[df.split == s].reset_index(drop=True)
            for s in ("train", "val", "test")
        )

    # Exclude auxiliary rows whose labels across all tasks are completely masked/NaN (zero supervision)
    if "source_dataset" in tr_df.columns:
        is_aux = tr_df["source_dataset"] != "multibully"
        if is_aux.any():
            from ..labels import label_to_id, IGNORE_INDEX
            has_sup = pd.Series(False, index=tr_df.index)
            for t in tasks:
                if t in tr_df.columns:
                    has_sup = has_sup | tr_df[t].apply(lambda v: label_to_id(t, v) != IGNORE_INDEX)
            zero_sup_aux = is_aux & (~has_sup)
            if zero_sup_aux.any():
                n_zero = int(zero_sup_aux.sum())
                log.info("Excluding %d zero-supervision auxiliary rows from training loader", n_zero)
                tr_df = tr_df[~zero_sup_aux].reset_index(drop=True)

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
    mode = lcfg.get("class_weighting", "none")
    max_w = float(lcfg.get("max_class_weight", 4.0))
    src_filter = lcfg.get("class_weight_source")
    if src_filter is None and "source_dataset" in tr_df.columns and tr_df["source_dataset"].nunique() > 1:
        src_filter = "multibully"

    balancing_meta = compute_fold_class_weights(
        tr_df, model.tasks, mode=mode, max_weight=max_w, source_filter=src_filter
    )
    (run_dir / "class_weights.json").write_text(json.dumps(balancing_meta, indent=2))

    cw = {t: meta["weights"] for t, meta in balancing_meta.items() if meta.get("weights") is not None}
    att_lambda = float(lcfg.get("attention_lambda", 0.0)) if cfg.get("rationale", {}).get("enabled", False) else 0.0
    focal_use_cw = bool(lcfg.get("focal_use_class_weights", False))
    criterion = MultiTaskLoss(model.tasks, lcfg.get("weighting", "fixed"), dict(lcfg.get("weights", {})), cw,
                              float(lcfg.get("focal_gamma", 0.0)), att_lambda,
                              severity_loss_type=lcfg.get("severity_loss_type", "ce"),
                              ordinal_penalty=float(lcfg.get("ordinal_penalty_weight", 0.5)),
                              focal_use_class_weights=focal_use_cw).to(device)
    groups = model.param_groups(float(cfg.train.encoder_lr), float(cfg.train.head_lr), float(cfg.train.weight_decay),
                                extra=criterion.log_vars.parameters() if lcfg.get("weighting") == "uncertainty" else (),
                                emb_lr_mult=float(cfg.train.get("embedding_lr_mult", 1.0)))
    opt = torch.optim.AdamW(groups)
    steps_per_epoch = cfg.train.get("steps_per_epoch")
    steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else len(dl_tr)
    epochs = int(cfg.train.epochs)
    total_steps = int(cfg.train.get("max_steps", steps_per_epoch * epochs))
    warmup_steps = int(float(cfg.train.get("warmup_ratio", 0.10)) * total_steps)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    use_amp = (device.type == "cuda" or bool(cfg.train.get("use_cpu_amp", False))) and bool(cfg.train.get("fp16", True))
    scaler_device = device.type if device.type in ("cuda", "cpu") else "cuda"
    scaler = torch.amp.GradScaler(scaler_device, enabled=use_amp)

    best, best_epoch, best_step, bad, history = -1.0, -1, -1, 0, []
    global_step = 0
    attempted_batches = 0
    skipped_updates = 0
    train_iter = iter(dl_tr)
    use_early_stopping = (not full_data) and bool(cfg.train.get("early_stopping", True)) and int(cfg.train.get("patience", 0)) > 0
    if not use_early_stopping:
        log.info("Fixed-budget mode active: early stopping is DISABLED. Training will complete all %d planned optimizer updates.", total_steps)

    max_consecutive_skipped = int(cfg.train.get("max_consecutive_skipped", 50))
    consecutive_skipped = 0

    for epoch in range(epochs):
        model.train(); t0 = time.time(); run_loss = 0.0
        epoch_successful = 0
        epoch_attempted = 0
        epoch_skipped = 0

        while epoch_successful < steps_per_epoch and global_step < total_steps:
            try:
                b = next(train_iter)
            except StopIteration:
                train_iter = iter(dl_tr)
                b = next(train_iter)
            attempted_batches += 1
            epoch_attempted += 1

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
            scale_after = scaler.get_scale()

            step_successful = (not scaler.is_enabled()) or (scale_before <= scale_after)
            if step_successful:
                sched.step()
                global_step += 1
                epoch_successful += 1
                consecutive_skipped = 0
                run_loss += float(loss.detach())
            else:
                skipped_updates += 1
                epoch_skipped += 1
                consecutive_skipped += 1
                log.warning(
                    "GradScaler skipped optimizer update due to inf/nan gradients (scale: %.1f -> %.1f). "
                    "Attempted: %d, successful: %d/%d, epoch skipped: %d.",
                    scale_before, scale_after, attempted_batches, global_step, total_steps, epoch_skipped
                )
                if consecutive_skipped >= max_consecutive_skipped:
                    raise RuntimeError(
                        f"GradScaler skipped {consecutive_skipped} consecutive updates due to inf/nan gradients. "
                        "Training diverged; aborting."
                    )

        epoch_loss = run_loss / max(1, epoch_successful)

        if full_data:
            # No validation set exists in all-data mode. Save the latest completed
            # epoch so the final checkpoint reflects the fixed training budget.
            score = float("nan")
            is_best = None
            best_epoch, best_step, bad = epoch + 1, global_step, 0
            va_res = None
            torch.save(
                {"state_dict": model.state_dict(), "cfg": cfg.to_dict()},
                run_dir / "model.pt",
            )
        else:
            pred = predict_loader(model, dl_va, device)
            va_res = evaluate_predictions(
                predictions_frame(va_df, ds_va, pred), model.tasks
            )
            score = va_res["mean_macro_f1"]
            is_best = score > best
            if is_best:
                best, best_epoch, best_step, bad = score, epoch + 1, global_step, 0
                torch.save(
                    {"state_dict": model.state_dict(), "cfg": cfg.to_dict()},
                    run_dir / "model.pt",
                )
            else:
                bad += 1
                if use_early_stopping and bad >= int(cfg.train.patience):
                    log.info("early stopping at epoch %d (step %d)", epoch + 1, global_step)
                    break

        history.append({
            "epoch": epoch + 1,
            "global_step": global_step,
            "successful_updates": global_step,
            "attempted_batches": attempted_batches,
            "skipped_updates": skipped_updates,
            "epoch_attempted": epoch_attempted,
            "epoch_successful": epoch_successful,
            "epoch_skipped": epoch_skipped,
            "train_loss": epoch_loss,
            "val_mean_macro_f1": None if full_data else score,
            "is_best": is_best,
            **({f"val_{k}": v for k, v in flatten_metrics(va_res).items() if k != "mean_macro_f1"} if va_res is not None else {}),
            "seconds": time.time() - t0
        })
        if full_data:
            log.info(
                "epoch %d (step %d/%d, attempted %d, skipped %d) "
                "loss %.4f [FULL-DATA FIT] (%.1fs)",
                epoch + 1, global_step, total_steps, attempted_batches,
                skipped_updates, epoch_loss, history[-1]["seconds"],
            )
        else:
            log.info(
                "epoch %d (step %d/%d, attempted %d, skipped %d)  loss %.4f  val mean macro-F1 %.4f %s (%.1fs)",
                epoch + 1, global_step, total_steps, attempted_batches, skipped_updates, epoch_loss, score,
                "[BEST]" if is_best else "", history[-1]["seconds"]
            )

        if global_step >= total_steps:
            break
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    if full_data:
        # All rows were used for fitting. Do not evaluate an empty partition
        # or emit misleading validation/test metrics.
        results = {
            "full_data_training": True,
            "train_examples": int(len(tr_df)),
            "total_steps": int(global_step),
            "successful_updates": int(global_step),
            "attempted_batches": int(attempted_batches),
            "skipped_updates": int(skipped_updates),
            "steps_budget": int(total_steps),
            "eval_checkpoint": str(run_dir / "model.pt"),
            "eval_checkpoint_type": "last_full_data_checkpoint",
            "pretrained_vocab": getattr(encoder, "pretrained_vocab_stats", None),
            "device": str(device),
            "n_params": sum(p.numel() for p in model.parameters()),
            "label_source": df["label_source"].iloc[0] if "label_source" in df and len(df) else None,
            "class_balancing": balancing_meta,
            "encoder_name": str(cfg.model.encoder_name),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "seed": int(cfg.get("seed", 42)),
        }
        (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return results

    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device)["state_dict"])
    results = {
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_val_score": best,
        "total_steps": global_step,
        "successful_updates": global_step,
        "attempted_batches": attempted_batches,
        "skipped_updates": skipped_updates,
        "steps_budget": total_steps,
        "eval_checkpoint": str(run_dir / "model.pt"),
        "eval_checkpoint_type": "best_val_macro_f1",
        "pretrained_vocab": getattr(encoder, "pretrained_vocab_stats", None),
        "device": str(device),
        "n_params": sum(p.numel() for p in model.parameters()),
        "label_source": df["label_source"].iloc[0] if "label_source" in df and len(df) else None,
        "class_balancing": balancing_meta,
        "encoder_name": str(cfg.model.encoder_name),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seed": int(cfg.get("seed", 42)),
    }
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
