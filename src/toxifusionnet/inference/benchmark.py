"""Latency / size benchmark incl. dynamic int8 quantisation (deployment objective of the project)."""
from __future__ import annotations

import copy
import io
import time

import numpy as np
import torch


def _size_mb(model) -> float:
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def benchmark(pipeline, texts: list[str], batch_sizes=(1, 8, 32), repeats: int = 5, quantize: bool = True) -> list[dict]:
    variants = {"fp32": pipeline}
    if quantize:
        try:
            from copy import copy as shallow
            q = shallow(pipeline)
            q.model = torch.ao.quantization.quantize_dynamic(copy.deepcopy(pipeline.model).cpu(), {torch.nn.Linear}, dtype=torch.qint8)
            variants["int8_dynamic"] = q
        except Exception as e:  # pragma: no cover
            print("quantisation unavailable:", e)
    rows = []
    for name, pipe in variants.items():
        size = _size_mb(pipe.model)
        for bs in batch_sizes:
            batch = (texts * (bs // max(len(texts), 1) + 1))[:bs]
            pipe.predict_proba(batch)  # warm-up
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter(); pipe.predict_proba(batch, batch_size=bs); ts.append(time.perf_counter() - t0)
            ms = float(np.median(ts) * 1000)
            rows.append({"model": name, "batch_size": bs, "latency_ms_per_batch": round(ms, 2),
                         "ms_per_text": round(ms / bs, 2), "texts_per_sec": round(bs / (ms / 1000), 1), "size_mb": round(size, 2)})
    return rows
