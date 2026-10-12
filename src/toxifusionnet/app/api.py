"""Minimal REST service:  TOXIFUSION_RUN_DIR=runs/exp uvicorn toxifusionnet.app.api:app --port 8000"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..inference.pipeline import ToxiFusionPipeline

app = FastAPI(title="ToxiFusionNet", version="0.1.0",
              description="Sarcasm-aware hate speech detection for Hindi-English code-mixed text")


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)
    explain: bool = True


@lru_cache(maxsize=1)
def get_pipeline() -> ToxiFusionPipeline:
    run_dir = os.environ.get("TOXIFUSION_RUN_DIR")
    if not run_dir:
        raise RuntimeError("set TOXIFUSION_RUN_DIR to a trained run directory")
    return ToxiFusionPipeline.from_run(run_dir, device=os.environ.get("TOXIFUSION_DEVICE", "cpu"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return {"results": get_pipeline().predict(req.texts, explain=req.explain)}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
