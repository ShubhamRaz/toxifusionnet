# One-click launch script to train ToxiFusionNet from scratch on the unified Tri-Dataset
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " ToxiFusionNet: End-to-End Training from Scratch          " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Activate Virtual Environment
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "[1/4] Activating virtual environment..." -ForegroundColor Green
    .\.venv\Scripts\Activate.ps1
} elseif (Test-Path ".\.venv\Scripts\activate.bat") {
    Write-Host "[1/4] Activating virtual environment..." -ForegroundColor Green
    cmd /c .\.venv\Scripts\activate.bat
} else {
    Write-Host "[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -e .
}

$env:PYTHONPATH="src"

# 2. Check Hardware / Device
Write-Host "[2/4] Detecting compute hardware..." -ForegroundColor Green
python -c "import torch; dev = 'CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (No GPU detected)'; print('Compute Device:', dev)"

# 3. Train from Scratch
$CONFIG = "configs/combined_all_3_datasets.yaml"
$OUT = "runs/exp_tri_dataset_xlmr"

Write-Host "[3/4] Launching training with config: $CONFIG..." -ForegroundColor Green
Write-Host "      Trained checkpoint will be saved to: $OUT" -ForegroundColor Gray
python -m toxifusionnet.cli train --config $CONFIG

# 4. Verify Model with Sample Predictions
Write-Host "[4/4] Verifying trained model with sample predictions..." -ForegroundColor Green
python -m toxifusionnet.cli predict `
    --run-dir $OUT `
    --text "Congrats bro, strict diet ke beech spicy golgappe ne prove kar diya ki hope pe world kayam hai" `
           "Saale ye lobby ke log itna disgusting aur pathetic harkat karta hai" `
           "Aloo paratha bahut tasty tha yaar"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Training & Verification Complete!                        " -ForegroundColor Cyan
Write-Host " Master Report: STAGE_3_FINAL_REPORT.md                   " -ForegroundColor Cyan
Write-Host " Checkpoints & Metrics: $OUT                              " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
