$ErrorActionPreference = "Stop"
Set-Location "c:/Users/SHUBHAM RAJ/Desktop/Others/Toxic/toxifusionnet/toxifusionnet"
.\.venv\Scripts\activate
$env:PYTHONPATH="src"

$CFG="configs/multibully.yaml"
$RAW="data/raw/multibully.csv"
$DATA="data/processed/multibully.csv"
$OUT="runs/multibully"
$RUN="$OUT/full/seed42"

Write-Host "Preparing data..." -ForegroundColor Green
python -m toxifusionnet.cli prepare --schema configs/schema_multibully.yaml --raw $RAW --out $DATA
python -m toxifusionnet.cli describe --data $DATA --out results/data_report.md
python -m toxifusionnet.cli baselines --data $DATA --out results/baselines.csv

Write-Host "Training models (this will take a while, you will see progress per epoch)..." -ForegroundColor Green
python -m toxifusionnet.cli ablate --config $CFG --override model.encoder_name=google/muril-base-cased --out-dir $OUT --seeds 42

Write-Host "Running robustness checks..." -ForegroundColor Green
python -m toxifusionnet.cli robust --run-dir $RUN --data $DATA --device cuda

Write-Host "Generating explanations..." -ForegroundColor Green
python -m toxifusionnet.cli explain --run-dir $RUN --data $DATA --device cuda

Write-Host "Benchmarking speed..." -ForegroundColor Green
python -m toxifusionnet.cli bench --run-dir $RUN --device cpu

Write-Host "Generating figures..." -ForegroundColor Green
python -m toxifusionnet.cli figures --run-dir $RUN

Write-Host "Running Cross-Validation (this will train 5 more models)..." -ForegroundColor Green
python -m toxifusionnet.cli cv --config $CFG --override model.encoder_name=google/muril-base-cased --data $DATA --out-dir "$OUT/cv"

Write-Host "Generating final report..." -ForegroundColor Green
python -m toxifusionnet.cli report --run-dir $RUN --ablation-dir $OUT --baselines results/baselines.csv --data-report results/data_report.md --cv-dir "$OUT/cv" --out results/RESULTS.md

Write-Host "Done! Check results/RESULTS.md for the final metrics." -ForegroundColor Green
