#!/usr/bin/env bash
# Full study on the MultiBully text annotations, one command:
#   bash scripts/run_final.sh data/raw/<MultiBully file>.csv|.xlsx [hf-encoder-name]
# GPU + pre-trained encoder (recommended):     bash scripts/run_final.sh data/raw/multibully.csv google/muril-base-cased
# CPU only, small encoder + GloVe vectors:     CFG=configs/multibully_cpu.yaml bash scripts/run_final.sh data/raw/multibully.csv
# Knobs (env): CFG  DEVICE  SEEDS="42 43 44"  EXTRA="train.epochs=10 ..."  CV=1 (5-fold cross-validation)  OUT=runs/multibully
set -euo pipefail
RAW=${1:?usage: scripts/run_final.sh <path to MultiBully .csv/.xlsx> [hf-encoder-name]}
ENC=${2:-}
CFG=${CFG:-configs/multibully.yaml}
DEVICE=${DEVICE:-cpu}
SEEDS=${SEEDS:-"42 43 44"}
EXTRA=${EXTRA:-}
OUT=${OUT:-runs/multibully}
SCHEMA=${SCHEMA:-configs/schema_multibully.yaml}
DATA=${DATA:-data/processed/multibully.csv}
export PYTHONPATH=src
CLI="python -m toxifusionnet.cli"
RUN=$OUT/full/seed${SEEDS%% *}
ENC_OVR=(); [ -n "$ENC" ] && ENC_OVR=(model.encoder_name="$ENC")

$CLI prepare   --schema "$SCHEMA" --raw "$RAW" --out "$DATA"
$CLI describe  --data "$DATA" --out results/data_report.md
$CLI baselines --data "$DATA" --out results/baselines.csv
$CLI ablate    --config "$CFG" --override "${ENC_OVR[@]}" $EXTRA --out-dir "$OUT" --seeds $SEEDS
$CLI robust    --run-dir "$RUN" --data "$DATA" --device "$DEVICE"
$CLI explain   --run-dir "$RUN" --data "$DATA" --device "$DEVICE"
$CLI bench     --run-dir "$RUN" --device cpu
$CLI figures   --run-dir "$RUN"
CVARG=()
if [ "${CV:-0}" = "1" ]; then
  $CLI cv --config "$CFG" --override "${ENC_OVR[@]}" $EXTRA --data "$DATA" --out-dir "$OUT/cv"
  CVARG=(--cv-dir "$OUT/cv")
fi
$CLI report    --run-dir "$RUN" --ablation-dir "$OUT" --baselines results/baselines.csv \
               --data-report results/data_report.md "${CVARG[@]}" --out results/RESULTS.md
echo; echo "Done. Read results/RESULTS.md  (figures in $RUN/figures, model in $RUN)"
