#!/bin/bash
set -e

echo "=== ToxiFusionNet Supercomputer Launch Script ==="

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ ! -d ".venv" ]; then
    echo "Creating virtual environment on Linux supercomputer..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ".[explain,serve,dev]"
fi

export PYTHONPATH="src"

# Optional: uncomment if compute nodes do not have outbound internet
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

EXP=${1:-all}

run_exp() {
    local cfg=$1
    echo ""
    echo "=========================================================="
    echo " Launching: $cfg"
    echo "=========================================================="
    python -m toxifusionnet.cli train --config "$cfg"
}

run_cv_exp() {
    local cfg=$1
    local out=$2
    local bs=${BATCH_SIZE:-64}
    echo ""
    echo "=========================================================="
    echo " Launching 5-Fold Grouped CV: $cfg -> $out (batch_size=$bs)"
    echo "=========================================================="
    python -m toxifusionnet.cli cv --config "$cfg" \
        --override train.batch_size=$bs \
        --data data/processed/multibully.csv \
        --folds-cache data/processed/multibully_v2_cv_folds.csv \
        --out-dir "$out" \
        --folds 5 \
        --seed 42
}

case "$EXP" in
    cv_e1)
        run_cv_exp configs/exp1_multibully_text.yaml results/v2_evaluation/e1_text
        ;;
    cv_e3)
        run_cv_exp configs/exp3_multibully_clip.yaml results/v2_evaluation/e3_clip
        ;;
    cv_e5)
        run_cv_exp configs/exp5_multibully_balanced_loss.yaml results/v2_evaluation/e5_balanced_loss
        ;;
    cv_all)
        echo "Running complete 5-fold CV suite (E1, E3, E5) sequentially..."
        run_cv_exp configs/exp1_multibully_text.yaml results/v2_evaluation/e1_text
        run_cv_exp configs/exp3_multibully_clip.yaml results/v2_evaluation/e3_clip
        run_cv_exp configs/exp5_multibully_balanced_loss.yaml results/v2_evaluation/e5_balanced_loss
        ;;
    exp1)
        run_exp configs/exp1_multibully_text.yaml
        ;;
    exp2)
        run_exp configs/exp2_multidataset_text.yaml
        ;;
    exp3)
        run_exp configs/exp3_multibully_clip.yaml
        ;;
    exp4)
        run_exp configs/exp4_multidataset_clip.yaml
        ;;
    all)
        echo "Running all 4 benchmark experiments sequentially..."
        run_exp configs/exp1_multibully_text.yaml
        run_exp configs/exp2_multidataset_text.yaml
        run_exp configs/exp3_multibully_clip.yaml
        run_exp configs/exp4_multidataset_clip.yaml
        echo ""
        echo "All 4 experiments completed successfully!"
        echo "Check metrics in runs/exp*/metrics.json"
        ;;
    *)
        echo "Usage: bash run_supercomputer.sh [cv_e1|cv_e3|cv_e5|cv_all|exp1|exp2|exp3|exp4|all]"
        exit 1
        ;;
esac
