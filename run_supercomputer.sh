#!/bin/bash
set -e

echo "=== ToxiFusionNet Supercomputer Launch Script ==="

# 1. Activate or create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment on Linux supercomputer..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Ensuring package dependencies are installed..."
pip install --upgrade pip
pip install -e ".[explain,serve,dev]"

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

case "$EXP" in
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
        echo "Usage: bash run_supercomputer.sh [exp1|exp2|exp3|exp4|all]"
        exit 1
        ;;
esac
