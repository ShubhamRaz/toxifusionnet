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
    cv_e1d)
        run_cv_exp configs/exp1d_multibully_text_weighted.yaml results/v3_journal_final/e1d_text_weighted
        ;;
    cv_e1d_cleaned)
        echo "Launching E1d on Curated Cleaned MultiBully dataset..."
        python -m toxifusionnet.cli cv --config configs/exp1d_multibully_cleaned.yaml \
            --data data/cleaned/multibully_cleaned.csv \
            --folds-cache data/cleaned/multibully_v2_cv_folds.csv \
            --out-dir results/v3_journal_final/e1d_cleaned \
            --folds 5 \
            --seed 42
        ;;
    cv_e6b)
        echo "Launching E6b with auxiliary HOT gold (train-only)..."
        python -m toxifusionnet.cli cv --config configs/exp6b_combined_text_gold_weighted.yaml \
            --data data/processed/multibully.csv \
            --aux-data data/processed/hot_gold_only.csv \
            --folds-cache data/processed/multibully_v2_cv_folds.csv \
            --out-dir results/v3_journal_final/e6b_text_gold_weighted \
            --folds 5 \
            --seed 42
        ;;
    cv_e6b_cleaned)
        echo "Launching E6b on Curated Cleaned Combined MultiBully + HOT dataset..."
        python -m toxifusionnet.cli cv --config configs/exp6b_combined_cleaned.yaml \
            --data data/cleaned/multibully_cleaned.csv \
            --aux-data data/cleaned/hot_cleaned.csv \
            --folds-cache data/cleaned/multibully_v2_cv_folds.csv \
            --out-dir results/v3_journal_final/e6b_cleaned \
            --folds 5 \
            --seed 42
        ;;
    validate)
        echo "Running data integrity validation suite..."
        python scripts/validate_cleaned_datasets.py
        pytest tests/test_data_integrity_audit.py tests/test_matched_cv_integrity.py
        ;;
    smoke)
        echo "Running pre-flight smoke test..."
        python -m toxifusionnet.cli train --config configs/smoke_test.yaml
        ;;
    cv_e1)
        run_cv_exp configs/exp1_multibully_text.yaml results/v2_evaluation/e1_text
        ;;
    cv_e3)
        run_cv_exp configs/exp3_multibully_clip.yaml results/v2_evaluation/e3_clip
        ;;
    cv_e5)
        run_cv_exp configs/exp5_multibully_balanced_loss.yaml results/v2_evaluation/e5_balanced_loss
        ;;
    cv_e6)
        run_cv_exp configs/exp6_combined_text.yaml results/v2_combined_dataset/e6_text
        ;;
    cv_e7)
        run_cv_exp configs/exp7_combined_clip.yaml results/v2_combined_dataset/e7_clip
        ;;
    cv_combined)
        echo "Running combined dataset 5-fold CV suite (E6, E7) sequentially..."
        run_cv_exp configs/exp6_combined_text.yaml results/v2_combined_dataset/e6_text
        run_cv_exp configs/exp7_combined_clip.yaml results/v2_combined_dataset/e7_clip
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
        echo "Running canonical weighted benchmark experiments (E1d, E6b)..."
        run_cv_exp configs/exp1d_multibully_text_weighted.yaml results/v3_journal_final/e1d_text_weighted
        python -m toxifusionnet.cli cv --config configs/exp6b_combined_text_gold_weighted.yaml \
            --data data/processed/multibully.csv \
            --aux-data data/processed/hot_gold_only.csv \
            --folds-cache data/processed/multibully_v2_cv_folds.csv \
            --out-dir results/v3_journal_final/e6b_text_gold_weighted \
            --folds 5 \
            --seed 42
        echo ""
        echo "All experiments completed successfully!"
        ;;
    *)
        echo "Usage: bash run_supercomputer.sh [cv_e1d|cv_e6b|validate|smoke|cv_e1|cv_e3|cv_e5|cv_all|all]"
        exit 1
        ;;
esac
