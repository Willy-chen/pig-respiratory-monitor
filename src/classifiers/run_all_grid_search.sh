#!/bin/bash
# Master Grid Search execution script for 20260330
set -e

# Ensure conda env is used
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$BASE_DIR"

# Eligible models for fast grid search (Logistic Regression and GPC are skipped due to speed)
models=("xgboost" "catboost" "lightgbm" "rf" "svm" "knn" "linear" "mlp2" "mlp3")

echo "=========================================================="
echo ">>> Starting MASSIVE JOINT GRID SEARCH for all models"
echo ">>> Range: Multipliers [1-10], Thresholds [0.1-0.9]"
echo "=========================================================="

for model in "${models[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo ">>> Grid Search: $model"
    echo "----------------------------------------------------------"
    python grid_search.py --model "$model"
done

echo ""
echo "=========================================================="
echo ">>> All grid searches completed."
echo ">>> Results are stored in classifiers/<model>/grid_search_best.txt"
echo "=========================================================="
