#!/bin/bash
# Master execution script for the 20260330 Classifier Exploration
set -e

# Ensure conda env is used
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$BASE_DIR"

models=("xgboost" "logreg" "rf" "svm" "knn" "gpc" "catboost" "lightgbm" "linear" "mlp2" "mlp3")

for model in "${models[@]}"; do
    echo "=========================================================="
    echo ">>> Running Classifier: $model"
    echo "=========================================================="
    cd "classifiers/$model"
    python run.py
    cd "$BASE_DIR"
done

echo "=========================================================="
echo ">>> All models finished. Aggregating results..."
echo "=========================================================="
python compare_results.py
