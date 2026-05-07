#!/bin/bash
# Run all 9 baselines on the cached AST features.
# Override PYTHON_CMD to use a specific interpreter, e.g.:
#   PYTHON_CMD=~/miniconda3/envs/ml/bin/python bash run_all.sh

PYTHON_CMD="${PYTHON_CMD:-python3}"
BASELINES=(
    "yin_2021"
    "shen_2022"
    "wu_2022"
    "hou_2024"
    "sheikh_2024"
    "dorr_2026"
    "wang_2026"
    "mdpi_2026"
    "nithin_2026"
)

echo "----------------------------------------------------------"
echo "Starting Pig Cough Baseline Benchmarks"
echo "----------------------------------------------------------"

for baseline in "${BASELINES[@]}"; do
    echo ">>> Running Baseline: $baseline"
    $PYTHON_CMD baselines/$baseline/run.py
    if [ $? -eq 0 ]; then
        echo ">>> $baseline: Success"
    else
        echo ">>> $baseline: FAILED"
    fi
    echo "----------------------------------------------------------"
done

echo ">>> Generating Comparison Summary"
$PYTHON_CMD compare_results.py
