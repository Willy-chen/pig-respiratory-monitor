#!/usr/bin/env bash
# Reproduce the headline AST-XGBoost result from cached AST embeddings.
# Runs entirely on CPU; no GPU, no raw audio, no AST checkpoint required.
# Expected wall-clock: ~60 minutes (mostly inner-LOOCV XGBoost fits).

set -euo pipefail

PYTHON=${PYTHON:-python3}
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CACHE="$ROOT/data/features/features_3layer_mean.pkl"
OUT="$ROOT/results"

echo "==> Verifying cached AST features..."
if [ ! -f "$CACHE" ]; then
    echo "ERROR: cached features not found at $CACHE" >&2
    exit 1
fi
SIZE=$(du -h "$CACHE" | cut -f1)
echo "    Found: $CACHE ($SIZE)"

echo "==> Step 1/2: Nested LOOCV (~55 min on CPU)..."
$PYTHON src/nested_cv/run_nested_cv.py \
    --features "$CACHE" \
    --out-dir "$OUT/nested_cv_repro"

echo "==> Step 2/2: Bootstrap CI + paired-permutation tests + per-pig boxplot..."
$PYTHON src/nested_cv/run_analysis_final.py \
    --nested "$OUT/nested_cv_repro/per_segment_predictions.csv" \
    --yin    "$ROOT/results/baseline_probs/yin_2021_probs.csv" \
    --dorr   "$ROOT/results/baseline_probs/dorr_2026_probs.csv" \
    --out-dir "$OUT/analysis_repro"

echo
echo "===================================================================="
echo "Done. Artifacts written to:"
echo "  $OUT/nested_cv_repro/per_segment_predictions.csv"
echo "  $OUT/nested_cv_repro/fold_choices.csv"
echo "  $OUT/nested_cv_repro/per_pig_f1.csv"
echo "  $OUT/nested_cv_repro/summary.txt"
echo "  $OUT/analysis_repro/summary_final.json"
echo "  $OUT/analysis_repro/per_pig_f1_boxplot.png"
echo
echo "Compare against reference outputs in:"
echo "  $OUT/nested_cv/"
echo "  $OUT/analysis/"
echo "===================================================================="
