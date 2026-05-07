# Architecture & Pipeline Overview

End-to-end flow of the AST-XGBoost pipeline as deployed in the paper.

```
Raw stethoscope audio (private)
        |
        v
[1] Bandpass 50–3000 Hz (scipy.signal)
        |
        v
[2] Spectral-gating denoise (noisereduce, gate −20 dB)
        |
        v
[3] 10-second segmentation @ 16 kHz, mono
        |
        v
[4] Audio Spectrogram Transformer  (Gong et al., 2021)
        backbone:  ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        head:      classification head (3 classes)  ←— fine-tuned on the AST set (15 pigs)
        feature:   mean of last 3 hidden layers, mean-pooled over time → 768-d embedding
        |
        v
        (cached on disk: data/features.npy, 1553 × 768 float32)
        |
        v
[5] XGBoost classifier
        params: max_depth=4, eta=0.1, num_round=100, multi:softprob
        loss:   class-weighted cross-entropy
                weights:    base = N_total / (3 · N_c)
                            multiplied by α_N (Normal) and α_A (Abnormal)
        evaluation: Pig-Level Leave-One-Out cross-validation over 16 pigs
        hyperparams (α_N, α_A, t_N, t_A) chosen by NESTED inner LOOCV
                (see src/nested_cv/run_nested_cv.py)
        |
        v
[6] Asymmetric thresholding (hierarchical decision rule)
        if   prob[Abnormal]    >= t_A   → predict Abnormal
        elif prob[Normal]      >= t_N   → predict Normal
        else                            → predict No-Breathing
        |
        v
3-class label: {No-Breathing, Normal, Abnormal}
```

## Where each step lives in the repo

| Step | Code |
|------|------|
| 1 + 2 + 3 | `src/pipeline/data_utils.py` (preprocessing + segmentation) |
| 4 (training) | `src/ast_finetune/1_finetune_ast.py` |
| 4 (feature extraction) | `src/pipeline/run_ultimate_search.py::extract_optimal_features` |
| 5 (leaky aggregate grid search, kept as analytical surface only) | `src/pipeline/run_ultimate_search.py` |
| 5 (paper headline; nested LOOCV) | `src/nested_cv/run_nested_cv.py` |
| 6 | `src/nested_cv/run_nested_cv.py::apply_thresholds` |
| Bootstrap CIs + permutation tests + boxplot | `src/nested_cv/run_analysis_final.py` |
| 9 baseline models + LOOCV training driver | `src/baselines/` |
| 12 downstream-classifier head benchmark | `src/classifiers/` |
| SNR robustness + few-shot scaling | `src/stress_tests/` |

## Why the headline number is 0.8713 and not 0.8894

The original `run_ultimate_search.py` does a **post-hoc grid search over (α_N, α_A) × (t_N, t_A) on the same aggregate LOOCV predictions used to compute the reported Macro F1**. This selects the lucky upper tail of a noisy surface and is a textbook test-set leak; under that procedure the reported number is 0.8894.

`src/nested_cv/run_nested_cv.py` replaces that with a fully nested protocol: each outer-fold's (α, t) choice comes from an inner LOOCV over the 15 *training* pigs. Under this procedure the headline is 0.8713 — the 1.81 pp drop is the exact cost of removing the leak.

The leaky surface is preserved (in Appendix B of the paper and in `results/leaky_grid_best_config.txt`) only as an analytical reference, not as a metric for cross-paper comparison.

## Computational footprint

Single Linux box, conda env `ml`, torch 2.x + transformers 4.38 + xgboost 2.0:

| Step | One-time cost | Per-run cost |
|------|---------------|--------------|
| AST fine-tuning (5 epochs, 1 GPU) | ~3 GPU-h | — |
| Feature extraction (1553 segments) | ~5 min on 1 GPU | — |
| XGBoost LOOCV (single (α, t)) | — | ~30 s on 1 CPU |
| Nested LOOCV (108 cells × 16 outer × 15 inner) | — | ~60–75 min on 1 CPU |
| AlexNet baseline (10 epochs × 16 folds) | ~3 min on 1 GPU | — |
| BEATs baseline (10 epochs × 16 folds) | ~40 min on 1 GPU | — |
| SNR stress test (4 noise types × 4 SNRs × 9 baselines + AST-XGB) | ~24 GPU-h | — |

`reproduce.sh` reuses the cached embeddings so it runs only the nested LOOCV + analysis (~70 min CPU).
