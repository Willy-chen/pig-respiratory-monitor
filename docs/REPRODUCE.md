# Reproduction Instructions

Three reproduction tiers, in increasing order of cost and prerequisites.

---

## Tier 1 — Headline number from cached embeddings (~60 min CPU)

This is the fastest path and what `reproduce.sh` does. It re-derives the paper's headline AST-XGB nested-CV Macro F1 from the cached AST embeddings in `data/features.npy`.

**Prerequisites**: Python 3.10+, packages in `requirements.txt`.

```bash
# Code (developed openly):
git clone https://github.com/Willy-chen/pig-respiratory-monitor.git

# OR archived data + code snapshot (citable, immutable):
# https://doi.org/10.5281/zenodo.20084290
cd pig-respiratory-monitor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash reproduce.sh
```

Expected outputs in `results/nested_cv_repro/` and `results/analysis_repro/`:

| File | What |
|------|------|
| `per_segment_predictions.csv` | (1553 rows) outer-fold predictions, probabilities, chosen (α, t) per segment |
| `fold_choices.csv` | (16 rows) inner-CV-chosen (α_N, α_A, t_N, t_A) per outer fold |
| `per_pig_f1.csv` | (16 rows) per-pig aggregate Macro F1 |
| `summary.txt` | aggregate Macro F1 + per-class precision / recall / F1 |
| `summary_final.json` | bootstrap CIs + paired-permutation p-values |
| `per_pig_f1_boxplot.png` | the figure used in §4.4 of the paper |

**Reference values to compare against** (from `results/nested_cv/summary.txt` and `results/analysis/summary_final.json`):

```
AST-XGB nested-CV Macro F1     : 0.8713
                95 % CI by pigs : [0.7740, 0.9086]
AlexNet (argmax)         F1    : 0.8264   CI [0.6923, 0.8920]
BEATs   (argmax)         F1    : 0.8018   CI [0.6827, 0.8762]
AST-XGB vs AlexNet  : Δ = +0.0175,  p = 0.2773
AST-XGB vs BEATs    : Δ = +0.1107,  p = 0.0168
```

XGBoost is deterministic with respect to its built-in seed (sklearn default), so your nested-CV Macro F1 should match the reference to four decimal places. The bootstrap and permutation p-values use `np.random.default_rng(seed=42)` and are likewise deterministic.

---

## Tier 2 — Re-train AlexNet and BEATs baselines (~1 GPU-hour)

If you want to re-derive the per-segment soft-max probabilities in `results/baseline_probs/`:

**Additional prerequisites**:
1. Raw audio (private; see Tier 3 — the loaders in `src/baselines/data_loader.py` re-use `src/pipeline/data_utils.py::create_study_split` to recover the same 16-pig XGB set).
2. PyTorch with CUDA.
3. The official BEATs Iter3 + AS2M checkpoint at `src/baselines/pretrained_models/BEATs_iter3_plus_AS2M.pt`. Download from the upstream Microsoft BEATs release (the file is ~1.1 GB and is **not redistributed in this repo**); see <https://github.com/microsoft/unilm/tree/master/beats>.

```bash
# After raw audio + BEATs checkpoint are in place
python src/nested_cv/run_baseline_with_probs.py
```

This regenerates `results/baseline_probs/yin_2021_probs.csv` and `results/baseline_probs/dorr_2026_probs.csv`. Expected per-fold runtime: ~13 s per AlexNet fold, ~140 s per BEATs fold.

**Note on seed sensitivity**: AlexNet and BEATs training is stochastic (no explicit `torch.manual_seed`), so the per-segment probabilities will *not* be byte-identical across runs — single-seed differences of 4–6 pp in aggregate Macro F1 are typical with this small (16-pig) LOOCV. This is itself a finding of the paper and motivates the use of cluster-bootstrap CIs and paired-permutation tests rather than point-estimate comparisons.

---

## Tier 3 — Full pipeline from raw audio (~6 GPU-hours)

For complete end-to-end reproduction including AST fine-tuning + feature extraction.

**Additional prerequisites**:
1. Raw audio dataset under a data-use agreement; contact Prof. Chao-Wei Huang (`cwhuang@mail.npust.edu.tw`).
2. Strong-label TXT files (start/end/label triples per recording).
3. The pretrained AST checkpoint `MIT/ast-finetuned-audioset-10-10-0.4593` (auto-downloaded by `transformers` on first run).

Pipeline steps:

```bash
# 1. Fine-tune the AST on the 15-pig AST set
python src/ast_finetune/1_finetune_ast.py
#    Output: a checkpoint matching huggingface.co/willychenwii/pig-condition-ast-finetuned

# 2. Extract 768-d embeddings on the 16-pig XGB set
python src/pipeline/run_ultimate_search.py
#    Outputs: results/features_3layer_mean.pkl
#    (= same data as data/features.npy in this repo)

# 3. Run nested LOOCV from cached embeddings (Tier 1 step)
bash reproduce.sh
```

If you obtain the raw audio under a DUA, the cached embeddings in `data/features.npy` should be recoverable byte-for-byte from step 2 above as long as you use the same fine-tuned AST checkpoint and the same data-loader RNG seed (42).

---

## Other experiments (optional)

### SNR robustness stress test (Section 4.6 of the paper)

```bash
python src/stress_tests/run_snr.py
```

Requires raw audio + the ESC-50 dataset (<https://github.com/karolpiczak/ESC-50>) for the out-of-distribution noise tier.

### Few-shot scaling (Section 4.6)

```bash
python src/stress_tests/run_few_shot.py
```

### Downstream classifier-head benchmark (Section 4.5 / Appendix C)

```bash
cd src/classifiers
bash run_all.sh
```

Runs 12 different classification heads (Linear SVM, MLP-2, CatBoost, …) on the cached AST embeddings and writes `benchmark_summary.csv`.

---

## Troubleshooting

- **`features_3layer_mean.pkl` won't load**: the pickle is `(X, y, groups)` where `X` is `np.ndarray` shape `(1553, 768)` float32. If the pickle protocol fails on a newer Python, just use `data/features.npy` + `data/segments.csv` directly via `data/load.py`.
- **XGBoost numerical mismatch**: ensure `xgboost>=2.0`. XGBoost 1.x and 2.x have different default tree construction algorithms which can shift Macro F1 by ~0.001.
- **Nested CV is too slow on your machine**: pass `--threads N` (planned; for now edit `XGB_PARAMS['nthread']` in `src/nested_cv/run_nested_cv.py`).
