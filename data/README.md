# Pig Respiratory Audio Dataset — Pre-computed AST Features

This directory contains everything needed to **reproduce the AST-XGBoost classification results from the paper without access to the raw audio**.

## What is here

| Path | Size | Contents |
|------|------|----------|
| `features.npy` | 4.8 MB | (1553, 768) float32 AST embeddings, mean-pooled over the last 3 transformer layers of the fine-tuned AST. One row per 10-second audio segment. |
| `segments.csv` | ~50 KB | One row per segment: `segment_idx`, `pig_id` (filename used as the LOOCV grouping key), `label` (0/1/2), `label_name`. |
| `loocv_folds.csv` | <2 KB | One row per outer LOOCV fold (0-15): `fold_idx`, `test_pig`, `n_train_segments`, `n_test_segments`, per-class test counts. |
| `splits/fold_NN/` | <100 KB | Per-fold integer indices into `features.npy`: `train_idx.npy`, `test_idx.npy`, `test_pig.txt`. |
| `features/features_3layer_mean.pkl` | 4.8 MB | Original pickle (kept for backwards compat with code that imports `features_3layer_mean.pkl` directly). Same data as `features.npy` but as a `(X, y, groups)` tuple. |
| `load.py` | — | Python loader exposing `load_dataset(...)`, `iter_loocv_folds(...)`, `fold_data(...)`. |

## What the labels mean

| label | label_name | description |
|-------|------------|-------------|
| 0 | No-Breathing | environmental noise / silence (no detectable respiratory signal) |
| 1 | Normal | regular respiratory sounds without audible abnormality |
| 2 | Abnormal | wheezing, crackles, or laboured breathing indicative of respiratory distress |

Class counts in the full dataset:

| label | count | share |
|-------|-------|-------|
| No-Breathing | 533 | 34.3 % |
| Normal | 588 | 37.9 % |
| Abnormal | 432 | 27.8 % |
| **Total** | **1 553** | |

## What the LOOCV protocol means

Each segment carries a `pig_id` that corresponds to the source audio recording. Each `pig_id` is one unique pig — recordings were collected with one stethoscope per animal during a single session, so the file is the natural subject identifier.

We use Pig-Level Leave-One-Out Cross-Validation: 16 outer folds, where each outer fold holds out one pig's segments as the test set and trains on the remaining 15 pigs. This is **strictly subject-level evaluation** (no segment from the held-out pig appears anywhere in training).

The paper's headline number is the aggregate Macro F1 over the concatenation of all 16 outer-fold predictions, where each outer fold uses its own (alpha_N, alpha_A, t_N, t_A) chosen by an inner LOOCV on the remaining 15 training pigs (see `src/nested_cv/run_nested_cv.py`).

## How to load

### Python

```python
from data.load import load_dataset, iter_loocv_folds, fold_data

# Full dataset
X, y, pig_ids, label_names = load_dataset('data')
print(X.shape)              # (1553, 768)
print(label_names)          # ('No-Breathing', 'Normal', 'Abnormal')

# Iterate over the 16 LOOCV folds
for fold_idx, test_pig, train_idx, test_idx in iter_loocv_folds('data'):
    X_train = X[train_idx]
    X_test  = X[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]
    # train your classifier on (X_train, y_train); evaluate on (X_test, y_test)

# Or get one fold's data directly
X_train, y_train, X_test, y_test, train_pigs, test_pig = fold_data('data', fold_idx=0)
```

### Numpy / Pandas (no helper)

```python
import numpy as np
import pandas as pd

X       = np.load('data/features.npy')
seg     = pd.read_csv('data/segments.csv')
y       = seg['label'].to_numpy()
pig_ids = seg['pig_id'].to_numpy()
folds   = pd.read_csv('data/loocv_folds.csv')

for fold_idx in range(16):
    test_pig = folds.iloc[fold_idx]['test_pig']
    test_idx  = np.load(f'data/splits/fold_{fold_idx:02d}/test_idx.npy')
    train_idx = np.load(f'data/splits/fold_{fold_idx:02d}/train_idx.npy')
    ...
```

### R / MATLAB / etc.

`features.npy` and the `*.npy` indices follow the standard NumPy `.npy` v1.0 binary format (little-endian float32). They can be loaded by `numpy.load` from any language with NumPy bindings, or by reading the 80-byte header + raw float32s. The CSV files are plain UTF-8.

## What is **not** here

- **Raw audio (`.wav` files)** — confidential to the participating commercial pig farm; access may be requested from Prof. Chao-Wei Huang (`cwhuang@mail.npust.edu.tw`) at the Animal Nutrigenomics Laboratory, NPUST, under a data-use agreement.
- **Strong-label timestamps** — derived from the raw audio annotations; not redistributed for the same reason.
- **The fine-tuned AST checkpoint** that produced these embeddings — available on Hugging Face: <https://huggingface.co/willychenwii/pig-condition-ast-finetuned>. Anyone who downloads the checkpoint and obtains the raw audio under a DUA can regenerate `features.npy` byte-for-byte using `src/pipeline/run_ultimate_search.py`'s feature-extraction block.

## Citation & archival DOI

This dataset is permanently archived at Zenodo:

> [10.5281/zenodo.20084290](https://doi.org/10.5281/zenodo.20084290)

When citing the dataset specifically (separately from the paper), use the Zenodo record. The Zenodo and GitHub copies of  and  are byte-identical for tag .

## Provenance

These embeddings are the **mean of the last 3 hidden layers** of the AST checkpoint released as `willychenwii/pig-condition-ast-finetuned`, computed on 10-second 16 kHz mono segments after a 50–3000 Hz bandpass + spectral-gating denoise (`noisereduce` library, gate threshold –20 dB). Segment IDs are stable across re-runs as long as the same audio source files and the same data-loader RNG seed (42) are used; the segment ordering matches the order produced by `src/pipeline/data_utils.py::create_study_split` on the original dataset.

## Citation

If you use this dataset, please cite our paper (see `../README.md`).
