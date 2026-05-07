"""
Convenience loader for the AST-XGBoost pig respiratory dataset.

Three things are exposed:

    load_dataset(data_dir)          -> (X, y, pig_ids, label_names)
    iter_loocv_folds(data_dir)      -> yields (fold_idx, test_pig, train_idx, test_idx)
    fold_data(data_dir, fold_idx)   -> (X_train, y_train, X_test, y_test, train_pigs, test_pig)

The dataset folder structure is:

    data/
      features.npy            # (1553, 768) float32 AST embeddings
      segments.csv            # one row per segment: segment_idx, pig_id, label, label_name
      loocv_folds.csv         # one row per outer fold
      splits/fold_NN/         # per-fold integer indices into features.npy
        train_idx.npy
        test_idx.npy
        test_pig.txt

Quick example:

    from data.load import iter_loocv_folds, fold_data
    for fold_idx, test_pig, train_idx, test_idx in iter_loocv_folds('data'):
        Xtr, ytr, Xte, yte, tr_pigs, te_pig = fold_data('data', fold_idx)
        # train your classifier on (Xtr, ytr); evaluate on (Xte, yte)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

LABEL_NAMES = ('No-Breathing', 'Normal', 'Abnormal')


def _resolve(data_dir: str) -> str:
    if os.path.isabs(data_dir):
        return data_dir
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', data_dir)


def load_dataset(data_dir: str = 'data'):
    """Load the full dataset.

    Returns
    -------
    X : (N, 768) float32 ndarray
    y : (N,) int64 ndarray, values in {0, 1, 2}
    pig_ids : (N,) object ndarray (one filename per segment, used as the LOOCV grouping key)
    label_names : tuple[str, ...] = ('No-Breathing', 'Normal', 'Abnormal')
    """
    root = _resolve(data_dir)
    X = np.load(os.path.join(root, 'features.npy'))
    seg = pd.read_csv(os.path.join(root, 'segments.csv'))
    y = seg['label'].to_numpy(dtype=np.int64)
    pig_ids = seg['pig_id'].to_numpy()
    return X, y, pig_ids, LABEL_NAMES


def iter_loocv_folds(data_dir: str = 'data'):
    """Yield (fold_idx, test_pig, train_idx, test_idx) for each of the 16 outer folds."""
    root = _resolve(data_dir)
    folds = pd.read_csv(os.path.join(root, 'loocv_folds.csv')).sort_values('fold_idx')
    for _, row in folds.iterrows():
        fold_idx = int(row['fold_idx'])
        sub = os.path.join(root, 'splits', f'fold_{fold_idx:02d}')
        train_idx = np.load(os.path.join(sub, 'train_idx.npy'))
        test_idx  = np.load(os.path.join(sub, 'test_idx.npy'))
        yield fold_idx, row['test_pig'], train_idx, test_idx


def fold_data(data_dir: str, fold_idx: int):
    """Return all arrays for a single outer LOOCV fold."""
    root = _resolve(data_dir)
    X, y, pig_ids, _ = load_dataset(data_dir)
    sub = os.path.join(root, 'splits', f'fold_{int(fold_idx):02d}')
    train_idx = np.load(os.path.join(sub, 'train_idx.npy'))
    test_idx  = np.load(os.path.join(sub, 'test_idx.npy'))
    with open(os.path.join(sub, 'test_pig.txt')) as f:
        test_pig = f.read().strip()
    train_pigs = np.unique(pig_ids[train_idx]).tolist()
    return (X[train_idx], y[train_idx],
            X[test_idx],  y[test_idx],
            train_pigs, test_pig)


if __name__ == '__main__':
    X, y, pig_ids, labels = load_dataset('data')
    print(f"X: {X.shape} {X.dtype}")
    print(f"y class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"unique pigs: {len(np.unique(pig_ids))}")
    n_folds = 0
    for fold_idx, test_pig, train_idx, test_idx in iter_loocv_folds('data'):
        n_folds += 1
        print(f"  fold {fold_idx:2d} | test_pig={test_pig:<20s} | "
              f"train n={len(train_idx)} | test n={len(test_idx)}")
    print(f"total folds: {n_folds}")
