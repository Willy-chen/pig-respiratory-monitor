"""
Nested Pig-Level Leave-One-Out cross-validation for AST-XGBoost.

Implements the protocol described in section "Nested LOOCV for Hyperparameter
Selection and Statistical Reporting" of the paper. For each of the 16 outer
LOOCV folds, an inner LOOCV over the 15 training pigs is run across a
108-cell coarsened grid of (alpha_N, alpha_A, t_N, t_A); the cell that
maximises inner aggregate Macro F1 is selected and applied to the held-out
pig. The outer-fold predictions are concatenated across all 16 folds before
computing the final aggregate metric.

Outputs:
  - per_segment_predictions.csv : per-segment (pig, true, pred, prob_0/1/2, alpha, t)
  - fold_choices.csv            : inner-CV-chosen (alpha, t) per outer fold
  - per_pig_f1.csv              : per-pig aggregate Macro F1
  - summary.txt                 : aggregate Macro F1 + classification report

Usage:
    python run_nested_cv.py \\
        --features data/features/features_3layer_mean.pkl \\
        --out-dir  results/nested_cv_repro
"""
import argparse, os, pickle, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Coarsened grid (3 x 3 x 3 x 4 = 108 cells)
NORMAL_MULTS   = [2, 3, 4]
ABNORMAL_MULTS = [3, 5, 7]
TN_CANDS       = [0.6, 0.7, 0.8]
TA_CANDS       = [0.20, 0.25, 0.30, 0.35]

XGB_PARAMS = dict(objective='multi:softprob', num_class=3, max_depth=4,
                  eta=0.1, verbosity=0, nthread=8)
NUM_ROUND  = 100


def class_weights(y_train, n_mult, a_mult):
    """Inverse class-frequency weighting with extra (alpha_N, alpha_A) multipliers."""
    n_total = len(y_train)
    weight_map = {}
    for cls in np.unique(y_train):
        n_c = np.sum(y_train == cls)
        w = n_total / (3 * n_c)
        if cls == 1:
            w *= n_mult
        elif cls == 2:
            w *= a_mult
        weight_map[cls] = w
    return np.array([weight_map[label] for label in y_train])


def train_predict(X_train, y_train, X_test, n_mult, a_mult):
    weights = class_weights(y_train, n_mult, a_mult)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    dtest  = xgb.DMatrix(X_test)
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_ROUND)
    return booster.predict(dtest)


def apply_thresholds(probs, t_n, t_a):
    """Hierarchical asymmetric thresholding: prefer Abnormal > Normal > No-Breathing."""
    out = np.empty(len(probs), dtype=int)
    for i, p in enumerate(probs):
        if p[2] >= t_a:
            out[i] = 2
        elif p[1] >= t_n:
            out[i] = 1
        else:
            out[i] = 0
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', required=True,
                        help='Path to pickled (X, y, groups) AST embeddings.')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f">>> Loading cached features from {args.features}")
    with open(args.features, 'rb') as f:
        X, y, groups = pickle.load(f)
    y      = np.array(y)
    groups = np.array(groups)
    pigs   = np.unique(groups)
    print(f"    {len(X)} segments, {len(pigs)} pigs, X.shape={np.shape(X)}")

    rows, fold_choices = [], []
    t0 = time.time()

    for outer_idx, test_pig in enumerate(tqdm(pigs, desc="Outer LOOCV")):
        outer_test_mask  = (groups == test_pig)
        outer_train_mask = ~outer_test_mask
        train_pigs = [p for p in pigs if p != test_pig]

        # Inner LOOCV grid search on the 15 training pigs
        best_f1, best_cfg = -1.0, None
        for nm in NORMAL_MULTS:
            for am in ABNORMAL_MULTS:
                inner_probs, inner_targets = [], []
                for inner_test_pig in train_pigs:
                    inner_test_mask  = (groups == inner_test_pig)
                    inner_train_mask = outer_train_mask & ~inner_test_mask
                    probs = train_predict(X[inner_train_mask], y[inner_train_mask],
                                          X[inner_test_mask],  nm, am)
                    inner_probs.append(probs)
                    inner_targets.append(y[inner_test_mask])
                inner_probs   = np.concatenate(inner_probs)
                inner_targets = np.concatenate(inner_targets)
                for tn in TN_CANDS:
                    for ta in TA_CANDS:
                        f1 = f1_score(inner_targets,
                                      apply_thresholds(inner_probs, tn, ta),
                                      average='macro')
                        if f1 > best_f1:
                            best_f1, best_cfg = f1, (nm, am, tn, ta)

        nm_s, am_s, tn_s, ta_s = best_cfg
        fold_choices.append({'fold_idx': outer_idx, 'test_pig': test_pig,
                             'inner_f1': best_f1, 'n_mult': nm_s, 'a_mult': am_s,
                             't_n': tn_s, 't_a': ta_s})

        # Outer-fold evaluation with the chosen (alpha, t)
        outer_probs = train_predict(X[outer_train_mask], y[outer_train_mask],
                                    X[outer_test_mask], nm_s, am_s)
        outer_preds = apply_thresholds(outer_probs, tn_s, ta_s)
        seg_idx = np.where(outer_test_mask)[0]
        for i, si in enumerate(seg_idx):
            rows.append({
                'pig': test_pig, 'segment_idx': int(si),
                'true': int(y[si]), 'pred': int(outer_preds[i]),
                'prob_0': float(outer_probs[i, 0]),
                'prob_1': float(outer_probs[i, 1]),
                'prob_2': float(outer_probs[i, 2]),
                'n_mult': nm_s, 'a_mult': am_s, 't_n': tn_s, 't_a': ta_s,
            })
        elapsed = time.time() - t0
        print(f"    fold {outer_idx+1}/{len(pigs)}  pig={test_pig}  "
              f"cfg=({nm_s},{am_s},{tn_s},{ta_s})  inner_F1={best_f1:.4f}  "
              f"elapsed={elapsed:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'per_segment_predictions.csv'), index=False)
    pd.DataFrame(fold_choices).to_csv(
        os.path.join(args.out_dir, 'fold_choices.csv'), index=False)

    agg_f1 = f1_score(df['true'], df['pred'], average='macro')
    rep = classification_report(
        df['true'], df['pred'],
        target_names=['No-Breathing', 'Normal', 'Abnormal'], labels=[0, 1, 2])
    print(f"\n>>> Nested-CV Aggregate Macro F1: {agg_f1:.4f}\n{rep}")
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write(f"Nested-CV Aggregate Macro F1: {agg_f1:.4f}\n\n{rep}\n")

    per_pig = [{'pig': p,
                'n':   int((df['pig'] == p).sum()),
                'macro_f1': f1_score(df[df['pig'] == p]['true'],
                                     df[df['pig'] == p]['pred'], average='macro')}
               for p in pigs]
    pd.DataFrame(per_pig).to_csv(
        os.path.join(args.out_dir, 'per_pig_f1.csv'), index=False)
    print(f">>> Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
