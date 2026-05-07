"""
M1 + M5 paper analysis: cluster-bootstrap CIs by pigs, paired-permutation
tests vs argmax baselines, and per-pig F1 boxplot.

Inputs:
    --nested  per_segment_predictions.csv from run_nested_cv.py
    --yin     yin_2021_probs.csv (AlexNet per-segment soft-max probabilities)
    --dorr    dorr_2026_probs.csv (BEATs per-segment soft-max probabilities)

Outputs (in --out-dir):
    summary_final.json             aggregate F1 + CIs + p-values
    per_pig_f1_boxplot.png         3-method boxplot of per-pig macro F1
    per_pig_table_final.csv        per-pig F1 of all three methods
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def cluster_bootstrap_ci(df, n_iter=10000, seed=42):
    """Resample 16 pigs with replacement; recompute aggregate Macro F1; return percentile CI."""
    rng = np.random.default_rng(seed)
    pigs = df['pig'].unique()
    pig_groups = {p: df[df['pig'] == p] for p in pigs}
    boots = np.empty(n_iter)
    for b in range(n_iter):
        sample = rng.choice(pigs, size=len(pigs), replace=True)
        cat = pd.concat([pig_groups[p] for p in sample], ignore_index=True)
        boots[b] = f1_score(cat['true'], cat['pred'], average='macro')
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def per_pig_f1(df):
    return {p: f1_score(g['true'], g['pred'], average='macro')
            for p, g in df.groupby('pig')}


def paired_permutation_test(diffs, n_iter=10000, seed=42):
    """One-sided H1 = mean(diff) > 0."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs)
    obs = d.mean()
    n = len(d)
    cnt = 0
    for _ in range(n_iter):
        if (rng.choice([-1, 1], size=n) * d).mean() >= obs:
            cnt += 1
    return float(obs), cnt / n_iter


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nested', required=True,
                    help='per_segment_predictions.csv (AST-XGB nested CV).')
    ap.add_argument('--yin', required=True,
                    help='yin_2021_probs.csv (AlexNet probabilities).')
    ap.add_argument('--dorr', required=True,
                    help='dorr_2026_probs.csv (BEATs probabilities).')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(">>> Loading AST-XGB nested predictions...")
    ast = pd.read_csv(args.nested)

    print(">>> Loading baseline probabilities (AlexNet, BEATs)...")
    yin  = pd.read_csv(args.yin)
    dorr = pd.read_csv(args.dorr)
    yin['pred']  = np.argmax(yin[['prob_0','prob_1','prob_2']].values,  axis=1)
    dorr['pred'] = np.argmax(dorr[['prob_0','prob_1','prob_2']].values, axis=1)

    ast_f1  = f1_score(ast['true'],  ast['pred'],  average='macro')
    yin_f1  = f1_score(yin['true'],  yin['pred'],  average='macro')
    dorr_f1 = f1_score(dorr['true'], dorr['pred'], average='macro')
    print(f"\nAggregate Macro F1:")
    print(f"  AST-XGB (nested CV) : {ast_f1:.4f}")
    print(f"  AlexNet (argmax)    : {yin_f1:.4f}")
    print(f"  BEATs   (argmax)    : {dorr_f1:.4f}")

    print(">>> Cluster-bootstrap 95% CI by pigs (10 000 iter)...")
    ast_lo, ast_hi   = cluster_bootstrap_ci(ast)
    yin_lo, yin_hi   = cluster_bootstrap_ci(yin)
    dorr_lo, dorr_hi = cluster_bootstrap_ci(dorr)
    print(f"  AST-XGB : [{ast_lo:.4f}, {ast_hi:.4f}]")
    print(f"  AlexNet : [{yin_lo:.4f}, {yin_hi:.4f}]")
    print(f"  BEATs   : [{dorr_lo:.4f}, {dorr_hi:.4f}]")

    print(">>> Paired-permutation test (10 000 iter)...")
    ast_pp, yin_pp, dorr_pp = per_pig_f1(ast), per_pig_f1(yin), per_pig_f1(dorr)
    common = sorted(set(ast_pp) & set(yin_pp) & set(dorr_pp))
    obs_y, p_y = paired_permutation_test([ast_pp[p] - yin_pp[p]  for p in common])
    obs_d, p_d = paired_permutation_test([ast_pp[p] - dorr_pp[p] for p in common])
    print(f"  AST-XGB vs AlexNet : diff = {obs_y:+.4f}, p = {p_y:.4f}")
    print(f"  AST-XGB vs BEATs   : diff = {obs_d:+.4f}, p = {p_d:.4f}")

    summary = {
        'protocol': 'AST-XGB: nested LOOCV; baselines at argmax',
        'ast_xgb':          {'f1': ast_f1,  'ci_lo': ast_lo,  'ci_hi': ast_hi},
        'yin_2021_alexnet': {'f1': yin_f1,  'ci_lo': yin_lo,  'ci_hi': yin_hi},
        'dorr_2026_beats':  {'f1': dorr_f1, 'ci_lo': dorr_lo, 'ci_hi': dorr_hi},
        'permutation_test': {
            'ast_vs_alexnet': {'mean_per_pig_diff': obs_y, 'p_value_one_sided': p_y},
            'ast_vs_beats':   {'mean_per_pig_diff': obs_d, 'p_value_one_sided': p_d},
        },
        'n_pigs': len(common),
        'n_bootstrap_iter': 10000,
        'n_permutation_iter': 10000,
    }
    with open(os.path.join(args.out_dir, 'summary_final.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {args.out_dir}/summary_final.json")

    # Per-pig table
    pp_df = pd.DataFrame([{'pig': p, 'ast_xgb': ast_pp[p],
                           'alexnet': yin_pp[p], 'beats': dorr_pp[p]}
                          for p in common])
    pp_df.to_csv(os.path.join(args.out_dir, 'per_pig_table_final.csv'), index=False)

    # Boxplot
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = [list(ast_pp.values()), list(yin_pp.values()), list(dorr_pp.values())]
    labels = ['AST-XGB\n(nested CV)', 'AlexNet\n(argmax)', 'BEATs\n(argmax)']
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True,
                    meanprops={'marker': 'D', 'markerfacecolor': 'white',
                               'markeredgecolor': 'black'})
    for patch, c in zip(bp['boxes'], ['#2c7fb8', '#7fcdbb', '#edf8b1']):
        patch.set_facecolor(c)
    ax.set_ylabel('Per-pig Macro F1 (16 LOOCV folds)')
    ax.set_title('Per-pig generalisation: AST-XGB vs argmax baselines')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'per_pig_f1_boxplot.png'), dpi=300)
    print(f"Saved {args.out_dir}/per_pig_f1_boxplot.png")


if __name__ == "__main__":
    main()
