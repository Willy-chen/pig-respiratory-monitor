"""Regenerate the XGBoost weight/threshold ablation surfaces (Table 9, Figs 4-5).

Deterministic: fixed seed, single-threaded hist. Run from repo root with the ml env.
Outputs to results/ablation/: threshold_surface.csv, weight_surface.csv,
Figure4.{pdf,png,tiff}, Figure5.{pdf,png,tiff}.
"""
import os, pickle, numpy as np, xgboost as xgb
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs('results/ablation', exist_ok=True)
SEED = 42
X, y, g = pickle.load(open('results/features_3layer_mean.pkl', 'rb'))
X = np.asarray(X); y = np.asarray(y); g = np.asarray(g)
params = dict(objective='multi:softprob', num_class=3, max_depth=4, eta=0.1,
              verbosity=0, tree_method='hist', seed=SEED, nthread=1)
pigs = np.unique(g)

def loocv(aN, aA):
    R = np.zeros((len(y), 3))
    for tp in pigs:
        te = g == tp; tr = ~te; yt = y[tr]; nt = len(yt); wm = {}
        for c in np.unique(yt):
            nc = int(np.sum(yt == c)); w = nt / (3 * nc)
            w *= aN if c == 1 else (aA if c == 2 else 1.0); wm[c] = w
        wv = np.array([wm[l] for l in yt])
        b = xgb.train(params, xgb.DMatrix(X[tr], label=yt, weight=wv), num_boost_round=100)
        R[te] = b.predict(xgb.DMatrix(X[te]))
    return R

def ap(R, tn, ta):
    return np.array([2 if p[2] >= ta else (1 if p[1] >= tn else 0) for p in R])

TN = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TA = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
R35 = loocv(3, 5)
thr = np.array([[f1_score(y, ap(R35, tn, ta), average='macro') for ta in TA] for tn in TN])
pd.DataFrame(thr, index=TN, columns=TA).round(4).to_csv('results/ablation/threshold_surface.csv')

AN = [1, 2, 3, 4]; AA = [1, 3, 5, 7]
wts = np.zeros((len(AN), len(AA)))
for i, an in enumerate(AN):
    for j, aa in enumerate(AA):
        wts[i, j] = f1_score(y, ap(loocv(an, aa), 0.70, 0.25), average='macro')
pd.DataFrame(wts, index=AN, columns=AA).round(4).to_csv('results/ablation/weight_surface.csv')

print('PEAK_THR', round(thr[TN.index(0.70)][TA.index(0.25)], 4))
print('PEAK_WT', round(wts[AN.index(3)][AA.index(5)], 4))
print('WEIGHT_CELLS @(0.70,0.25)')
for an, aa in [(1, 1), (1, 7), (2, 5), (3, 5), (4, 5)]:
    print('  (%d,%d) %.4f' % (an, aa, wts[AN.index(an)][AA.index(aa)]))
print('THR_CELLS @w(3,5)')
for tn, ta in [(0.50, 0.50), (0.60, 0.40), (0.70, 0.30), (0.70, 0.25), (0.75, 0.20)]:
    print('  (%.2f,%.2f) %.4f' % (tn, ta, thr[TN.index(tn)][TA.index(ta)]))

# Figure 4 - threshold landscape
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(thr, aspect='auto', cmap='viridis', origin='lower')
ax.set_xticks(range(len(TA))); ax.set_xticklabels(['%.2f' % t for t in TA])
ax.set_yticks(range(len(TN))); ax.set_yticklabels(['%.2f' % t for t in TN])
ax.set_xlabel('Abnormal threshold $t_A$'); ax.set_ylabel('Normal threshold $t_N$')
for i in range(len(TN)):
    for j in range(len(TA)):
        ax.text(j, i, '%.3f' % thr[i, j], ha='center', va='center', color='w', fontsize=6.5)
fig.colorbar(im, label='Macro F1'); fig.tight_layout()
for ext in ['pdf', 'png', 'tiff']:
    fig.savefig('results/ablation/Figure4.%s' % ext, dpi=300)
plt.close(fig)

# Figure 5 - weight heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(wts, aspect='auto', cmap='viridis', origin='lower')
ax.set_xticks(range(len(AA))); ax.set_xticklabels(AA)
ax.set_yticks(range(len(AN))); ax.set_yticklabels(AN)
ax.set_xlabel(r'Abnormal multiplier $\alpha_A$'); ax.set_ylabel(r'Normal multiplier $\alpha_N$')
for i in range(len(AN)):
    for j in range(len(AA)):
        ax.text(j, i, '%.3f' % wts[i, j], ha='center', va='center', color='w', fontsize=8)
fig.colorbar(im, label='Macro F1'); fig.tight_layout()
for ext in ['pdf', 'png', 'tiff']:
    fig.savefig('results/ablation/Figure5.%s' % ext, dpi=300)
plt.close(fig)
print('DONE')
