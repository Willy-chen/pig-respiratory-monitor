"""
Re-run yin_2021 (AlexNet) and dorr_2026 (BEATs) baselines with per-segment
softmax probability logging, so we can apply the same nested threshold
optimization that AST-XGB receives. Baseline parity (top-2 only).

Requires raw audio + the BEATs Iter3 + AS2M checkpoint (not redistributed in
this repo; see docs/REPRODUCE.md Tier 2).

Override BASELINES_DIR or BASELINES_OUT_DIR via environment variables to use
a non-default location.
"""
import os, sys, time
import numpy as np, pandas as pd
import torch, torch.nn as nn

# Repo-relative defaults; override with env vars
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASEDIR = os.environ.get("BASELINES_DIR", os.path.join(_REPO_ROOT, "src", "baselines"))
OUT     = os.environ.get("BASELINES_OUT_DIR",
                          os.path.join(_REPO_ROOT, "results", "baseline_probs"))

sys.path.insert(0, BASEDIR)
from data_loader import get_loocv_df, PigSegmentDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

os.makedirs(OUT, exist_ok=True)

CLASS_NAMES = ['No-Breathing', 'Normal', 'Abnormal']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_loaders(train_ds, test_ds, batch_size, num_workers=4):
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True))

def loocv_with_probs(name, model_fn, xgb_df, epochs, lr, batch_size, transform_fn, out_csv):
    files = xgb_df['Filename'].unique()
    rows = []
    t0 = time.time()
    for fold, test_file in enumerate(files):
        train_df = xgb_df[xgb_df['Filename'] != test_file]
        test_df  = xgb_df[xgb_df['Filename'] == test_file]
        train_ds = PigSegmentDataset(train_df, transform_fn=transform_fn)
        test_ds  = PigSegmentDataset(test_df,  transform_fn=transform_fn)
        train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size)

        model = model_fn().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for ep in range(epochs):
            model.train()
            for X, y in train_loader:
                X, y = X.to(DEVICE, dtype=torch.float32), y.to(DEVICE, dtype=torch.long)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward(); optimizer.step()

        model.eval()
        seg_idx = test_df.index.tolist()
        with torch.no_grad():
            i = 0
            for X, y in test_loader:
                X = X.to(DEVICE, dtype=torch.float32)
                logits = model(X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                for j in range(probs.shape[0]):
                    rows.append({
                        'pig': test_file,
                        'segment_idx': int(seg_idx[i + j]),
                        'true': int(y[j].item()),
                        'prob_0': float(probs[j, 0]),
                        'prob_1': float(probs[j, 1]),
                        'prob_2': float(probs[j, 2]),
                    })
                i += probs.shape[0]
        print(f"  [{name}] fold {fold+1}/{len(files)} pig={test_file} elapsed={time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [{name}] saved {len(rows)} predictions to {out_csv}")
    print(f"  [{name}] total time: {time.time()-t0:.0f}s\n")

def main():
    xgb_df = get_loocv_df()
    print(f">>> XGB set: {len(xgb_df)} segments, {xgb_df['Filename'].nunique()} pigs")

    # -- yin_2021 (AlexNet + MelSpectrogram) ----------------------------
    print("\n=== yin_2021 (AlexNet) ===")
    sys.path.insert(0, os.path.join(BASEDIR, 'baselines/yin_2021'))
    from baselines.yin_2021.model import SpectrogramAlexNet
    loocv_with_probs(
        'yin_2021',
        lambda: SpectrogramAlexNet(num_classes=3, pretrained=True),
        xgb_df, epochs=10, lr=1e-4, batch_size=32, transform_fn=None,
        out_csv=f"{OUT}/yin_2021_probs.csv"
    )

    # -- dorr_2026 (BEATs Iter3) ----------------------------------------
    print("\n=== dorr_2026 (BEATs Iter3) ===")
    from baselines.dorr_2026.model import DorrBEATsOfficial
    ckpt = os.path.abspath(os.path.join(BASEDIR, 'pretrained_models/BEATs_iter3_plus_AS2M.pt'))
    loocv_with_probs(
        'dorr_2026',
        lambda: DorrBEATsOfficial(ckpt_path=ckpt, num_classes=3),
        xgb_df, epochs=10, lr=1e-3, batch_size=16, transform_fn=None,
        out_csv=f"{OUT}/dorr_2026_probs.csv"
    )

if __name__ == "__main__":
    main()
