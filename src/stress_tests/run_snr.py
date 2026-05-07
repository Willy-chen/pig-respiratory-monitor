"""
run_snr.py — SNR Robustness Stress Test
========================================
Trains each model on clean data (16-fold LOOCV), then evaluates each fold's
test set with audio corrupted at SNR levels: [Clean, +10, 0, -5, -10] dB.

Noise sources (Three-Tier):
  1. Farm (Internal) : Class-0 (No-Breathing) segments from the training fold
  2. White           : Gaussian white noise generated on-the-fly
  3. Pink            : 1/f pink noise generated on-the-fly

Usage:
  python run_snr.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
import pickle
import xgboost as xgb
from sklearn.metrics import f1_score
from importlib import import_module

SR = 16000
SEG_LEN = SR * 10

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../20260322')))
from data_loader import get_loocv_df, make_loaders, PigSegmentDataset
from train_evaluate import train_one_epoch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SNR_LEVELS = [None, 10, 0, -5, -10]   # None = Clean
NOISE_TYPES = ['Farm', 'White', 'Pink']

# ── Baseline registry (model_name -> factory info) ──────────────────────────
CKPT_BEATS = os.path.abspath('../20260322/pretrained_models/BEATs_iter3_plus_AS2M.pt')

BASELINE_REGISTRY = {
    'ultimate_ast_xgb': None,  # handled separately
    'yin_2021':   {'module': 'baselines.yin_2021.model',   'cls': 'SpectrogramAlexNet', 'args': {'num_classes': 3, 'pretrained': True},  'epochs': 10, 'lr': 1e-4, 'bs': 32},
    'dorr_2026':  {'module': 'baselines.dorr_2026.model',  'cls': 'DorrBEATsOfficial', 'args': {'ckpt_path': CKPT_BEATS, 'num_classes': 3}, 'epochs': 5,  'lr': 1e-3, 'bs': 16},
    'sheikh_2024':{'module': 'baselines.sheikh_2024.model','cls': 'WhisperClassifier',  'args': {'num_classes': 3},                        'epochs': 5,  'lr': 1e-4, 'bs': 16},
    'wu_2022':    {'module': 'baselines.wu_2022.model',    'cls': 'MFCC_LSTM',          'args': {'num_classes': 3},                        'epochs': 10, 'lr': 1e-3, 'bs': 32},
    'shen_2022':  {'module': 'baselines.shen_2022.model',  'cls': 'LeNet5Fusion',       'args': {'num_classes': 3},                        'epochs': 10, 'lr': 1e-3, 'bs': 32},
    'hou_2024':   {'module': 'baselines.hou_2024.model',   'cls': 'MultiFeatureMLP',    'args': {'num_classes': 3},                        'epochs': 10, 'lr': 1e-3, 'bs': 32},
    'wang_2026':  {'module': 'baselines.wang_2026.model',  'cls': 'CoughRNet',          'args': {'num_classes': 3},                        'epochs': 10, 'lr': 1e-3, 'bs': 32},
    'mdpi_2026':  {'module': 'baselines.mdpi_2026.model',  'cls': 'PigCoughCNN',        'args': {'num_classes': 3},                        'epochs': 10, 'lr': 1e-3, 'bs': 32},
    'nithin_2026':{'module': 'baselines.nithin_2026.model','cls': 'LSTMKAN',            'args': {'num_classes': 3},                        'epochs': 10, 'lr': 1e-3, 'bs': 32},
}

# ── Noise generators ─────────────────────────────────────────────────────────

def white_noise(length):
    return np.random.randn(length).astype(np.float32)

def pink_noise(length):
    wn = np.random.randn(length)
    X = np.fft.rfft(wn)
    f = np.fft.rfftfreq(length)
    f[0] = 1.0
    X_pink = X / np.sqrt(f)
    pn = np.fft.irfft(X_pink, n=length)
    return pn.astype(np.float32)

def load_farm_noise_pool(noise_df):
    """Load all Class-0 segments from a dataframe into a list of arrays."""
    pool = []
    for _, row in noise_df.iterrows():
        try:
            y, _ = librosa.load(row['Audio_Path'], sr=SR, offset=float(row['Start']), duration=10.0)
            if len(y) < SEG_LEN:
                y = np.pad(y, (0, SEG_LEN - len(y)))
            pool.append(y[:SEG_LEN].astype(np.float32))
        except:
            pass
    return pool

def load_esc50_noise_pool():
    """Load ESC-50 'Rain' and 'Wind' samples as environmental noise pool."""
    meta_path = "ESC-50/meta/esc50.csv"
    audio_dir = "ESC-50/audio"
    if not os.path.exists(meta_path): return []
    
    df = pd.read_csv(meta_path)
    # Categories: rain (10), wind (16). These often overlap with open barn acoustics.
    target_cats = ['rain', 'wind']
    subset = df[df['category'].isin(target_cats)]
    
    pool = []
    print(f">>> Loading {len(subset)} samples from ESC-50 (category: {target_cats})...")
    for _, row in subset.iterrows():
        try:
            path = os.path.join(audio_dir, row['filename'])
            y, _ = librosa.load(path, sr=SR)
            # ESC-50 samples are 5sec. Tile to 10sec.
            y = np.tile(y, 2)
            pool.append(y[:SEG_LEN].astype(np.float32))
        except:
            pass
    return pool

def add_noise_at_snr(signal, noise, snr_db):
    """Scale noise to achieve target SNR (dB), then mix."""
    s_rms = np.sqrt(np.mean(signal ** 2) + 1e-9)
    n_rms = np.sqrt(np.mean(noise ** 2) + 1e-9)
    target_n_rms = s_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_n_rms / (n_rms + 1e-9))
    mixed = np.clip(signal + noise_scaled, -1.0, 1.0)
    return mixed

def get_noise_sample(noise_type, farm_pool, esc50_pool, length):
    if noise_type == 'White':
        return white_noise(length)
    elif noise_type == 'Pink':
        return pink_noise(length)
    elif noise_type == 'Farm':
        pool = farm_pool if farm_pool else []
        if not pool: return white_noise(length)
        n = pool[np.random.randint(len(pool))]
    elif noise_type == 'ESC-50':
        pool = esc50_pool if esc50_pool else []
        if not pool: return white_noise(length)
        n = pool[np.random.randint(len(pool))]
    else:
        return white_noise(length)
        
    if len(n) < length:
        n = np.tile(n, int(np.ceil(length / len(n))))
    return n[:length].astype(np.float32)

# ── AST feature extraction for noisy audio ───────────────────────────────────

def get_ast_model_and_processor():
    from transformers import ASTModel, ASTFeatureExtractor
    AST_MODEL_PATH = "../20260209_n/best_ast_model"
    m = ASTModel.from_pretrained(AST_MODEL_PATH, output_hidden_states=True).to(DEVICE).eval()
    p = ASTFeatureExtractor.from_pretrained(AST_MODEL_PATH)
    return m, p

def encode_with_ast(audio_arr, ast_model, processor):
    with torch.no_grad():
        inputs = processor(audio_arr, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
        out = ast_model(inputs)
        hs = torch.stack(out.hidden_states[-3:])
        avg = torch.mean(hs, dim=0)
        gp = torch.mean(avg, dim=1).cpu().numpy().squeeze()
    return gp

# ── Evaluation helpers ────────────────────────────────────────────────────────

def load_audio_arr(row):
    try:
        y, _ = librosa.load(row['Audio_Path'], sr=SR, offset=float(row['Start']), duration=10.0)
        if len(y) < SEG_LEN: y = np.pad(y, (0, SEG_LEN - len(y)))
        return y[:SEG_LEN].astype(np.float32)
    except:
        return np.zeros(SEG_LEN, dtype=np.float32)

def eval_ultimate_noisy(bst, test_df, farm_pool, esc50_pool, noise_type, snr_db, ast_model, processor):
    """Evaluate xgboost on noisy test segments by re-extracting AST features."""
    preds, targets = [], []
    for _, row in test_df.iterrows():
        signal = load_audio_arr(row)
        if snr_db is not None:
            noise = get_noise_sample(noise_type, farm_pool, esc50_pool, len(signal))
            signal = add_noise_at_snr(signal, noise, snr_db)
        feat = encode_with_ast(signal, ast_model, processor).reshape(1, -1)
        probs = bst.predict(xgb.DMatrix(feat))[0]
        # Apply optimal thresholds
        if probs[2] >= 0.25:
            pred = 2
        elif probs[1] >= 0.70:
            pred = 1
        else:
            pred = 0
        preds.append(pred)
        targets.append(int(row['Target']))
    return np.array(preds), np.array(targets)

class NoisyAudioDataset(torch.utils.data.Dataset):
    """Wraps a dataframe and injects noise on-the-fly."""
    def __init__(self, df, farm_pool, esc50_pool, noise_type, snr_db, transform_fn=None):
        self.df = df.reset_index(drop=True)
        self.farm_pool = farm_pool
        self.esc50_pool = esc50_pool
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = load_audio_arr(row)
        if self.snr_db is not None:
            noise = get_noise_sample(self.noise_type, self.farm_pool, self.esc50_pool, len(signal))
            signal = add_noise_at_snr(signal, noise, self.snr_db)
        if self.transform_fn:
            # Note: transform_fn usually handles waveform to spectrogram
            return self.transform_fn(signal, SR), int(row['Target'])
        return torch.tensor(signal, dtype=torch.float32), int(row['Target'])

def eval_dl_noisy(model, test_df, farm_pool, esc50_pool, noise_type, snr_db, transform_fn=None):
    ds = NoisyAudioDataset(test_df, farm_pool, esc50_pool, noise_type, snr_db, transform_fn=transform_fn)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE, dtype=torch.float32)
            out = model(X).argmax(dim=1).cpu().numpy()
            preds.extend(out)
            targets.extend(y.numpy())
    return np.array(preds), np.array(targets)

# ── Whisper Feature Extractor (shared) ───────────────────────────────────────
from transformers import WhisperFeatureExtractor
_W_EXTRACTOR = None

def get_whisper_transform():
    global _W_EXTRACTOR
    if _W_EXTRACTOR is None:
        print(">>> Initializing Whisper Feature Extractor...")
        _W_EXTRACTOR = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    def transform(y, sr):
        # Whisper expects 80x3000 input. Extractor handles padding.
        # squeeze(0) to get (80, 3000)
        t = _W_EXTRACTOR(y, sampling_rate=sr, return_tensors="pt").input_features.squeeze(0)
        return t
    return transform

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    xgb_df = get_loocv_df()
    unique_files = xgb_df['Filename'].unique()

    # Load cached AST features for Ultimate (clean training)
    cache_path = "../20260302_ultimate/results/features_3layer_mean.pkl"
    with open(cache_path, 'rb') as f:
        X_ast, y_ast, groups_ast = pickle.load(f)

    # 1. Load Environmental Noise (ESC-50)
    esc50_pool = load_esc50_noise_pool()

    # 2. Get Specialized Transforms
    whisper_tf = get_whisper_transform()

    # 3. Load AST model for live re-encoding noisy test data
    print(">>> Loading AST model for live encoding of noisy test segments...")
    ast_model, ast_processor = get_ast_model_and_processor()

    # 3. Instantiate DL baselines
    dl_models = {}
    for name, cfg in BASELINE_REGISTRY.items():
        if cfg is None: continue
        try:
            mod = import_module(cfg['module'])
            Cls = getattr(mod, cfg['cls'])
            dl_models[name] = cfg
            print(f"  Loaded {name}")
        except Exception as e:
            print(f"  SKIP {name}: {e}")

    all_results = []
    
    # 4. Expansion of Noise Types
    ACTUAL_NOISE_TYPES = ['Farm', 'White', 'Pink', 'ESC-50']
    csv_file = "snr_results.csv"

    for fold_idx, test_file in enumerate(unique_files):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{len(unique_files)}: {test_file}")
        print(f"{'='*60}")

        train_df = xgb_df[xgb_df['Filename'] != test_file].reset_index(drop=True)
        test_df  = xgb_df[xgb_df['Filename'] == test_file].reset_index(drop=True)

        # Build farm noise pool from training fold Class-0 segments
        farm_pool = load_farm_noise_pool(train_df[train_df['Target'] == 0])

        # Check what's already done for this fold
        existing_models_for_fold = []
        if os.path.exists(csv_file):
            edf = pd.read_csv(csv_file)
            existing_models_for_fold = edf[edf['Fold'] == fold_idx]['Model'].unique().tolist()

        # ── Train/Eval Ultimate (clean training, noisy eval) ────────────────
        if 'ultimate_ast_xgb' not in existing_models_for_fold:
            train_mask = (groups_ast != test_file)
            X_tr, y_tr = X_ast[train_mask], y_ast[train_mask]
            n_total = len(y_tr)
            wm = {}
            for cls in np.unique(y_tr):
                n_c = np.sum(y_tr == cls)
                w = n_total / (3 * n_c)
                if cls == 1: w *= 3.0
                if cls == 2: w *= 5.0
                wm[cls] = w
            sample_w = np.array([wm.get(l, 1.0) for l in y_tr])
            bst = xgb.train(
                {'objective': 'multi:softprob', 'num_class': 3, 'max_depth': 4, 'eta': 0.1, 'verbosity': 0},
                xgb.DMatrix(X_tr, label=y_tr, weight=sample_w),
                num_boost_round=100
            )
            print("  Ultimate XGBoost trained.")

            conditions = [('Clean', None, None)]
            for nt in ACTUAL_NOISE_TYPES:
                for snr in [10, 0, -5, -10]:
                    conditions.append((nt, nt, snr))

            for label, noise_type, snr_db in conditions:
                snr_str = 'Clean' if snr_db is None else f"{snr_db:+d}dB"
                up, ut = eval_ultimate_noisy(bst, test_df, farm_pool, esc50_pool, noise_type, snr_db, ast_model, ast_processor)
                res = {
                    'Model': 'ultimate_ast_xgb', 'Fold': fold_idx,
                    'Noise': noise_type or 'Clean', 'SNR_dB': snr_db,
                    'F1': f1_score(ut, up, average='macro', zero_division=0)
                }
                pd.DataFrame([res]).to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
            print("  Ultimate evaluated.")
        else:
            print("  Ultimate already done for this fold.")

        # ── Train/Eval DL baselines ────────────────────────────────────────
        for name, cfg in dl_models.items():
            if name in existing_models_for_fold:
                print(f"  {name} already done for this fold.")
                continue

            # Determine transform function
            t_fn = None
            if name == 'sheikh_2024':
                t_fn = whisper_tf

            train_ds = PigSegmentDataset(train_df, transform_fn=t_fn)
            mod = import_module(cfg['module'])
            Cls = getattr(mod, cfg['cls'])
            model = Cls(**cfg['args']).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
            tl = torch.utils.data.DataLoader(train_ds, batch_size=cfg['bs'], shuffle=True,
                                              num_workers=0, drop_last=True)
            for ep in range(cfg['epochs']):
                train_one_epoch(model, tl, criterion, optimizer, DEVICE)
            print(f"  {name} trained.")

            conditions = [('Clean', None, None)]
            for nt in ACTUAL_NOISE_TYPES:
                for snr in [10, 0, -5, -10]:
                    conditions.append((nt, nt, snr))

            for label, noise_type, snr_db in conditions:
                dp, dt = eval_dl_noisy(model, test_df, farm_pool, esc50_pool, noise_type, snr_db, transform_fn=t_fn)
                res = {
                    'Model': name, 'Fold': fold_idx,
                    'Noise': noise_type or 'Clean', 'SNR_dB': snr_db,
                    'F1': f1_score(dt, dp, average='macro', zero_division=0)
                }
                pd.DataFrame([res]).to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
            print(f"  {name} evaluated.")

    # Final aggregation
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # Ensure SNR_dB is numeric (Clean=None becomes NaN in CSV)
        agg = df.groupby(['Model', 'Noise', 'SNR_dB'], dropna=False)['F1'].mean().reset_index()
        agg.to_csv("snr_results_aggregated.csv", index=False)
        print("\n>>> Done! Aggregated results saved to snr_results_aggregated.csv")

if __name__ == "__main__":
    main()
