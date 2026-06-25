"""
data_loader.py — Unified Data Loader for all 20260322 Baselines
================================================================
CRITICAL DESIGN NOTES:
  - Uses the EXACT SAME dataset as 20260302_ultimate (same files, same segments, same splits).
  - `create_study_split` is called from the original data_utils.py. 
  - The baselines use the XGB_SET (the same holdout set the ultimate method's XGBoost
    classifier was evaluated on). This ensures fair apples-to-apples comparison.
  - Evaluation protocol: file-level Leave-One-Out Cross Validation (LOOCV), 
    matching the ultimate method's protocol exactly.
  - All audio is loaded at 16kHz, 10-second segments.
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader

# Point to the original data_utils from 20260302_ultimate
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20260302_ultimate')))
from data_utils import get_full_dataset, create_study_split

SR = 16000
SEGMENT_DURATION = 10.0
TARGET_LEN = SR * int(SEGMENT_DURATION)

CLASS_NAMES = ['No-Breathing', 'Normal', 'Abnormal']

_WAVE_CACHE = {}
def load_segment_cached(audio_path, start):
    key = (audio_path, round(float(start), 4))
    y = _WAVE_CACHE.get(key)
    if y is None:
        try:
            y, _ = librosa.load(audio_path, sr=SR, offset=float(start), duration=SEGMENT_DURATION)
        except Exception:
            y = np.zeros(TARGET_LEN, dtype=np.float32)
        y = np.pad(y, (0, TARGET_LEN - len(y))) if len(y) < TARGET_LEN else y[:TARGET_LEN]
        y = y.astype(np.float32)
        _WAVE_CACHE[key] = y
    return y

_MANIFEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "segments_xgb_1553_manifest.csv"))

def get_loocv_df():
    """XGB_DF (1,553 segs / 16 pigs) from the reconstructed headline manifest."""
    m = pd.read_csv(_MANIFEST_PATH)
    df = pd.DataFrame({
        "Filename": m["pig_id"].astype(str),
        "Audio_Path": m["Audio_Path"].astype(str),
        "Start": m["Start"].astype(float),
        "Target": m["label"].astype(int),
    })
    df.index = m["segment_idx"].astype(int)
    return df


class PigSegmentDataset(Dataset):
    """
    Loads pig audio segments from a given DataFrame.
    Returns (processed_features, label) where:
      - processed_features: either raw waveform float32 tensor (160000,)
        or the output of transform_fn.
      - label: int in {0, 1, 2}
    """
    def __init__(self, df, transform_fn=None):
        self.df = df.reset_index(drop=True)
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = load_segment_cached(row['Audio_Path'], row['Start'])

        if self.transform_fn:
            return self.transform_fn(y, SR), int(row['Target'])
        
        return torch.tensor(y, dtype=torch.float32), int(row['Target'])


import pandas as pd

def get_loocv_folds(xgb_df, transform_fn=None):
    """
    Returns a generator of (train_dataset, test_dataset, test_filename) tuples.
    Implements file-level Leave-One-Out Cross Validation. Re-merges the AST_SET
    into the training data to ensure fair representation compared to the ultimate method.
    """
    unique_files = xgb_df['Filename'].unique()
    for test_file in unique_files:
        train_df = xgb_df[xgb_df['Filename'] != test_file]
        test_df  = xgb_df[xgb_df['Filename'] == test_file]
        yield (
            PigSegmentDataset(train_df, transform_fn=transform_fn),
            PigSegmentDataset(test_df, transform_fn=transform_fn),
            test_file
        )


def make_loaders(train_ds, test_ds, batch_size=32):
    # Set drop_last=True for training to avoid BatchNorm errors with batch size 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader
