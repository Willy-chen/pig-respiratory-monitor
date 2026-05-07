import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

# Configuration
# These paths point to the private raw-audio dataset, which is NOT redistributed
# with this repository. Override via environment variables, e.g.:
#   PIG_AUDIO_ROOT=/data/pig_audio python run_ultimate_search.py
# See docs/REPRODUCE.md (Tier 3) for how to obtain the audio under a DUA.
_PIG_AUDIO_ROOT = os.environ.get("PIG_AUDIO_ROOT", os.path.expanduser("~/pig/audio"))

NEW_AUDIO_ROOTS = [
    os.path.join(_PIG_AUDIO_ROOT, "diagnostic_labelled", "normal"),
    # os.path.join(_PIG_AUDIO_ROOT, "diagnostic_labelled", "abnormal"),
]
OLD_AUDIO_ROOT = _PIG_AUDIO_ROOT
LABEL_DIR = os.environ.get("PIG_LABEL_DIR", os.path.expanduser("~/pig/strong_labels"))
SEGMENT_DURATION = 10.0
SR = 16000
THRESHOLD_DB = -40.0
SEED = 42

def compute_energy_db(y):
    rms = np.sqrt(np.mean(y**2))
    return 20 * np.log10(rms + 1e-9)

def load_new_data():
    print(">>> Loading New Data (Breathing: 5s hop, No-Breathing: 30s hop)...")
    data = []
    abs_roots = [os.path.abspath(os.path.expanduser(p)) for p in NEW_AUDIO_ROOTS]
    
    BREATHING_HOP = 5.0
    NO_BREATHING_HOP = 30.0
    
    for root in abs_roots:
        if not os.path.exists(root): continue
        files = glob.glob(os.path.join(root, "*.wav"))
        folder_name = os.path.basename(root)
        label = 1 if "正常" in folder_name else 2
        
        for fpath in tqdm(files, desc=f"Scanning {folder_name}", leave=False):
            try:
                y, sr = librosa.load(fpath, sr=SR)
                duration = len(y) / sr
                filename = os.path.basename(fpath)
                
                # Pass 1: Breathing (High Energy) - Hop 5s
                curr = 0
                while curr + SEGMENT_DURATION <= duration:
                    start_sample = int(curr * sr)
                    end_sample = int((curr + SEGMENT_DURATION) * sr)
                    segment = y[start_sample:end_sample]
                    energy = compute_energy_db(segment)
                    
                    if energy >= THRESHOLD_DB:
                        data.append({
                            'Filename': filename,
                            'Start': curr,
                            'End': curr + SEGMENT_DURATION,
                            'Target': label,
                            'Energy': energy,
                            'Source': 'New',
                            'Audio_Path': fpath
                        })
                    curr += BREATHING_HOP
                    
                # Pass 2: No-Breathing (Low Energy) - Hop 30s
                # Independent scan to ensure undersampling rate
                curr = 0
                while curr + SEGMENT_DURATION <= duration:
                    start_sample = int(curr * sr)
                    end_sample = int((curr + SEGMENT_DURATION) * sr)
                    segment = y[start_sample:end_sample]
                    energy = compute_energy_db(segment)
                    
                    if energy < THRESHOLD_DB:
                         data.append({
                            'Filename': filename,
                            'Start': curr,
                            'End': curr + SEGMENT_DURATION,
                            'Target': 0, # Class 0
                            'Energy': energy,
                            'Source': 'New',
                            'Audio_Path': fpath
                        })
                    curr += NO_BREATHING_HOP
                    
            except Exception as e:
                print(f"Error {fpath}: {e}")
    return pd.DataFrame(data)

def load_old_data():
    print(">>> Loading Original Data (Breathing: Label-guided 5s hop, No-Breathing: Gaps 30s hop)...")
    data = []
    label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    
    BREATHING_HOP = 5.0
    NO_BREATHING_HOP = 30.0
    
    for l_file in tqdm(label_files, desc="Scanning Labels", leave=False):
        filename = os.path.basename(l_file).replace('.txt', '')
        norm_path = os.path.join(OLD_AUDIO_ROOT, "Normal", f"{filename}.wav")
        abn_path = os.path.join(OLD_AUDIO_ROOT, "Abnormal", f"{filename}.wav")
        
        if os.path.exists(norm_path):
            audio_path = norm_path
        elif os.path.exists(abn_path):
            audio_path = abn_path
        else:
            continue
            
        try:
            y, sr = librosa.load(audio_path, sr=SR)
            duration = len(y) / sr
            labels = pd.read_csv(l_file, sep='\t', names=['Start', 'End', 'Label'])
            labels = labels.dropna()
            
            target = 2 if "Abnormal" in audio_path else 1
            
            # 1. Collect Breathing Intervals
            breathing_intervals = []
            for _, row in labels.iterrows():
                start, end = row['Start'], row['End']
                breathing_intervals.append((start, end))
                
                # Extract Breathing Samples
                # Logic: If event >= 10s, slide 5s. Else center crop.
                if (end - start) >= SEGMENT_DURATION:
                    curr = start
                    while curr + SEGMENT_DURATION <= end:
                        data.append({
                            'Filename': filename,
                            'Start': curr,
                            'End': curr + SEGMENT_DURATION,
                            'Target': target,
                            'Energy': 0, # Placeholder, not checking energy for labeled
                            'Source': 'Old',
                            'Audio_Path': audio_path
                        })
                        curr += BREATHING_HOP
                else:
                     center = (start + end) / 2
                     new_start = max(0, center - SEGMENT_DURATION / 2)
                     new_end = min(duration, new_start + SEGMENT_DURATION)
                     if new_end - new_start < SEGMENT_DURATION: new_start = max(0, new_end - SEGMENT_DURATION)
                     
                     data.append({
                        'Filename': filename,
                        'Start': new_start,
                        'End': new_end,
                        'Target': target,
                        'Energy': 0,
                        'Source': 'Old',
                        'Audio_Path': audio_path
                    })

            # 2. Extract No-Breathing from Gaps
            gaps = []
            curr_pos = 0
            sorted_intervals = sorted(breathing_intervals)
            for b_start, b_end in sorted_intervals:
                if b_start - curr_pos >= SEGMENT_DURATION:
                    gaps.append((curr_pos, b_start))
                curr_pos = max(curr_pos, b_end)
            if duration - curr_pos >= SEGMENT_DURATION:
                gaps.append((curr_pos, duration))
            
            for g_start, g_end in gaps:
                curr = g_start
                while curr + SEGMENT_DURATION <= g_end:
                    # Optional: Double check energy? User said "segments that is not in the strong label... as noise".
                    # Implicitly Class 0.
                    data.append({
                        'Filename': filename,
                        'Start': curr,
                        'End': curr + SEGMENT_DURATION,
                        'Target': 0,
                        'Energy': 0,
                        'Source': 'Old',
                        'Audio_Path': audio_path
                    })
                    curr += NO_BREATHING_HOP

        except Exception as e:
            print(f"Error {filename}: {e}")
            
    return pd.DataFrame(data)

def get_full_dataset():
    df1 = load_new_data()
    df2 = load_old_data()
    return pd.concat([df1, df2], ignore_index=True)

def create_study_split(df):
    """
    Splits files into TWO disjoint sets:
    1. AST_SET: Used for fine-tuning AST. (e.g. 50% of files)
    2. XGB_SET: Used for LOOCV XGBoost. (e.g. 50% of files)
    """
    # 1. Get One Label Per File for Stratification
    # Logic: Files have segments of 0 (Noise) and 1 or 2 (Signal).
    # taking max() implies:
    #   Normal File (0, 1) -> 1
    #   Abnormal File (0, 2) -> 2
    file_labels = df.groupby('Filename')['Target'].max()
    unique_filenames = file_labels.index.values
    unique_targets = file_labels.values
    
    # 2. Stratified Split of UNIQUE FILES
    from sklearn.model_selection import train_test_split
    
    train_files, test_files = train_test_split(
        unique_filenames,
        test_size=0.5, 
        stratify=unique_targets,
        random_state=SEED
    )
    
    # 3. Create DataFrames
    ast_df = df[df['Filename'].isin(train_files)].reset_index(drop=True)
    xgb_df = df[df['Filename'].isin(test_files)].reset_index(drop=True)
    
    print(f"Split Layout:")
    print(f"  AST Set: {len(ast_df)} segments, {len(train_files)} files")
    print(f"  XGB Set: {len(xgb_df)} segments, {len(test_files)} files")
    
    # Verification
    inter = set(train_files).intersection(set(test_files))
    if inter:
        raise ValueError(f"CRITICAL: Data Leakage Detected! {len(inter)} files in both sets.")
    
    return ast_df, xgb_df
