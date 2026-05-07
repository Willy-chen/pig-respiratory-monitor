import os
import glob
import pandas as pd
import librosa
import soundfile as sf
import data_utils
from tqdm import tqdm

# Configuration
INPUT_CSV = "xgb_results/wrong_predictions.csv"
OUTPUT_DIR = "xgb_results/error_samples"
SR = 16000

def find_file_map():
    file_map = {}
    
    # 1. New Data Roots
    roots = data_utils.NEW_AUDIO_ROOTS
    for r in roots:
        if not os.path.exists(r):
            r = os.path.join(os.path.dirname(__file__), r)
        full_root = os.path.abspath(r)
        files = glob.glob(os.path.join(full_root, "*.wav"))
        for f in files:
            file_map[os.path.basename(f)] = f
            
    # 2. Old Data Roots
    old_root = data_utils.OLD_AUDIO_ROOT
    if not os.path.exists(old_root):
        old_root = os.path.join(os.path.dirname(__file__), old_root)
    old_full_root = os.path.abspath(old_root)
    all_old_files = glob.glob(os.path.join(old_full_root, "*", "*.wav"))
    for f in all_old_files:
        file_map[os.path.basename(f)] = f
        
    return file_map

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} error entries.")
    
    file_map = find_file_map()
    print(f"Found {len(file_map)} source audio files.")
    
    success_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['Filename']
        if not fname.endswith(".wav"):
            fname += ".wav"
            
        start, end = float(row['Start']), float(row['End'])
        true_lbl, pred_lbl = int(row['True_Label']), int(row['Pred_Label'])
        
        conf = row.get(f"Prob_{pred_lbl}", 0.0)
        
        if fname not in file_map:
            continue
            
        src_path = file_map[fname]
        try:
            y, sr = librosa.load(src_path, sr=SR, offset=start, duration=end-start)
            safe_fname = fname.replace(" ", "_").replace(".wav", "")
            out_name = f"True{true_lbl}_Pred{pred_lbl}_Conf{conf:.2f}_{safe_fname}_{int(start)}s.wav"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            sf.write(out_path, y, SR)
            success_count += 1
        except Exception as e:
            print(f"Error {fname}: {e}")
            
    print(f"\nCompleted! Extracted {success_count} segments to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
