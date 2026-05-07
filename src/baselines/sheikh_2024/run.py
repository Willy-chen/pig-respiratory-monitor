import os, sys
import torch
from transformers import WhisperFeatureExtractor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import get_loocv_df
from train_evaluate import run_loocv, save_report
from baselines.sheikh_2024.model import WhisperClassifier

# Setup feature extractor
# NOTE: Using a global extractor to avoid re-loading in every dataset access
extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

def whisper_transform(y, sr):
    # Whisper expects 80x3000 input for 30s, but we have 10s.
    # Feature extractor handles padding/truncation to the model's expected shape.
    inputs = extractor(y, sampling_rate=sr, return_tensors="pt")
    return inputs.input_features.squeeze(0) # (80, 3000)

def main():
    print("=== Sheikh et al. 2024 Baseline (Whisper + CNN/FCN) ===")
    xgb_df = get_loocv_df()
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    # Lambda ensures a the model is initialized within the fold as before
    model_fn = lambda: WhisperClassifier(num_classes=3)
    
    report, macro_f1 = run_loocv(
        model_fn, xgb_df,
        epochs=8, lr=1e-3, batch_size=16,
        log_dir=log_dir,
        transform_fn=whisper_transform
    )
    save_report(f"Macro F1: {macro_f1:.4f}\n\n{report}",
                os.path.join(os.path.dirname(__file__), 'results.txt'))

if __name__ == "__main__":
    main()
