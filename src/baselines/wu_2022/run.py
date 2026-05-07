import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import get_loocv_df
from train_evaluate import run_loocv, save_report
from baselines.wu_2022.model import MFCC_LSTM

def main():
    print("=== Wu et al. 2022 Baseline (MFCC + Bi-LSTM) ===")
    xgb_df = get_loocv_df()
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    model_fn = lambda: MFCC_LSTM(num_classes=3)
    report, macro_f1 = run_loocv(
        model_fn, xgb_df,
        epochs=15, lr=5e-4, batch_size=32,
        log_dir=log_dir
    )
    save_report(f"Macro F1: {macro_f1:.4f}\n\n{report}",
                os.path.join(os.path.dirname(__file__), 'results.txt'))

if __name__ == "__main__":
    main()
