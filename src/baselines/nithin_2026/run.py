import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import get_loocv_df
from train_evaluate import run_loocv, save_report
from baselines.nithin_2026.model import LSTMKAN

def main():
    print("=== Nithinkumar et al. 2026 Baseline (LSTM-KAN) ===")
    xgb_df = get_loocv_df()
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    model_fn = lambda: LSTMKAN(num_classes=3)
    report, macro_f1 = run_loocv(
        model_fn, xgb_df,
        epochs=15, lr=5e-4, batch_size=32,
        log_dir=log_dir
    )
    save_report(f"Macro F1: {macro_f1:.4f}\n\n{report}",
                os.path.join(os.path.dirname(__file__), 'results.txt'))

if __name__ == "__main__":
    main()
