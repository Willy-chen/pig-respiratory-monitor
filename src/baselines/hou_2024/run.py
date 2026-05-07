import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import get_loocv_df
from train_evaluate import run_loocv, save_report
from baselines.hou_2024.model import MultiFeatureMLP

def main():
    print("=== Hou et al. 2024 Baseline (Multi-feature BP Network) ===")
    xgb_df = get_loocv_df()
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    model_fn = lambda: MultiFeatureMLP(num_classes=3)
    report, macro_f1 = run_loocv(
        model_fn, xgb_df,
        epochs=10, lr=1e-3, batch_size=32,
        log_dir=log_dir
    )
    save_report(f"Macro F1: {macro_f1:.4f}\n\n{report}",
                os.path.join(os.path.dirname(__file__), 'results.txt'))

if __name__ == "__main__":
    main()
