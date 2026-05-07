import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import get_loocv_df
from train_evaluate import run_loocv, save_report
from baselines.yin_2021.model import SpectrogramAlexNet

def main():
    print("=== Yin et al. 2021 Baseline (AlexNet + MelSpectrogram) ===")
    xgb_df = get_loocv_df()
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    model_fn = lambda: SpectrogramAlexNet(num_classes=3, pretrained=True)
    report, macro_f1 = run_loocv(
        model_fn, xgb_df,
        epochs=10, lr=1e-4, batch_size=32,
        log_dir=log_dir
    )
    save_report(f"Macro F1: {macro_f1:.4f}\n\n{report}",
                os.path.join(os.path.dirname(__file__), 'results.txt'))

if __name__ == "__main__":
    main()
