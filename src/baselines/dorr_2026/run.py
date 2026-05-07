import os, sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import get_loocv_df
from train_evaluate import run_loocv, save_report
from baselines.dorr_2026.model import DorrBEATsOfficial

def main():
    print("=== Dörr et al. 2026 Baseline (Official BEATs Iter3) ===")
    
    # Path to the official checkpoint found in the workspace
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pretrained_models/BEATs_iter3_plus_AS2M.pt'))
    
    if not os.path.exists(ckpt_path):
        print(f"ERROR: BEATs checkpoint not found at {ckpt_path}")
        return

    xgb_df = get_loocv_df()
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    
    # Passing the checkpoint path to the model factory
    model_fn = lambda: DorrBEATsOfficial(ckpt_path=ckpt_path, num_classes=3)
    
    # BEATs handles its own preprocessing, so transform_fn is None (passes raw audio)
    report, macro_f1 = run_loocv(
        model_fn, xgb_df,
        epochs=10, lr=1e-3, batch_size=16,
        log_dir=log_dir,
        transform_fn=None 
    )
    save_report(f"Macro F1: {macro_f1:.4f}\n\n{report}",
                os.path.join(os.path.dirname(__file__), 'results.txt'))

if __name__ == "__main__":
    main()
