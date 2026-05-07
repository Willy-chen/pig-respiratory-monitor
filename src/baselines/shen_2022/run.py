import os, sys
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import logging, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import get_loocv_df, get_loocv_folds, make_loaders
from train_evaluate import train_one_epoch, setup_logger
from baselines.shen_2022.model import LeNet5Fusion
from train_evaluate import save_report, CLASS_NAMES

def main():
    print("=== Shen et al. 2022 Baseline (LeNet-5 + SVM) ===")
    xgb_df = get_loocv_df()
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, 'training.log'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unique_files = xgb_df['Filename'].unique()
    all_preds, all_targets = [], []
    total_start = time.time()

    logger.info(f"Starting LOOCV over {len(unique_files)} files | Shen 2022 (LeNet-5 + SVM)")

    for fold_idx, (train_ds, test_ds, test_file) in enumerate(get_loocv_folds(xgb_df)):
        fold_start = time.time()
        logger.info(f"--- Fold {fold_idx+1}/{len(unique_files)} | Test file: {test_file} ---")
        train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=32)

        model = LeNet5Fusion(num_classes=3).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Phase 1: Pre-train CNN
        for epoch in range(5):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            logger.info(f"  [CNN pre-train] Epoch {epoch+1}/5 | Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f}")

        # Phase 2: Extract embeddings for SVM
        def extract(loader):
            model.eval()
            Xs, ys = [], []
            with torch.no_grad():
                for X, y in loader:
                    emb = model.get_embedding(X.to(device, dtype=torch.float32))
                    Xs.append(emb.cpu().numpy())
                    ys.append(y.numpy())
            return np.vstack(Xs), np.concatenate(ys)

        X_tr, y_tr = extract(train_loader)
        X_te, y_te = extract(test_loader)

        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm.fit(X_tr, y_tr)
        fold_preds = svm.predict(X_te)

        all_preds.extend(fold_preds)
        all_targets.extend(y_te)

        fold_f1 = f1_score(y_te, fold_preds, average='macro', zero_division=0)
        logger.info(f"  Fold {fold_idx+1} | SVM Macro F1: {fold_f1:.4f} | Time: {time.time()-fold_start:.1f}s")

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    logger.info(f"LOOCV FINAL | Macro F1: {macro_f1:.4f}\n{report}")
    save_report(f"Macro F1: {macro_f1:.4f}\n\n{report}", os.path.join(os.path.dirname(__file__), 'results.txt'))

if __name__ == "__main__":
    main()
