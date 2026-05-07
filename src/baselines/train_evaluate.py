"""
train_evaluate.py — Standardized LOOCV Training & Evaluation Loop
==================================================================
Implements file-level Leave-One-Out Cross Validation, matching the protocol
used by 20260302_ultimate. Provides full logging of epoch loss, train/val
accuracy, and time taken per fold and overall.
"""

import time
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

CLASS_NAMES = ['No-Breathing', 'Normal', 'Abnormal']


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
            out = model(X)
            total_loss += criterion(out, y).item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def run_loocv(model_fn, xgb_df, epochs=10, lr=1e-3, batch_size=32, log_dir='.', device=None, transform_fn=None):
    """
    Full LOOCV run. model_fn is a callable that returns a freshly-initialized model.
    Returns the final classification report string.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, 'training.log'))

    from data_loader import get_loocv_folds, make_loaders

    all_preds, all_targets = [], []
    unique_files = xgb_df['Filename'].unique()
    total_start = time.time()

    logger.info(f"Starting LOOCV over {len(unique_files)} files | epochs={epochs} | lr={lr} | device={device}")

    for fold_idx, (train_ds, test_ds, test_file) in enumerate(get_loocv_folds(xgb_df, transform_fn=transform_fn)):
        fold_start = time.time()
        logger.info(f"--- Fold {fold_idx+1}/{len(unique_files)} | Test file: {test_file} ---")

        train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=batch_size)

        model = model_fn().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate_epoch(model, test_loader, criterion, device)
            logger.info(
                f"  Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        # Collect this fold's predictions
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device, dtype=torch.float32)
                preds = model(X).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.numpy())

        fold_elapsed = time.time() - fold_start
        logger.info(f"  Fold {fold_idx+1} completed in {fold_elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal LOOCV time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    report = classification_report(
        all_targets, all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    logger.info(f"\n{'='*50}")
    logger.info(f"LOOCV FINAL RESULT | Macro F1: {macro_f1:.4f}")
    logger.info(f"\n{report}")

    return report, macro_f1


def save_report(report: str, filepath: str):
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"Results saved to {filepath}")
