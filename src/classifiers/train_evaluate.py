import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

def calculate_class_weights(y, n_classes=3):
    """Calculate balanced class weights to handle imbalanced datasets."""
    weights = np.ones(len(y))
    n_total = len(y)
    for cls in range(n_classes):
        n_c = np.sum(y == cls)
        if n_c > 0:
            w = n_total / (n_classes * n_c)
            weights[y == cls] = w
    return weights

def run_loocv(X, y, groups, train_fn, predict_fn, **kwargs):
    """
    Standardized File-Level Leave-One-Out Cross Validation.
    
    Args:
        X: Feature matrix (N, 768)
        y: Target labels (N,)
        groups: Filename groups for LOOCV (N,)
        train_fn: Callable `train_fn(X_train, y_train, weights_train, **kwargs)` returning a trained model
        predict_fn: Callable `predict_fn(model, X_test, **kwargs)` returning raw predictions (N_test, ) or probas
        **kwargs: Additional args passed to train/predict
        
    Returns:
        dict: Detailed metrics including Macro F1, Accuracy, Train Time, and Latency
    """
    unique_files = np.unique(groups)
    n_files = len(unique_files)
    
    all_targets = []
    all_preds = []
    
    total_train_time = 0.0
    total_inference_time = 0.0
    total_test_samples = 0
    
    print(f"Starting LOOCV over {n_files} files...")
    
    for i, test_file in enumerate(unique_files):
        test_mask = (groups == test_file)
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        weights_train = calculate_class_weights(y_train)
        
        # --- Training Phase ---
        t0 = time.time()
        model = train_fn(X_train, y_train, weights_train, **kwargs)
        t_train = time.time() - t0
        total_train_time += t_train
        
        # --- Inference Phase ---
        t0 = time.time()
        preds = predict_fn(model, X_test, **kwargs)
        t_infer = time.time() - t0
        
        total_inference_time += t_infer
        total_test_samples += len(X_test)
        
        all_targets.extend(y_test)
        all_preds.extend(preds)
        
        print(f"  Fold {i+1}/{n_files} [{test_file}] - Train Time: {t_train:.2f}s, Test Samples: {len(X_test)}")
        
    # --- Aggregate Metrics ---
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    acc = accuracy_score(all_targets, all_preds)
    
    # Calculate Latency per sample in milliseconds
    latency_ms = (total_inference_time / total_test_samples) * 1000
    
    print("\n" + "="*40)
    print("LOOCV RESULTS")
    print("="*40)
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Total Train Time:  {total_train_time:.2f}s")
    print(f"Inference Latency: {latency_ms:.4f} ms/sample")
    print("\nClassification Report:")
    report = classification_report(all_targets, all_preds, target_names=['No-Breathing', 'Normal', 'Abnormal'], zero_division=0)
    print(report)
    
    return {
        'macro_f1': macro_f1,
        'accuracy': acc,
        'total_train_time_s': total_train_time,
        'inference_latency_ms': latency_ms,
        'report': report
    }
