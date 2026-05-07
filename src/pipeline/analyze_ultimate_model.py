import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# --- ULTIMATE CONFIGURATION (from 20260302 Joint Search) ---
CONFIG = {
    'N_Mult': 3.0,
    'A_Mult': 5.0,
    'T_N': 0.70,
    'T_A': 0.25
}
OUTPUT_DIR = "./results/plots"
FEATURE_CACHE = "./results/features_3layer_mean.pkl"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Features
    if not os.path.exists(FEATURE_CACHE):
        print(f"Error: Could not find features at {FEATURE_CACHE}. Please run run_ultimate_search.py first.")
        return
        
    with open(FEATURE_CACHE, 'rb') as f:
        X, y, groups = pickle.load(f)
        
    unique_files = np.unique(groups)
    all_probs, all_targets = [], []
    
    print(f">>> Reconstructing ultimate model performance (0.8894 Pipeline)...")
    
    # LOOCV Loop to get unbiased confusion matrix
    for test_file in unique_files:
        test_mask = (groups == test_file)
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Calculate weights based on config
        n_total = len(y_train)
        weight_map = {}
        for cls in np.unique(y_train):
            n_c = np.sum(y_train == cls)
            base_w = n_total / (3 * n_c)
            if cls == 1: base_w *= CONFIG['N_Mult']
            elif cls == 2: base_w *= CONFIG['A_Mult']
            weight_map[cls] = base_w
        
        weights = np.array([weight_map.get(lbl, 1.0) for lbl in y_train])
        
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 4,
            'eta': 0.1,
            'verbosity': 0,
            'tree_method': 'hist'
        }
        
        bst = xgb.train(params, dtrain, num_boost_round=100)
        preds = bst.predict(dtest)
        
        all_probs.extend(preds)
        all_targets.extend(y_test)
        
    # Apply optimal thresholds
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    final_preds = []
    for p in all_probs:
        if p[2] >= CONFIG['T_A']: final_preds.append(2)
        elif p[1] >= CONFIG['T_N']: final_preds.append(1)
        else: final_preds.append(0)
        
    # --- Visualization 1: Confusion Matrix ---
    target_names = ['No-Breathing', 'Normal', 'Abnormal']
    cm = confusion_matrix(all_targets, final_preds)
    f1 = f1_score(all_targets, final_preds, average='macro')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 24})

    plt.title(f'Ultimate Model Confusion Matrix (F1={f1:.4f})\nConfig: N_Mult={CONFIG["N_Mult"]}, A_Mult={CONFIG["A_Mult"]}, T_N={CONFIG["T_N"]}, T_A={CONFIG["T_A"]}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = f"{OUTPUT_DIR}/ultimate_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # --- Visualization 2: Feature Importance ---
    # To get stable importance, train one final model on the COMPLETE set
    print(">>> Calculating stable feature importance on full dataset...")
    n_total_full = len(y)
    weight_map_full = {}
    for cls in np.unique(y):
        n_c = np.sum(y == cls)
        base_w = n_total_full / (3 * n_c)
        if cls == 1: base_w *= CONFIG['N_Mult']
        elif cls == 2: base_w *= CONFIG['A_Mult']
        weight_map_full[cls] = base_w
    
    weights_full = np.array([weight_map_full.get(lbl, 1.0) for lbl in y])
    feature_names = [f"AST_{i}" for i in range(X.shape[1])]
    
    dtrain_full = xgb.DMatrix(X, label=y, weight=weights_full, feature_names=feature_names)
    final_bst = xgb.train(params, dtrain_full, num_boost_round=100)
    
    score_dict = final_bst.get_score(importance_type='gain')
    sorted_idx = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:25]
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x=[v for k,v in sorted_idx], y=[k for k,v in sorted_idx], palette='viridis')
    plt.title('Top 25 AST Feature Importances (Signal Gain)')
    plt.xlabel('Importance (Gain)')
    plt.ylabel('AST Embedding Dimension')
    plt.tight_layout()
    fi_path = f"{OUTPUT_DIR}/ultimate_feature_importance.png"
    plt.savefig(fi_path)
    print(f"Feature importance graph saved to {fi_path}")
    
    print("\n" + "="*50)
    print("FINAL ULTIMATE STATISTICS")
    print("="*50)
    print(classification_report(all_targets, final_preds, target_names=target_names))

if __name__ == "__main__":
    main()
