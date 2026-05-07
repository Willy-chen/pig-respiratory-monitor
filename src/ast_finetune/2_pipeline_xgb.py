import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import librosa
from transformers import ASTModel, ASTFeatureExtractor
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import data_utils
import pickle
import feature_utils

# Configuration
AST_MODEL_PATH = "./best_ast_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./xgb_results"

def extract_features(df):
    print(">>> Extracting Features (AST + Traditional)...")
    
    # Load AST Backbone
    model = ASTModel.from_pretrained(AST_MODEL_PATH, output_hidden_states=True)
    model.to(DEVICE)
    model.eval()
    processor = ASTFeatureExtractor.from_pretrained(AST_MODEL_PATH)
    
    features = []
    targets = []
    groups = [] 
    
    feature_names = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Load Audio
            y, sr = librosa.load(row['Audio_Path'], sr=16000, offset=row['Start'], duration=10.0)
            target_len = 16000 * 10
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
            
            # 1. Traditional (New Comprehensive Set)
            trad_dict = feature_utils.extract_all_features(y, sr=16000)
            trad_feat = np.array(list(trad_dict.values()))
            if feature_names is None:
                # AST names + Trad names
                trad_names = list(trad_dict.keys())
                feature_names = [f"AST_{i}" for i in range(768)] + trad_names
            
            # 2. AST
            with torch.no_grad():
                inputs = processor(y, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
                outputs = model(inputs)
                hidden_states = outputs.hidden_states[-4:] 
                hs_stack = torch.stack(hidden_states)
                avg_layers = torch.mean(hs_stack, dim=0)
                global_pool = torch.mean(avg_layers, dim=1)
                ast_feat = global_pool.cpu().numpy().squeeze()
            
            # Combine
            final_feat = np.concatenate([ast_feat, trad_feat])
            
            features.append(final_feat)
            targets.append(row['Target'])
            groups.append(row['Filename'])
            
        except Exception as e:
            print(f"Error {row['Filename']} at {row['Start']}: {e}")
            
    return np.array(features), np.array(targets), np.array(groups), feature_names

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    full_df = data_utils.get_full_dataset()
    _, xgb_df = data_utils.create_study_split(full_df)
    
    print(f"XGBoost Pool: {len(xgb_df)} samples from {xgb_df['Filename'].nunique()} files")
    
    # 2. Extract Features
    X, y, groups, feature_names = extract_features(xgb_df)
    
    # Save features cache
    with open(f"{OUTPUT_DIR}/features_comprehensive.pkl", 'wb') as f:
        pickle.dump((X, y, groups, feature_names), f)
        
    # 3. Grid Search Setup
    unique_files = np.unique(groups)
    print(f"\n" + "="*40)
    print(f"XGBoost LOOCV GRID SEARCH")
    print(f"Total Files: {len(unique_files)}")
    print(f"Total Samples: {len(y)}")
    targets_counts = np.bincount(y)
    for i, c in enumerate(targets_counts):
        print(f"  Class {i}: {c}")
    print("="*40 + "\n")

    NORMAL_MULT_CANDIDATES = [1.0, 2.0, 3.0, 4.0, 5.0] 
    ABNORMAL_MULT_CANDIDATES = [1.0, 5.0, 7.0, 10.0, 15.0] 
    # NORMAL_MULT_CANDIDATES = [4.0] 
    # ABNORMAL_MULT_CANDIDATES = [5.0] 
    
    best_loocv_f1 = -1
    best_weight_probs = None
    best_weight_labels = None
    best_nm, best_am = 1.0, 1.0
    
    print(f">>> Starting Weight Search ({len(NORMAL_MULT_CANDIDATES) * len(ABNORMAL_MULT_CANDIDATES)} combinations)...")

    for nm in NORMAL_MULT_CANDIDATES:
        for am in ABNORMAL_MULT_CANDIDATES:
            print(f"  [Testing] Normal={nm}x, Abnormal={am}x ...", end="", flush=True)
            
            all_preds = []
            all_targets = []
            all_probs = []
            
            # LOOCV Loop
            for test_file in unique_files:
                try:
                    test_mask = (groups == test_file)
                    train_mask = ~test_mask
                    if np.sum(test_mask) == 0: continue
                    
                    X_train, y_train = X[train_mask], y[train_mask]
                    X_test, y_test = X[test_mask], y[test_mask]
                    
                    # Dynamic Weight Calculation
                    classes_in_fold = np.unique(y_train)
                    n_classes_fixed = 3
                    n_total = len(y_train)
                    weight_map = {}
                    for cls in classes_in_fold:
                        n_c = np.sum(y_train == cls)
                        base_w = n_total / (n_classes_fixed * n_c)
                        if cls == 1: base_w *= nm
                        elif cls == 2: base_w *= am
                        weight_map[cls] = base_w
                    
                    weights = np.array([weight_map.get(lbl, 1.0) for lbl in y_train])
                    
                    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights, feature_names=feature_names)
                    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
                    
                    params = {
                        'objective': 'multi:softprob',
                        'num_class': 3,
                        'max_depth': 4,
                        'eta': 0.1,
                        'verbosity': 0,
                        'eval_metric': ['mlogloss'],
                        'tree_method': 'hist'
                    }
                    
                    bst = xgb.train(params, dtrain, num_boost_round=100)
                    preds = bst.predict(dtest)
                    
                    all_probs.extend(preds)
                    all_targets.extend(y_test)
                    
                except Exception as e:
                    print(f" Err {test_file}: {e}")
            
            probs_arr = np.array(all_probs)
            preds_arr = np.argmax(probs_arr, axis=1)
            labels_arr = np.array(all_targets)
            
            f1 = f1_score(labels_arr, preds_arr, average='macro')
            print(f" Macro F1: {f1:.4f}")
            
            if f1 > best_loocv_f1:
                best_loocv_f1 = f1
                best_weight_probs = probs_arr
                best_weight_labels = labels_arr
                best_nm, best_am = nm, am

    print(f"\n>>> Best Multipliers: Normal={best_nm}, Abnormal={best_am} (F1={best_loocv_f1:.4f})")
    
    # 4. Threshold Optimization
    print("\n>>> Optimizing Prediction Thresholds (Searching t1, t2)...")
    print(f"    {'Normal (t1)':<12} | {'Abnormal (t2)':<12} | {'Macro F1'}")
    print("-" * 45)
    
    best_t_f1 = -1
    best_thresholds = [0.5, 0.5]
    
    # Grid search for thresholds (0.1 to 0.7)
    for t1 in np.arange(0.1, 0.75, 0.05):
        for t2 in np.arange(0.1, 0.75, 0.05):
            temp_preds = []
            for p in best_weight_probs:
                if p[2] >= t2: temp_preds.append(2)
                elif p[1] >= t1: temp_preds.append(1)
                else: temp_preds.append(0)
            
            f1 = f1_score(best_weight_labels, temp_preds, average='macro')
            
            # Print candidate result (compactly)
            # print(f"    {t1:.2f}        | {t2:.2f}        | {f1:.4f}") 
            # Printing ALL might be too much (12*12=144 lines). 
            # User asked "show the threshold f1 score for each candidate". 
            # I will print the Top 10 or print all if requested? "show... for EACH candidate". 
            # I will print all.
            print(f"    {t1:.2f}        | {t2:.2f}        | {f1:.4f}")
            
            if f1 > best_t_f1:
                best_t_f1 = f1
                best_thresholds = [t1, t2]
                
    print("-" * 45)
    print(f"Best Thresholds: Normal={best_thresholds[0]:.2f}, Abnormal={best_thresholds[1]:.2f}")
    
    # 5. Final Report
    final_preds = []
    t1, t2 = best_thresholds
    for p in best_weight_probs:
        if p[2] >= t2: final_preds.append(2)
        elif p[1] >= t1: final_preds.append(1)
        else: final_preds.append(0)
        
    target_names = ['No-Breathing', 'Normal', 'Abnormal']
    print("\n" + "="*40)
    print("FINAL OPTIMIZED RESULTS")
    print("="*40)
    print(classification_report(best_weight_labels, final_preds, target_names=target_names, labels=[0,1,2], zero_division=0))
    
    cm = confusion_matrix(best_weight_labels, final_preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Optimized (N={best_nm}x, A={best_am}x, T1={t1:.2f}, T2={t2:.2f})')
    plt.savefig(f"{OUTPUT_DIR}/final_optimized_cm.png")
    
    # Feature Importance (Gain) - Re-calculate on FULL set using best weights to get importance list? 
    # Or just average over folds? The previous code averaged gain.
    # But now we just ran a grid search. 'bst' variable is lost.
    # We should re-train one final model on LOOCV (or just one fold?) to get importance?
    # Better: Accumulate importance during the *Winning* Grid Search iteration? 
    # Too late, logic is nested.
    # Solution: We will report that feature importance requires a dedicated run or just skip for now 
    # as the user emphasized threshold scores. 
    # OR: Just run one final training on the full set (all valid) to get importance.
    
    # Final feature importance on full dataset
    print("Calculating final feature importance...")
    # Base weights
    classes_in_fold = np.unique(y)
    n_classes_fixed = 3
    n_total = len(y)
    weight_map = {}
    for cls in classes_in_fold:
        n_c = np.sum(y == cls)
        base_w = n_total / (n_classes_fixed * n_c)
        if cls == 1: base_w *= best_nm
        elif cls == 2: base_w *= best_am
        weight_map[cls] = base_w
    weights = np.array([weight_map.get(lbl, 1.0) for lbl in y])
    
    dtrain_full = xgb.DMatrix(X, label=y, weight=weights, feature_names=feature_names)
    bst_final = xgb.train(params, dtrain_full, num_boost_round=100)
    
    score_dict = bst_final.get_score(importance_type='gain')
    sorted_idx = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:25]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=[v for k,v in sorted_idx], y=[k for k,v in sorted_idx])
    plt.title('Top 25 Feature Importance (Signal Gain)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")

    print(f"Results saved to {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
