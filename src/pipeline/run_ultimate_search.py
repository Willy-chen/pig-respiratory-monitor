import os
import sys
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
import pickle

# Configuration
AST_MODEL_PATH = "../20260209_n/best_ast_model"
OUTPUT_DIR = "./results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimal Feature strategy from ablation
NUM_LAYERS = 3
POOL_METHOD = 'mean'

sys.path.append(os.path.abspath('.'))
import data_utils

def extract_optimal_features(df):
    print(f">>> Extracting AST Features (Last {NUM_LAYERS} Layers, {POOL_METHOD})...")
    model = ASTModel.from_pretrained(AST_MODEL_PATH, output_hidden_states=True)
    model.to(DEVICE)
    model.eval()
    processor = ASTFeatureExtractor.from_pretrained(AST_MODEL_PATH)
    
    features, targets, groups = [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            y, sr = librosa.load(row['Audio_Path'], sr=16000, offset=row['Start'], duration=10.0)
            target_len = 16000 * 10
            if len(y) < target_len: y = np.pad(y, (0, target_len - len(y)))
            else: y = y[:target_len]
            
            with torch.no_grad():
                inputs = processor(y, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
                outputs = model(inputs)
                hidden_states = outputs.hidden_states
                hs_subset = torch.stack(hidden_states[-NUM_LAYERS:])
                avg_layers = torch.mean(hs_subset, dim=0)
                global_pool = torch.mean(avg_layers, dim=1).cpu().numpy().squeeze()
                
            features.append(global_pool)
            targets.append(row['Target'])
            groups.append(row['Filename'])
        except Exception as e:
            pass
    return np.array(features), np.array(targets), np.array(groups)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load/Extract Features
    cache_path = f"{OUTPUT_DIR}/features_3layer_mean.pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        with open(cache_path, 'rb') as f:
            X, y, groups = pickle.load(f)
    else:
        full_df = data_utils.get_full_dataset()
        _, xgb_df = data_utils.create_study_split(full_df)
        X, y, groups = extract_optimal_features(xgb_df)
        with open(cache_path, 'wb') as f:
            pickle.dump((X, y, groups), f)

    unique_files = np.unique(groups)
    
    # Grid Search Candidates (Expanded)
    NORMAL_MULT_CANDS = list(range(1, 11))
    ABNORMAL_MULT_CANDS = list(range(1, 11))
    THRESH_CANDS = np.arange(0.1, 0.95, 0.05)
    
    best_overall_f1 = -1
    best_config = {}
    best_report = ""
    
    results_log = []

    print(f"\n>>> Starting Expanded Joint Grid Search ({len(NORMAL_MULT_CANDS) * len(ABNORMAL_MULT_CANDS)} weight combos)...")
    
    for nm in NORMAL_MULT_CANDS:
        for am in ABNORMAL_MULT_CANDS:
            print(f"  [Testing] Weights: N={nm}x, A={am}x ...", end="", flush=True)
            
            all_probs, all_targets = [], []
            
            # LOOCV
            for test_file in unique_files:
                test_mask = (groups == test_file)
                train_mask = ~test_mask
                X_train, y_train = X[train_mask], y[train_mask]
                X_test, y_test = X[test_mask], y[test_mask]
                
                n_total = len(y_train)
                weight_map = {}
                for cls in np.unique(y_train):
                    n_c = np.sum(y_train == cls)
                    base_w = n_total / (3 * n_c)
                    if cls == 1: base_w *= nm
                    elif cls == 2: base_w *= am
                    weight_map[cls] = base_w
                
                weights = np.array([weight_map.get(lbl, 1.0) for lbl in y_train])
                
                dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
                dtest = xgb.DMatrix(X_test, label=y_test)
                
                params = {'objective': 'multi:softprob', 'num_class': 3, 'max_depth': 4, 'eta': 0.1, 'verbosity': 0}
                bst = xgb.train(params, dtrain, num_boost_round=100)
                preds = bst.predict(dtest)
                
                all_probs.extend(preds)
                all_targets.extend(y_test)
            
            # Sub-grid search for thresholds
            best_t_f1 = -1
            best_t = (0.5, 0.5)
            
            all_probs = np.array(all_probs)
            all_targets = np.array(all_targets)
            
            for t1 in THRESH_CANDS:
                for t2 in THRESH_CANDS:
                    temp_preds = []
                    for p in all_probs:
                        if p[2] >= t2: temp_preds.append(2)
                        elif p[1] >= t1: temp_preds.append(1)
                        else: temp_preds.append(0)
                    
                    f1 = f1_score(all_targets, temp_preds, average='macro')
                    
                    # Log every combination
                    results_log.append({
                        'N_Mult': nm, 
                        'A_Mult': am, 
                        'T_N': t1, 
                        'T_A': t2, 
                        'F1': f1
                    })
                    
                    if f1 > best_t_f1:
                        best_t_f1 = f1
                        best_t = (t1, t2)
            
            print(f" Best F1: {best_t_f1:.4f} (T_N={best_t[0]:.2f}, T_A={best_t[1]:.2f})")
            
            if best_t_f1 > best_overall_f1:
                best_overall_f1 = best_t_f1
                best_config = {'N_Mult': nm, 'A_Mult': am, 'T_N': best_t[0], 'T_A': best_t[1]}
                
                # Capture best report
                final_preds = []
                for p in all_probs:
                    if p[2] >= best_t[1]: final_preds.append(2)
                    elif p[1] >= best_t[0]: final_preds.append(1)
                    else: final_preds.append(0)
                best_report = classification_report(all_targets, final_preds, target_names=['No-Breathing', 'Normal', 'Abnormal'], labels=[0,1,2])
                
    # Final Output
    print("\n" + "="*50)
    print("JOINT OPTIMIZATION BEST RESULT")
    print("="*50)
    print(f"Macro F1: {best_overall_f1:.4f}")
    print(f"Config: {best_config}")
    print("\n" + best_report)
    
    # Save massive log
    print(f">>> Saving full log ({len(results_log)} rows) to results/joint_optimization_log.csv")
    pd.DataFrame(results_log).to_csv(f"{OUTPUT_DIR}/joint_optimization_log.csv", index=False)
    
    with open(f"{OUTPUT_DIR}/best_config.txt", "w") as f:
        f.write(f"Best Macro F1: {best_overall_f1:.4f}\n")
        f.write(f"Config: {best_config}\n\n")
        f.write(best_report)

if __name__ == "__main__":
    main()
