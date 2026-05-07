import os
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import data_utils
from sklearn.metrics import confusion_matrix, classification_report

# Configuration (Best Parameters for 20260209_n)
NORMAL_MULT = 2.0
ABNORMAL_MULT = 5.0
THRESH_NORM = 0.70
THRESH_ABN = 0.40
FEATURE_FILE = "./xgb_results/features_comprehensive.pkl"
OUTPUT_CSV = "./xgb_results/wrong_predictions.csv"

def main():
    print(">>> Analyzing Misclassified Samples for 20260209_n...")
    
    # 1. Load Data/Features
    full_df = data_utils.get_full_dataset()
    _, xgb_df = data_utils.create_study_split(full_df)
    
    if not os.path.exists(FEATURE_FILE):
        print(f"Error: {FEATURE_FILE} not found. Run pipeline first.")
        return

    with open(FEATURE_FILE, 'rb') as f:
        X, y, groups, feature_names = pickle.load(f)
        
    print(f"Loaded {len(X)} samples.")
    
    unique_files = np.unique(groups)
    all_errors = []
    
    # 2. LOOCV Loop
    print(f">>> Running LOOCV with N={NORMAL_MULT}x, A={ABNORMAL_MULT}x...")
    
    for test_file in unique_files:
        test_mask = (groups == test_file)
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        test_indices = np.where(test_mask)[0]
        
        # Dynamic Weights
        classes_in_fold = np.unique(y_train)
        n_total = len(y_train)
        weight_map = {}
        for cls in classes_in_fold:
            n_c = np.sum(y_train == cls)
            if n_c == 0: continue
            base_w = n_total / (3 * n_c)
            if cls == 1: base_w *= NORMAL_MULT
            elif cls == 2: base_w *= ABNORMAL_MULT
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
        preds_proba = bst.predict(dtest)
        
        final_preds = []
        for p in preds_proba:
            if p[2] >= THRESH_ABN: final_preds.append(2)
            elif p[1] >= THRESH_NORM: final_preds.append(1)
            else: final_preds.append(0)
            
        for i, pred_lbl in enumerate(final_preds):
            true_lbl = y_test[i]
            if pred_lbl != true_lbl:
                idx_in_df = test_indices[i]
                row = xgb_df.iloc[idx_in_df]
                
                all_errors.append({
                    'Filename': row['Filename'],
                    'Start': row['Start'],
                    'End': row['End'],
                    'True_Label': int(true_lbl),
                    'Pred_Label': int(pred_lbl),
                    'Prob_0': preds_proba[i][0],
                    'Prob_1': preds_proba[i][1],
                    'Prob_2': preds_proba[i][2]
                })

    err_df = pd.DataFrame(all_errors)
    if len(err_df) > 0:
        err_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n>>> Visualization of Errors saved to {OUTPUT_CSV}")
        print(f"Total Errors: {len(err_df)} / {len(X)} ({len(err_df)/len(X)*100:.1f}%)")
        print(err_df.head(10))
    else:
        print("Amazing! No errors found.")

if __name__ == "__main__":
    main()
