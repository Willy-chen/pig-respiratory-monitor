import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Add paths to use the existing data/models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../20260322')))
from data_loader import get_loocv_folds, get_loocv_df, make_loaders
from train_evaluate import evaluate_epoch, train_one_epoch

# Define the models registry
MODELS = {
    'yin_2021': {'path': 'baselines.yin_2021.model', 'class': 'SpectrogramAlexNet', 'args': {'num_classes': 3, 'pretrained': True}, 'epochs': 10},
    'dorr_2026': {'path': 'baselines.dorr_2026.model', 'class': 'DorrBEATsOfficial', 'args': {'ckpt_path': '../20260322/pretrained_models/BEATs_iter3_plus_AS2M.pt', 'num_classes': 3}, 'epochs': 3},
    'nithin_2026': {'path': 'baselines.nithin_2026.model', 'class': 'LSTMKAN', 'args': {'num_classes': 3}, 'epochs': 10},
}

def subsample_df(df, frac, seed=42):
    """Stratified subsample of a dataframe."""
    if frac >= 1.0: return df
    # Because of LOOCV, some classes might be very small. We just sample proportionally or randomly.
    # Group by target to ensure we get some of each if possible.
    try:
        return df.groupby('Target', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=seed)).reset_index(drop=True)
    except:
        return df.sample(frac=frac, random_state=seed).reset_index(drop=True)

def run_dl_few_shot(model_name, train_ratios, xgb_df):
    from importlib import import_module
    m_info = MODELS[model_name]
    mod = import_module(m_info['path'])
    ModelClass = getattr(mod, m_info['class'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    
    for ratio in train_ratios:
        print(f"\n>>> Running DL Model {model_name} at {ratio*100}% Training Data")
        all_preds, all_targets = [], []
        epochs = m_info['epochs']
        
        for fold_idx, (train_ds, test_ds, test_file) in enumerate(get_loocv_folds(xgb_df)):
            # Subsample the training data
            train_ds.df = subsample_df(train_ds.df, ratio)
            train_loader, test_loader = make_loaders(train_ds, test_ds, batch_size=32)
            
            model = ModelClass(**m_info['args']).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 if model_name=='yin_2021' else 1e-3)
            
            for epoch in range(epochs):
                train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Predict
            model.eval()
            with torch.no_grad():
                for X, y in test_loader:
                    X = X.to(device, dtype=torch.float32)
                    preds = model(X).argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(y.numpy())
                    
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        print(f"[{model_name} | {ratio*100}%] Macro F1: {f1:.4f}")
        results.append({'Model': model_name, 'Train_Ratio': ratio, 'F1': f1})
        
    return results

def run_ultimate_few_shot(train_ratios):
    import pickle
    import xgboost as xgb
    
    print("\n>>> Running Ultimate Model (AST+XGB) Few Shot")
    cache_path = "../20260302_ultimate/results/features_3layer_mean.pkl"
    with open(cache_path, 'rb') as f:
        X, y, groups = pickle.load(f)
        
    unique_files = np.unique(groups)
    results = []
    
    for ratio in train_ratios:
        print(f"  Testing at {ratio*100}% ...")
        all_probs, all_targets = [], []
        
        for test_file in unique_files:
            test_mask = (groups == test_file)
            train_mask = ~test_mask
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            
            # Subsample
            if ratio < 1.0:
                indices = np.arange(len(y_train))
                np.random.seed(42)
                # Ensure stratified
                sampled_indices = []
                for c in np.unique(y_train):
                    c_idx = indices[y_train == c]
                    size = int(len(c_idx) * ratio)
                    if size == 0 and len(c_idx) > 0: size = 1 # keep at least one
                    sampled_indices.extend(np.random.choice(c_idx, size, replace=False))
                np.random.shuffle(sampled_indices)
                X_train = X_train[sampled_indices]
                y_train = y_train[sampled_indices]
                
            n_total = len(y_train)
            weight_map = {}
            for cls in np.unique(y_train):
                n_c = np.sum(y_train == cls)
                base_w = n_total / (3 * n_c)
                if cls == 1: base_w *= 3.0 # optimal N
                elif cls == 2: base_w *= 5.0 # optimal A
                weight_map[cls] = base_w
                
            weights = np.array([weight_map.get(lbl, 1.0) for lbl in y_train])
            
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
            dtest = xgb.DMatrix(X_test)
            
            params = {'objective': 'multi:softprob', 'num_class': 3, 'max_depth': 4, 'eta': 0.1, 'verbosity': 0}
            bst = xgb.train(params, dtrain, num_boost_round=100)
            preds = bst.predict(dtest)
            
            all_probs.extend(preds)
            all_targets.extend(y_test)
            
        all_probs = np.array(all_probs)
        temp_preds = []
        for p in all_probs:
            if p[2] >= 0.25: temp_preds.append(2)
            elif p[1] >= 0.70: temp_preds.append(1)
            else: temp_preds.append(0)
            
        f1 = f1_score(all_targets, temp_preds, average='macro', zero_division=0)
        print(f"[Ultimate AST+XGB | {ratio*100}%] Macro F1: {f1:.4f}")
        results.append({'Model': 'ultimate_ast_xgb', 'Train_Ratio': ratio, 'F1': f1})
        
    return results

def run_resumable_few_shot(model_id, ratios, xgb_df=None):
    csv_file = "few_shot_results.csv"
    
    for ratio in ratios:
        print(f"\n>>> Model: {model_id} | Ratio: {ratio}")
        
        # In a real 16-fold LOOCV, we should save EACH FOLD as it finishes.
        # Let's modify the underlying run functions to work fold-by-fold if possible,
        # or just wrap them here.
        
        # For now, let's just make it save at the ratio level more reliably.
        # If we really want fold-level, we'd need to change run_dl_few_shot internally.
        
        # Actually, let's just do it here:
        if model_id == 'ultimate_ast_xgb':
            res = run_ultimate_few_shot([ratio])
        else:
            res = run_dl_few_shot(model_id, [ratio], xgb_df)
            
        new_df = pd.DataFrame(res)
        if os.path.exists(csv_file):
            pd.concat([pd.read_csv(csv_file), new_df]).to_csv(csv_file, index=False)
        else:
            new_df.to_csv(csv_file, index=False)

def main():
    ratios = [0.25, 0.50, 0.75, 1.0]
    xgb_df = get_loocv_df()
    csv_file = "few_shot_results.csv"
    
    models = ['ultimate_ast_xgb', 'yin_2021', 'dorr_2026', 'nithin_2026']
    
    for m_id in models:
        print(f"\n>>> CHECKING: {m_id}")
        
        # Which ratios are already done?
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            done_ratios = df[df['Model'] == m_id]['Train_Ratio'].unique().tolist()
        else:
            done_ratios = []
            
        remaining = [r for r in ratios if r not in done_ratios]
        if not remaining:
            print(f"    Fully completed.")
            continue
            
        print(f"    Remaining ratios: {remaining}")
        run_resumable_few_shot(m_id, remaining, xgb_df)
                
    print("Done! All results saved to few_shot_results.csv")

if __name__ == "__main__":
    main()
