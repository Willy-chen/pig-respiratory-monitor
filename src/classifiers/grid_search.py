import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score, classification_report

# Ensure dependencies resolve
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_loader import load_ast_features

def get_model_fns(model_name):
    """Factory returning (train_fn(X,y,w), predict_proba_fn(m,X)) for the requested model."""
    if model_name == "xgboost":
        import xgboost as xgb
        def tr(X, y, w):
            dtrain = xgb.DMatrix(X, label=y, weight=w)
            return xgb.train({'objective': 'multi:softprob', 'num_class': 3, 'max_depth': 4, 'eta': 0.1, 'verbosity': 0}, dtrain, num_boost_round=100)
        def pr(m, X):
            return m.predict(xgb.DMatrix(X))
        return tr, pr
        
    elif model_name == "svm":
        from sklearn.svm import SVC
        def tr(X, y, w):
            model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            model.fit(X, y, sample_weight=w)
            return model
        def pr(m, X):
            return m.predict_proba(X)
        return tr, pr
        
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        def tr(X, y, w):
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y, sample_weight=w)
            return model
        def pr(m, X):
            return m.predict_proba(X)
        return tr, pr
        
    elif model_name == "lightgbm":
        import lightgbm as lgb
        def tr(X, y, w):
            model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X, y, sample_weight=w)
            return model
        def pr(m, X):
            return m.predict_proba(X)
        return tr, pr
        
    elif model_name in ["linear", "mlp2", "mlp3"]:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        
        # Define architectures internally for independence
        class LinearHead(nn.Module):
            def __init__(self): super().__init__(); self.fc = nn.Linear(768, 3)
            def forward(self, x): return self.fc(x)
            
        class MLP2Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
            def forward(self, x): return self.net(x)
            
        class MLP3Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3), 
                                         nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))
            def forward(self, x): return self.net(x)

        def tr(X, y, w):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = {"linear": LinearHead, "mlp2": MLP2Head, "mlp3": MLP3Head}[model_name]().to(device)
            
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)
            
            # Convert per-sample weights to unique class weights for CrossEntropy
            class_weights = torch.ones(3, dtype=torch.float32, device=device)
            for c in range(3):
                mask = (y == c)
                if mask.any(): class_weights[c] = torch.tensor(w[mask][0], dtype=torch.float32)
                    
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True, drop_last=True)
            
            model.train()
            for _ in range(20):
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
            return model
            
        def pr(m, X):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            m.eval()
            with torch.no_grad():
                out = m(torch.tensor(X, dtype=torch.float32).to(device))
                return torch.nn.functional.softmax(out, dim=1).cpu().numpy()
        return tr, pr
        
    elif model_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        def tr(X, y, w):
            # KNN doesn't utilize sample_weights natively in distance calculation without hacks, skip weighting
            model = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
            model.fit(X, y)
            return model
        def pr(m, X):
            return m.predict_proba(X)
        return tr, pr

    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        def tr(X, y, w):
            model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, loss_function='MultiClass', verbose=0, random_seed=42, thread_count=-1)
            model.fit(X, y, sample_weight=w)
            return model
        def pr(m, X):
            return m.predict_proba(X)
        return tr, pr
        
    else:
        raise ValueError(f"Model {model_name} not supported for grid search. Try: xgboost, lightgbm, rf, svm, linear, mlp2, mlp3, catboost.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Classifier to grid search grid (e.g. xgboost, mlp2, lightgbm)")
    args = parser.parse_args()
    
    X, y, groups = load_ast_features()
    train_fn, predict_fn = get_model_fns(args.model)
    unique_files = np.unique(groups)
    
    # 1. Multiplier Combinations
    NORMAL_MULT_CANDS = list(range(1, 11))
    ABNORMAL_MULT_CANDS = list(range(1, 11))
    THRESH_CANDS = np.arange(0.1, 0.95, 0.05)
    
    print(f"=== Starting Distributed Grid Search for {args.model.upper()} ===")
    print(f"Grid Size: {len(NORMAL_MULT_CANDS)}x{len(ABNORMAL_MULT_CANDS)} = {len(NORMAL_MULT_CANDS)*len(ABNORMAL_MULT_CANDS)} iterations")
    print(f"Each iteration executes {len(unique_files)} LOOCV Folds")
    print("="*60)
    
    best_overall_f1 = -1
    best_config = {}
    grid_results = []
    
    # Massive Loop
    for nm in NORMAL_MULT_CANDS:
        for am in ABNORMAL_MULT_CANDS:
            print(f"  [Testing] Weights: N={nm}x, A={am}x ...", end="", flush=True)
            t0 = time.time()
            all_probs, all_targets = [], []
            
            # File-Level LOOCV (16 iterations per multiplier combo)
            for test_file in unique_files:
                test_mask = (groups == test_file)
                train_mask = ~test_mask
                X_train, y_train = X[train_mask], y[train_mask]
                X_test, y_test = X[test_mask], y[test_mask]
                
                # Base Sample Weights logic equivalent to 20260302_ultimate
                n_total = len(y_train)
                weight_map = {}
                for cls in np.unique(y_train):
                    n_c = np.sum(y_train == cls)
                    base_w = n_total / (3 * n_c)
                    if cls == 1: base_w *= nm
                    elif cls == 2: base_w *= am
                    weight_map[cls] = base_w
                
                weights = np.array([weight_map.get(lbl, 1.0) for lbl in y_train])
                
                model = train_fn(X_train, y_train, weights)
                preds = predict_fn(model, X_test)
                
                all_probs.extend(preds)
                all_targets.extend(y_test)
                
            # Internal Sub-Grid Search for Decision Thresholds (289 combos)
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
                    if f1 > best_t_f1:
                        best_t_f1 = f1
                        best_t = (t1, t2)
                        
            time_taken = time.time() - t0
            print(f" F1: {best_t_f1:.4f} (T_N={best_t[0]:.2f}, T_A={best_t[1]:.2f}) | Time: {time_taken:.1f}s")
            
            grid_results.append({
                'Normal_Mult': nm,
                'Abnormal_Mult': am,
                'Best_F1': best_t_f1,
                'Best_T_N': best_t[0],
                'Best_T_A': best_t[1]
            })

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
                
    # Final Output Dumping
    print("\n" + "="*50)
    print(f"[{args.model.upper()}] JOINT OPTIMIZATION BEST RESULT")
    print("="*50)
    print(f"Macro F1: {best_overall_f1:.4f}")
    print(f"Config: {best_config}")
    print("\n" + best_report)
    
    out_dir = f"classifiers/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/grid_search_best.txt", "w") as f:
        f.write(f"Best Macro F1: {best_overall_f1:.4f}\n")
        f.write(f"Config: {best_config}\n\n")
        f.write(best_report)
    
    # Save the complete log for graphing
    pd.DataFrame(grid_results).to_csv(f"{out_dir}/grid_search_all.csv", index=False)
    print(f"Complete grid log saved to {out_dir}/grid_search_all.csv")

if __name__ == "__main__":
    main()
