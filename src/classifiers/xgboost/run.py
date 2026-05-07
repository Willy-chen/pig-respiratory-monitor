import os
import sys
import json
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import load_ast_features
from train_evaluate import run_loocv

def train_xgb(X_train, y_train, weights_train):
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
    params = {
        'objective': 'multi:softprob', 
        'num_class': 3, 
        'max_depth': 4, 
        'eta': 0.1, 
        'verbosity': 0
    }
    bst = xgb.train(params, dtrain, num_boost_round=100)
    return bst

def predict_xgb(model, X_test):
    dtest = xgb.DMatrix(X_test)
    preds_proba = model.predict(dtest)
    return preds_proba.argmax(axis=1)

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    
    print("\n--- Running Baseline: XGBoost ---")
    results = run_loocv(X, y, groups, train_xgb, predict_xgb)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results.json")
