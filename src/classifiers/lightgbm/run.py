import os
import sys
import json
import lightgbm as lgb
import warnings

# Suppress annoying LightGBM verbosity
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import load_ast_features
from train_evaluate import run_loocv

def train_lgb(X_train, y_train, weights_train):
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model

def predict_lgb(model, X_test):
    return model.predict(X_test)

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    
    print("\n--- Running Baseline: LightGBM ---")
    results = run_loocv(X, y, groups, train_lgb, predict_lgb)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results.json")
