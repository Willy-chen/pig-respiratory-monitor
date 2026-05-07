import os
import sys
import json
try:
    from catboost import CatBoostClassifier
except ImportError:
    print("CatBoost not installed. Please run: pip install catboost")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import load_ast_features
from train_evaluate import run_loocv

def train_catboost(X_train, y_train, weights_train):
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=4,
        loss_function='MultiClass',
        verbose=0,
        random_seed=42,
        auto_class_weights='Balanced',
        thread_count=-1
    )
    model.fit(X_train, y_train)
    return model

def predict_catboost(model, X_test):
    preds = model.predict(X_test)
    return preds.flatten() # CatBoost returns (N, 1)

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    
    print("\n--- Running Baseline: CatBoost ---")
    results = run_loocv(X, y, groups, train_catboost, predict_catboost)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results.json")
