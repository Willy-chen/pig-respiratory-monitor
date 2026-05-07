import os
import sys
import json
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import load_ast_features
from train_evaluate import run_loocv

def train_rf(X_train, y_train, weights_train):
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def predict_rf(model, X_test):
    return model.predict(X_test)

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    
    print("\n--- Running Baseline: Random Forest ---")
    results = run_loocv(X, y, groups, train_rf, predict_rf)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results.json")
