import os
import sys
import json
from sklearn.svm import SVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import load_ast_features
from train_evaluate import run_loocv

def train_svm(X_train, y_train, weights_train):
    model = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_svm(model, X_test):
    return model.predict(X_test)

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    
    print("\n--- Running Baseline: SVM (RBF) ---")
    results = run_loocv(X, y, groups, train_svm, predict_svm)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results.json")
