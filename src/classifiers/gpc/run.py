import os
import sys
import json
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import load_ast_features
from train_evaluate import run_loocv

def train_gpc(X_train, y_train, weights_train):
    # GP is very memory intensive O(N^3). N is small enough here hopefully.
    kernel = 1.0 * RBF(1.0)
    model = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def predict_gpc(model, X_test):
    return model.predict(X_test)

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    
    print("\n--- Running Baseline: Gaussian Process Classifier ---")
    results = run_loocv(X, y, groups, train_gpc, predict_gpc)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results.json")
