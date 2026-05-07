import os
import sys
import json
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import load_ast_features
from train_evaluate import run_loocv

def train_knn(X_train, y_train, weights_train):
    # KNN doesn't directly support sample_weights in fit like others,
    # but for simple evaluation, we use default distance weighting.
    model = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def predict_knn(model, X_test):
    return model.predict(X_test)

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    
    print("\n--- Running Baseline: KNN (k=5) ---")
    results = run_loocv(X, y, groups, train_knn, predict_knn)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results.json")
