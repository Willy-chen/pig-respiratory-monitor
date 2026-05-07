import pickle
import os
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_PATH = os.environ.get(
    "PIG_FEATURES_PKL",
    os.path.join(_REPO_ROOT, "data", "features", "features_3layer_mean.pkl"),
)

def load_ast_features():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Cached AST features not found at {CACHE_PATH}. Run ultimate search first.")
    
    with open(CACHE_PATH, 'rb') as f:
        X, y, groups = pickle.load(f)
    
    print(f"Loaded {len(X)} segments from {len(np.unique(groups))} files.")
    print(f"Feature shape: {X.shape}")
    return X, y, groups

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))
