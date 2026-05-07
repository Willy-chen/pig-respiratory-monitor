import os
import sys
import json
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import load_ast_features
from train_evaluate import run_loocv
from nn_utils import train_pytorch, predict_pytorch

class MLP2Head(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_mlp2(X_train, y_train, weights_train):
    model = MLP2Head()
    return train_pytorch(model, X_train, y_train, weights_train, epochs=20, lr=1e-3)

def predict_mlp2(model, X_test):
    return predict_pytorch(model, X_test)

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    
    print("\n--- Running Baseline: 2-Layer MLP Head ---")
    results = run_loocv(X, y, groups, train_mlp2, predict_mlp2)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results.json")
