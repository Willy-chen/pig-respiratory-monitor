import os
import sys
import json
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import load_ast_features
from train_evaluate import run_loocv
from nn_utils import train_pytorch, predict_pytorch

class MLP3Head(nn.Module):
    def __init__(self, input_dim=768, hidden_dim1=256, hidden_dim2=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_mlp3(X_train, y_train, weights_train):
    model = MLP3Head()
    return train_pytorch(model, X_train, y_train, weights_train, epochs=20, lr=1e-3)

def predict_mlp3(model, X_test):
    return predict_pytorch(model, X_test)

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    
    print("\n--- Running Baseline: 3-Layer MLP Head ---")
    results = run_loocv(X, y, groups, train_mlp3, predict_mlp3)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results.json")
