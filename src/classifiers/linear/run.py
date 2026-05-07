import os
import sys
import json
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_loader import load_ast_features
from train_evaluate import run_loocv
from nn_utils import train_pytorch, predict_pytorch

class LinearHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

def train_linear(X_train, y_train, weights_train):
    model = LinearHead()
    return train_pytorch(model, X_train, y_train, weights_train, epochs=20, lr=1e-3)

def predict_linear(model, X_test):
    return predict_pytorch(model, X_test)

if __name__ == "__main__":
    X, y, groups = load_ast_features()
    
    print("\n--- Running Baseline: Linear Head (AST -> 3) ---")
    results = run_loocv(X, y, groups, train_linear, predict_linear)
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to results.json")
