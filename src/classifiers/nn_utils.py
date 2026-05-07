import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def train_pytorch(model, X_train, y_train, weights_train, epochs=20, lr=1e-3, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    
    # Extract unique class weights from the per-sample array
    class_weights = torch.ones(3, dtype=torch.float32, device=device)
    for c in range(3):
        mask = (y_train == c)
        if mask.any():
            class_weights[c] = torch.tensor(weights_train[mask][0], dtype=torch.float32)
            
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    return model

def predict_pytorch(model, X_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(X_t)
        preds = out.argmax(dim=1).cpu().numpy()
    return preds
