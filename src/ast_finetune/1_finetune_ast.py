import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ASTForAudioClassification, ASTFeatureExtractor
from sklearn.metrics import f1_score
from tqdm import tqdm
import data_utils

# Config
OUTPUT_DIR = "./best_ast_model"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-5

class PigDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, sr = librosa.load(row['Audio_Path'], sr=16000, offset=row['Start'], duration=10.0)
        
        # Pad if short
        target_len = 16000 * 10
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
            
        inputs = self.processor(y, sampling_rate=16000, return_tensors="pt")
        # Remap labels if necessary. Assuming 1=Normal, 2=Abnormal.
        # AST needs 0, 1, ...
        # If we only have 1 and 2, map 1->0, 2->1? 
        # Or keep 0=Noise, 1=Normal, 2=Abnormal.
        # Based on user description, loss weights exist for Class 0.
        # But our threshold filter might have removed Class 0.
        # We will assume standard mapping.
        return inputs.input_values.squeeze(0), torch.tensor(row['Target'], dtype=torch.long)

import librosa

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    full_df = data_utils.get_full_dataset()
    
    # 2. Strict Split
    ast_df, _ = data_utils.create_study_split(full_df)
    
    # 3. Internal Train/Val Split for AST (File-aware)
    unique_files = ast_df['Filename'].unique()
    np.random.shuffle(unique_files)
    split_idx = int(len(unique_files) * 0.8)
    train_files = unique_files[:split_idx]
    val_files = unique_files[split_idx:]
    
    train_df = ast_df[ast_df['Filename'].isin(train_files)]
    val_df = ast_df[ast_df['Filename'].isin(val_files)]
    
    print("\n" + "="*40)
    print("AST DATASET SPLIT DETAILS")
    print("="*40)
    print(f"Total AST Set Samples: {len(ast_df)}")
    
    def print_dist(name, df):
        counts = df['Target'].value_counts().to_dict()
        print(f"\n{name} Set: {len(df)} samples")
        for cls in sorted(counts.keys()):
            print(f"  Class {cls}: {counts[cls]}")
            
    print_dist("Training", train_df)
    print_dist("Validation", val_df)
    print("="*40 + "\n")
    
    # 4. Setup
    processor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    # Check max label to set num_labels
    num_labels = 3 # 0, 1, 2
    model = ASTForAudioClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True)
    model.to(DEVICE)
    
    train_loader = DataLoader(PigDataset(train_df, processor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PigDataset(val_df, processor), batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    best_f1 = -1
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        preds, targets = [], []
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs, labels=labels)
                val_loss += outputs.loss.item()
                preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
                
        f1 = f1_score(targets, preds, average='macro')
        print(f"Epoch {epoch+1} Val Loss: {val_loss/len(val_loader):.4f} Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            print(">>> Saved Best Model")

if __name__ == "__main__":
    train()
