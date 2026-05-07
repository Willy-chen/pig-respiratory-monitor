import torch
import torch.nn as nn
import torchaudio

class PigCoughCNN(nn.Module):
    """
    MDPI/ResearchGate 2026: PigCough-CNN.
    Co-training on Spectrograms and 8 MFCCs. We implement a dual-input CNN.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        # Audio transformations
        self.spec = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512)
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=8)
        
        # Spectrogram CNN path
        self.cnn_spec = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # MFCC CNN path (treat MFCC map as 2D image)
        self.cnn_mfcc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32*4*4 + 32*4*4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        s = self.spec(x).unsqueeze(1) # (batch, 1, freqs, time)
        m = self.mfcc(x).unsqueeze(1) # (batch, 1, 8, time)
        
        feat_s = self.cnn_spec(s).flatten(1)
        feat_m = self.cnn_mfcc(m).flatten(1)
        
        fused = torch.cat([feat_s, feat_m], dim=1)
        return self.fc(fused)
