import torch
import torch.nn as nn
import torchaudio

class LeNet5Fusion(nn.Module):
    """
    Shen 2022: CQT/STFT + LeNet-5 feature extractor.
    Since torchaudio CQT might require additional libraries like nnAudio,
    we use standard STFT/Spectrogram as a proxy, simulating the dual-input fusion.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512)
        
        # LeNet-5 inspired
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((10, 10)) # Fix dimensions
        )
        
        self.embedding = nn.Sequential(
            nn.Linear(16 * 10 * 10, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(84, num_classes)
        
    def get_embedding(self, x):
        # x shape (batch, length)
        spec = self.stft(x) # (batch, freq, time)
        spec = spec.unsqueeze(1) # Add channel dim
        feat = self.features(spec)
        feat = feat.view(feat.size(0), -1)
        emb = self.embedding(feat)
        return emb

    def forward(self, x):
        emb = self.get_embedding(x)
        return self.classifier(emb)
