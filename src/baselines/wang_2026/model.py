import torch
import torch.nn as nn
import torchaudio

class CoughRNet(nn.Module):
    """
    Wang et al., 2026: CoughRNet (CNN-SVM fusion of Spectrogram and Thermal Imaging).
    Dataset lacks thermal imaging, so we duplicate the acoustic feature branch to preserve structural fusion logic.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
        
        # Branch 1: Acoustic
        self.acoustic_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        # Branch 2: Visual (Mocked Thermal)
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fusion_fc = nn.Linear(32*4*4 + 32*4*4, 128)
        self.classifier = nn.Linear(128, num_classes)
        
    def get_embedding(self, x):
        s = self.spec(x).unsqueeze(1) # (batch, 1, mels, time)
        
        # Audio Branch
        feat_audio = self.acoustic_cnn(s).flatten(1)
        
        # Visual Branch (Simulating thermal crop with repeated spectrogram)
        feat_visual = self.visual_cnn(s).flatten(1)
        
        # Early Fusion
        fused = torch.cat([feat_audio, feat_visual], dim=1)
        emb = torch.relu(self.fusion_fc(fused))
        return emb

    def forward(self, x):
        return self.classifier(self.get_embedding(x))
