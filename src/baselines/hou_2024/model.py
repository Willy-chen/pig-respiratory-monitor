import torch
import torch.nn as nn
import torchaudio

class MultiFeatureMLP(nn.Module):
    """
    Hou et al. 2024: Multi-feature fusion approach with BP Neural Network (MLP).
    We extract multi-domain features globally from the waveform and pass them into MLP.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        self.spec = torchaudio.transforms.Spectrogram(n_fft=1024)
        
        # Statistics pooling from frames -> global feature array
        # 40 (mfcc stats) * 3 + 513 (spec stats) * 3 = 120 + 1539 = 1659
        self.mlp = nn.Sequential(
            nn.Linear(1659, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Time-domain -> Frequency domain frames
        m = self.mfcc(x)     # (batch, n_mfcc, time)
        s = self.spec(x)     # (batch, freq, time)
        
        # Extract global statistical features
        m_mean, m_var, m_max = m.mean(-1), m.var(-1), m.max(-1).values
        s_mean, s_var, s_max = s.mean(-1), s.var(-1), s.max(-1).values
        
        # FUSION
        feat = torch.cat([m_mean, m_var, m_max, s_mean, s_var, s_max], dim=1) # (batch, 1659)
        return self.mlp(feat)
