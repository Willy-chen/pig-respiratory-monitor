import torch
import torch.nn as nn
import torchaudio

class KANLinearPlaceholder(nn.Module):
    """
    Dummy approximation if PyKAN is not installed in the environment.
    A true KAN uses learnable activation functions on edges.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.SiLU(), # SiLU is often used as a smooth approximation in KANs
            nn.Linear(in_features * 2, out_features)
        )
    def forward(self, x):
        return self.layer(x)

class LSTMKAN(nn.Module):
    """
    Nithinkumar et al. 2026: Hybrid LSTM-KAN Architecture.
    LSTM for temporal sequencing, KAN for dealing with the heavy non-linearities and class imbalance.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        self.lstm = nn.LSTM(input_size=40, hidden_size=64, num_layers=2, batch_first=True)
        
        try:
            from kan import KAN
            self.kan = KAN([64, 32, num_classes])
            self.has_kan = True
        except ImportError:
            print("WARNING: pykan library not found. Falling back to an MLP imitation block.")
            self.kan = KANLinearPlaceholder(64, num_classes)
            self.has_kan = False
            
    def forward(self, x):
        m = self.mfcc(x) # (batch, n_mfcc, time)
        m = m.transpose(1, 2) # (batch, time, 40)
        
        out, (hn, cn) = self.lstm(m)
        last_hidden = hn[-1] # Grabbing last layer's final output (batch, 64)
        
        # KANs often require float64 or specific reshaping depending on the lib, 
        # but for PyTorch compatibility we pass it directly
        return self.kan(last_hidden)
