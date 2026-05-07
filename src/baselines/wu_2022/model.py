import torch
import torch.nn as nn
import torchaudio

class MFCC_LSTM(nn.Module):
    """
    Wu et al. 2022: MLMC (spectral/speech features) modeling with RNN/LSTM/GRU.
    Here we use MFCCs as the spectral/speech feature and string them into a Bi-LSTM.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={'n_mels': 128}
        )
        self.lstm = nn.LSTM(input_size=40, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        feat = self.mfcc(x) # (batch, n_mfcc, time)
        feat = feat.transpose(1, 2) # (batch, time, n_mfcc)
        
        out, (hn, cn) = self.lstm(feat)
        
        # Take the last hidden state from both directions (2 layers * 2 directions = 4 states)
        # We grab the final layer's forward and backward states
        last_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1) # (batch, 128)
        
        return self.fc(last_hidden)
