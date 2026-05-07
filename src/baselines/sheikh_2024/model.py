import torch
import torch.nn as nn
from transformers import WhisperModel

class WhisperClassifier(nn.Module):
    """
    Sheikh et al. 2024 (Bird Whisperer): Whisper encoder + CNN/FCN Classifier.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        # Using tiny encoder to keep memory low
        self.encoder = WhisperModel.from_pretrained("openai/whisper-tiny").encoder
        
        # CNN on top of Whisper embeddings (batch, len, 384)
        self.cnn = nn.Conv1d(in_channels=384, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # FCN
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x is (batch, 80, 3000) Log-mel spectrogram prepped by WhisperFeatureExtractor
        # We don't trace gradients in the encoder to save memory unless full fine-tuning
        with torch.no_grad():
            out = self.encoder(x)
            hidden = out.last_hidden_state # (batch, seq_len, 384)
        
        # Conv1d expects (batch, channels, seq_len)
        hidden = hidden.transpose(1, 2)
        
        cnn_out = self.cnn(hidden)
        cnn_out = torch.relu(cnn_out)
        pooled = self.pool(cnn_out).squeeze(-1) # (batch, 128)
        
        return self.fc(pooled)
