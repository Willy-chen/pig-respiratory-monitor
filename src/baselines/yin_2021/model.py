import torch
import torch.nn as nn
import torchaudio

class SpectrogramAlexNet(nn.Module):
    """
    Implements the CNN approach modeled after Yin et al. (2021).
    Takes raw audio, converts to MelSpectrogram, reshapes to (3, 227, 227)
    and passes it through a pre-trained AlexNet model.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        from torchvision.models import alexnet, AlexNet_Weights
        weights = AlexNet_Weights.DEFAULT if pretrained else None
        self.alexnet = alexnet(weights=weights)
        
        # Modify final classifier for num_classes
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)
        
        # Audio transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=227,
            n_fft=1024,
            hop_length=512
        )
        self.resize = nn.AdaptiveAvgPool2d((227, 227))
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length) -> e.g. 10s audio at 16kHz
        # Apply transform on GPU if x is on GPU
        spec = self.mel_spectrogram(x) # (batch, n_mels, time)
        
        # Log-mel scaling
        spec = 10.0 * torch.log10(torch.clamp(spec, min=1e-10))
        
        # Add channel dim
        spec = spec.unsqueeze(1) # (batch, 1, 227, time)
        
        # Repeat to 3 channels for ImageNet compatibility
        spec = spec.repeat(1, 3, 1, 1) # (batch, 3, 227, time)
        
        # Resize time dimension to 227 so it's exactly 227x227
        img = self.resize(spec) # (batch, 3, 227, 227)
        
        # Instance normalization to approximate ImageNet mean/std
        # Avoids manual channel-wise normalization of spectrogram pixels
        mean = img.mean(dim=(2, 3), keepdim=True)
        std = img.std(dim=(2, 3), keepdim=True) + 1e-6
        img = (img - mean) / std
        
        out = self.alexnet(img)
        return out
