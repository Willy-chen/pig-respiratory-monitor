import os, sys
import torch
import torch.nn as nn

# Ensure the current directory is in the path for local imports (BEATs, backbone, etc.)
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from BEATs import BEATsModel

class DorrBEATsOfficial(nn.Module):
    """
    Dörr et al. 2026 baseline using the self-contained official BEATs implementation.
    """
    def __init__(self, ckpt_path, num_classes=3):
        super().__init__()
        # Load the BEATs model from our self-contained official implementation
        self.beats = BEATsModel(ckpt_path)
        
        # Freeze the backbone for efficient baseline comparison
        for param in self.beats.parameters():
            param.requires_grad = False
            
        # BEATs embeddings are 768-dim for the base/iter3 model
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        # x: (batch, samples) raw audio
        # The BEATs internal preprocess() handles normalization and fbank extraction
        with torch.no_grad():
            # BEATsModel.forward returns a dict: {"global": (B, 768), "frame": ...}
            out = self.beats(x)
            emb = out["global"]
        return self.fc(emb)
