import torch
import torch.nn as nn

class MetaParameterNet(nn.Module):
    """
    Lightweight CNN to estimate restoration parameters from the input image.
    Outputs:
    1. Alpha (Fractional order)
    2. Relaxation (Update rate)
    3. Edge Threshold (Sensitivity)
    4. Mode (Quality vs Speed probability)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4), nn.Sigmoid() # All params normalized to [0, 1]
        )

    def forward(self, x):
        feat = self.features(x)
        return self.regressor(feat)
