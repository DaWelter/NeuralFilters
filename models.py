from typing import NamedTuple

import torch.nn as nn
from torch import Tensor


class Out(NamedTuple):
    '''
    Output of observation models
    '''
    y : Tensor # (L x B x 1 x C)  # Observation vector. Lives in the output space of the image processing CNN backbone.
    r : Tensor # (L x B x 1 x C)  # FIXME: Variances as vector? Maybe use diagonal matrix? Will currently only work for C=1


class Backbone(nn.Sequential):
    """
    Simple 1D conv net. 
    Input: (B x C x W), where channels C=1, and image width W=7
    Output (B x C)
    """
    num_features = 32
    def __init__(self):
        super().__init__(
            # 7 -> 5
            nn.Conv1d(1, 8, 3, 2, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            
            # 5->3
            nn.Conv1d(8, 16, 3, 2, 1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            # 3->1
            nn.Conv1d(16, 32, 3, 2, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Flatten()
        )



class SingleFrameRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        
        hidden_dim = 64
        
        self.fc = nn.Sequential(
            nn.Linear(self.backbone.num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return Out(x, None)

    @property
    def device(self):
        return next(self.parameters()).device