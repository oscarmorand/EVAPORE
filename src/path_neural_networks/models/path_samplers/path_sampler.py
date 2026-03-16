from abc import ABC
import torch
import torch.nn as nn

class PathSampler(ABC, nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = None

    def forward(self,
                feature_maps: torch.Tensor,
                path: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def as_dict(self):
        raise NotImplementedError