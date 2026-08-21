from abc import ABC
import torch
import torch.nn as nn
import warnings

class PathSampler(ABC, nn.Module):
    def __init__(self, 
                 in_channels: int,
                 ndim: int = 2, 
                 **kwargs):
        
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be a positive integer, got {in_channels}")
        if not isinstance(in_channels, int):
            raise TypeError(f"in_channels must be an integer, got {type(in_channels)}")

        if ndim <= 0:
            raise ValueError(f"ndim must be a positive integer, got {ndim}")
        if not isinstance(ndim, int):
            raise TypeError(f"ndim must be an integer, got {type(ndim)}")
        if ndim not in [2, 3]:
            warnings.warn(f"PathSampler is designed for 2D or 3D data, got ndim={ndim}")
        
        self.in_channels = in_channels
        self.out_channels = None
        self.ndim = ndim

    def _create_ndim_square_offsets(self, 
                                    half_size: int, 
                                    ndim: int
                                    ) -> torch.Tensor:
        ranges = [
            torch.arange(-half_size, half_size + 1)
                for _ in range(ndim)
        ]
        offsets = torch.stack(
            torch.meshgrid(*ranges, indexing='ij'), 
            dim=-1
        ).reshape(-1, ndim)
        return offsets

    def forward(self,
                feature_maps: torch.Tensor,
                path: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def as_dict(self):
        raise NotImplementedError