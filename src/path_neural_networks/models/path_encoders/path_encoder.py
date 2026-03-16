import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class PathEncoder(nn.Module, ABC):
    def __init__(self, ):
        super(PathEncoder, self).__init__()

    @abstractmethod
    def forward(self, 
                path: torch.Tensor
    ) -> torch.Tensor:
        dim = path.dim()
        if dim < 2:
            raise ValueError("Input tensor must have at least 2 dimensions.")
        if dim == 2:
            path = path.unsqueeze(0)  # Add batch dimension
        elif dim > 3:
            raise ValueError("Input tensor has too many dimensions.")
        return path

    def as_dict(self) -> dict:
        raise NotImplementedError