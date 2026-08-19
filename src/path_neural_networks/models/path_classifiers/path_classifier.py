from abc import ABC
import torch
import torch.nn as nn

class PathClassifier(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, 
                x: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError