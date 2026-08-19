from abc import ABC
import torch
import torch.nn as nn

class SamplingAggregationMethod(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, sampled: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def as_dict(self):
        return {
            'cls': self.__class__.__name__
        }