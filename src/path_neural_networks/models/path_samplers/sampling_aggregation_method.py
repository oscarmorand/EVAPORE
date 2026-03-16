from abc import ABC
import torch

class SamplingAggregationMethod(ABC):
    def __init__(self):
        pass

    def __call__(self, sampled: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def as_dict(self):
        return {
            'cls': self.__class__.__name__
        }

class SamplingMaxAggregation(SamplingAggregationMethod):
    def __call__(self, sampled, return_coords: bool = False):
        max_vals, max_idx = sampled.max(dim=-1)
        if return_coords:
            return max_vals, max_idx
        return max_vals

class SamplingMinAggregation(SamplingAggregationMethod):
    def __call__(self, sampled, return_coords: bool = False):
        return sampled.min(dim=-1).values

class SamplingMeanAggregation(SamplingAggregationMethod):
    def __call__(self, sampled):
        return sampled.mean(dim=-1)
    
class SamplingSumAggregation(SamplingAggregationMethod):
    def __call__(self, sampled):
        return sampled.sum(dim=-1)