import torch

from path_neural_networks.models.path_samplers import SamplingAggregationMethod

class SamplingMaxAggregation(SamplingAggregationMethod):
    def __call__(self, 
                 sampled: torch.Tensor, 
                 return_coords: bool = False
    ) -> torch.Tensor:
        max_vals, max_idx = sampled.max(dim=-1)
        if return_coords:
            return max_vals, max_idx
        return max_vals

class SamplingMinAggregation(SamplingAggregationMethod):
    def __call__(self, 
                 sampled: torch.Tensor, 
                 return_coords: bool = False
    ) -> torch.Tensor:
        min_vals, min_idx = sampled.min(dim=-1)
        if return_coords:
            return min_vals, min_idx
        return min_vals