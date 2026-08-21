import torch
from torch import nn

from path_neural_networks.models.path_samplers import SamplingAggregationMethod

class SamplingSoftAdaptedAggregation(SamplingAggregationMethod):
    def __init__(self, 
                 n_channels: int, 
                 init_weights: torch.Tensor = None):
        super().__init__()
        self.n_channels = n_channels

        if init_weights is not None:
            assert init_weights.shape == (n_channels,), f"init_weights must have shape ({n_channels},), but got {init_weights.shape}"
            weights = init_weights
        else:
            weights = torch.randn(self.n_channels) # Random init values with mean 0 and variance 1
        self.register_parameter("weights", nn.Parameter(weights, requires_grad=False)) # Not learnable, just for scaling the inputs before softmax

    def __call__(self, 
                 sampled: torch.Tensor # shape (batch_size, n_channels, path_length, square_size^2)
    ) -> torch.Tensor:
        s = (self.weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * sampled) # Scaled sampled values by weights
        weights = torch.softmax(s, dim=-1)  # Compute the weights of each input value using softmax
        values = weights * sampled # Apply the weights to the sampled values
        out = values.sum(dim=-1) # Aggregate the weighted values by summing over the last dimension (the square_size^2 dimension)

        return out

    def as_dict(self):
        return {
            'cls': self.__class__.__name__,
            'n_channels': self.n_channels,
            'weights': self.weights.data.cpu().numpy().tolist()
        }


class SamplingSoftMaxAggregation(SamplingAggregationMethod):
    def __call__(self, 
                 sampled: torch.Tensor, # shape (batch_size, n_channels, path_length, square_size^2)
                 return_coords: bool = False
    ) -> torch.Tensor:
        weights = torch.softmax(sampled, dim=-1) # shape (batch_size, n_channels, path_length, square_size^2)
        values = weights * sampled # shape (batch_size, n_channels, path_length, square_size^2)
        out = values.sum(dim=-1) # shape (batch_size, n_channels, path_length)
        if return_coords:
            return out, weights
        return out

class SamplingMeanAggregation(SamplingAggregationMethod):
    def __call__(self, 
                 sampled: torch.Tensor, # shape (batch_size, n_channels, path_length, square_size^2)
                 return_coords: bool = False
    ) -> torch.Tensor:
        mean_vals = sampled.mean(dim=-1)
        if return_coords:
            return mean_vals, None
        return mean_vals

class SamplingSumAggregation(SamplingAggregationMethod):
    def __call__(self, 
                 sampled: torch.Tensor, 
                 return_coords: bool = False
    ) -> torch.Tensor:
        sum_vals = sampled.sum(dim=-1)
        if return_coords:
            return sum_vals, None
        return sum_vals