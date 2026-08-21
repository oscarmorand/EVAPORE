import torch
import torch.nn as nn

from path_neural_networks.models.path_samplers import SamplingAggregationMethod

class SamplingSoftAdaptiveAggregation(SamplingAggregationMethod):
    def __init__(self, 
                 n_channels: int, 
                 init_weights: torch.Tensor = None):
        super().__init__()
        self.n_channels = n_channels
        self.init_weights = init_weights

        if init_weights is not None:
            assert init_weights.shape == (n_channels,), f"init_weights must have shape ({n_channels},), but got {init_weights.shape}"

            init_alphas = init_weights
            init_taus = torch.log(torch.exp(torch.abs(torch.tanh(init_alphas))) - 1) 

        if init_alphas is not None:
            alphas = init_alphas
        else:
            alphas = torch.randn(self.n_channels) # Random init values with mean 0 and variance 1, so that after tanh we get values in range [-1, 1] with mean around 0
        self.register_parameter("alphas", nn.Parameter(alphas, requires_grad=True))

        if init_taus is not None:
            taus = init_taus
        else:
            taus = torch.rand(self.n_channels) - 0.5 # Random init values in range [-0.5, 0.5]
        self.register_parameter("taus", nn.Parameter(taus, requires_grad=True))

    def __call__(self, 
                 sampled: torch.Tensor # shape (batch_size, n_channels, path_length, square_size^2)
    ) -> torch.Tensor:
        # Ensure softness is positive
        softness = nn.functional.softplus(self.taus) + 1e-6  # shape (n_channels,)

        # Constrain alpha to [-1, 1] (-1 = min pooling, 0 = mean pooling, 1 = max pooling)
        tan_alpha = torch.tanh(self.alphas) # shape (n_channels,)

        s = (tan_alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * sampled) / (softness.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) # Scaled sampled values by alpha and temperature
        weights = torch.softmax(s, dim=-1)  # Compute the weights of each input value using softmax
        values = weights * sampled # Apply the weights to the sampled values
        out = values.sum(dim=-1) # Aggregate the weighted values by summing over the last dimension (the square_size^2 dimension)

        return out

    def as_dict(self):
        return {
            'cls': self.__class__.__name__,
            'n_channels': self.n_channels,
            'init_weights': self.init_weights.data.cpu().numpy().tolist() if self.init_weights is not None else None,
            'alphas': self.alphas.data.cpu().numpy().tolist(),
            'taus': self.taus.data.cpu().numpy().tolist()
        }
    

class SamplingSimpleSoftAdaptiveAggregation(SamplingAggregationMethod):
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
        self.register_parameter("weights", nn.Parameter(weights, requires_grad=True))

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