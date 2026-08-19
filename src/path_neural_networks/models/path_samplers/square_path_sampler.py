import torch
import warnings

from path_neural_networks.models.path_samplers import PathSampler
from path_neural_networks.models.path_samplers import SamplingAggregationMethod
from path_neural_networks.models.path_samplers import SamplingMaxAggregation

class SquarePathSampler(PathSampler):
    def __init__(self, 
                 in_channels: int, 
                 square_size: int = 3, 
                 aggregation: SamplingAggregationMethod = SamplingMaxAggregation, 
                 **kwargs
    ):
        super().__init__(in_channels)
        self.out_channels = in_channels

        if square_size <= 0:
            raise ValueError(f"Sampling square size must be a non zero positive integer (preferably an odd number), got {square_size}")
        if square_size % 2 == 0:
            warnings.warn(f"Sampling square size should preferably be an odd number, got {square_size}")
        self.square_size = square_size
        self.square_size2 = self.square_size * self.square_size
        self.aggregation = aggregation

        self.half_size = square_size // 2
        
        # Pre-compute relative offsets for the square
        # shape: (square_size * square_size, 2)
        offsets = torch.stack([
            torch.arange(-self.half_size, self.half_size + 1).repeat_interleave(square_size),
            torch.arange(-self.half_size, self.half_size + 1).repeat(square_size)
        ], dim=1)
        self.register_buffer('offsets', offsets)
    
    def forward(self,
                feature_maps: torch.Tensor,
                path: torch.Tensor,
                return_coords: bool = False
        ) -> torch.Tensor:
        """
        Args:
            feature_maps: (batch, channels, height, width)
            path: (1, path_length, 2) or (path_length, 2)
        
        Returns:
            path_features: (batch, channels, path_length)
        """
        path = path.type(torch.long).squeeze(dim=0)  # shape (path_length, 2)
        batch_size, channels, height, width = feature_maps.shape
        path_length = path.shape[0]
        
        # Expand path coordinates with offsets
        # path: (path_length, 2) -> (path_length, 1, 2)
        # offsets: (square_size^2, 2) -> (1, square_size^2, 2)
        path_expanded = path.unsqueeze(1)  # (path_length, 1, 2)
        offsets_expanded = self.offsets.unsqueeze(0)  # (1, square_size^2, 2)
        
        # Broadcast and add: (path_length, square_size^2, 2)
        sample_coords = path_expanded + offsets_expanded
        
        # Flatten to (path_length * square_size^2, 2)
        sample_coords = sample_coords.reshape(-1, 2)
        
        # Clamp coordinates to valid range
        sample_coords[:, 0] = torch.clamp(sample_coords[:, 0], 0, height - 1)
        sample_coords[:, 1] = torch.clamp(sample_coords[:, 1], 0, width - 1)
        
        # Sample features: (batch, channels, path_length * square_size^2)
        sampled = feature_maps[:, :, sample_coords[:, 0], sample_coords[:, 1]]
        
        # Reshape to (batch, channels, path_length, square_size^2)
        sampled = sampled.reshape(batch_size, channels, path_length, self.square_size2)
        
        # Aggregate over the square
        if return_coords:
            path_features, selected_idx = self.aggregation(sampled) # (batch, channels, path_length), (batch, channels, path_length)
            coords = sample_coords.view(path_length, self.square_size2, 2)
            coords = coords[torch.arange(path_length, device=feature_maps.device).view(1,1,-1), selected_idx] # (batch, channels, path_length, 2)
            return path_features, coords
        
        path_features = self.aggregation(sampled) # (batch, channels, path_length)
        
        return path_features
    
    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "square_size": self.square_size,
            "aggregation": self.aggregation.as_dict() 
                if hasattr(self.aggregation, "as_dict")
                else self.aggregation.__class__.__name__
        }