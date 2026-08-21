import torch
import warnings

from path_neural_networks.models.path_samplers import PathSampler
from path_neural_networks.models.path_samplers import SamplingAggregationMethod
from path_neural_networks.models.path_samplers import SamplingMaxAggregation

class SquarePathSampler(PathSampler):
    def __init__(self, 
                 in_channels: int, 
                 square_size: int = 3, 
                 ndim: int = 2,
                 aggregation: SamplingAggregationMethod = SamplingMaxAggregation, 
                 **kwargs
    ):
        super().__init__(in_channels, ndim=ndim, **kwargs)

        if square_size <= 0:
            raise ValueError(f"Sampling square size must be a non zero positive integer (preferably an odd number), got {square_size}")
        if square_size % 2 == 0:
            warnings.warn(f"Sampling square size should preferably be an odd number, got {square_size}")

        self.out_channels = in_channels
        self.square_size = square_size
        self.aggregation = aggregation

        self.half_size = square_size // 2
        
        # Pre-compute relative offsets for the square
        # shape: (square_size^ndim, ndim)
        offsets = self._create_ndim_square_offsets(half_size=self.half_size, ndim=ndim)
        self.register_buffer('offsets', offsets)
    
    def forward(self,
                feature_maps: torch.Tensor,
                path: torch.Tensor,
                return_coords: bool = False
        ) -> torch.Tensor:
        """
        Args:
            feature_maps: 
                2D: (1, channels, height, width)
                3D: (1, channels, depth, height, width)
            path: 
                2D: (1, path_length, 2)
                3D: (1, path_length, 3)
            return_coords: bool, whether to return the coordinates of the selected samples
        
        Returns:
            (1, channels, path_length)
        """
        if path.shape[-1] != self.ndim:
            raise ValueError(f"Path coordinates must have {self.ndim} dimensions, got {path.shape[-1]}")
        if feature_maps.ndim != self.ndim + 2:
            raise ValueError(f"Feature maps must have {self.ndim + 2} dimensions, got {feature_maps.ndim}")

        path = path.long().squeeze(dim=0) # (path_length, ndim)
        channels = feature_maps.shape[1]
        path_length = path.shape[0]
        
        # Expand path coordinates with offsets
        path_expanded = path.unsqueeze(1) # (path_length, 1, ndim)
        offsets_expanded = self.offsets.unsqueeze(0)  # (1, square_size^ndim, ndim)
        
        # Broadcast and add:
        sample_coords = path_expanded + offsets_expanded # (path_length, square_size^ndim, ndim)

        # Clamp coordinates to valid range
        spatial_shape = torch.tensor(feature_maps.shape[2:], device=feature_maps.device) # (ndim)
        sample_coords = torch.minimum(sample_coords, spatial_shape - 1)
        sample_coords.clamp_(min=0)

        # Sample the whole square / cube of features for the path
        sampled = feature_maps[:, :, *sample_coords.reshape(-1, self.ndim).unbind(dim=1)] # (1, channels, path_length * square_size^ndim)
        
        # Reshape to get back the path dimension separated from the square / cube dimension
        sampled = sampled.reshape(1, channels, path_length, -1) # (1, channels, path_length, square_size^ndim)
        
        # Aggregate over the square / cube
        if return_coords:
            path_features, selected_idx = self.aggregation(sampled, return_coords=True) # (1, channels, path_length), (1, channels, path_length)
            coords = sample_coords.reshape(path_length, -1 , self.ndim)
            coords = coords[torch.arange(path_length, device=feature_maps.device).view(1,1,-1), selected_idx] # (1, channels, path_length, ndim)
            return path_features, coords
        
        path_features = self.aggregation(sampled) # (1, channels, path_length)
        
        return path_features
    
    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "ndim": self.ndim,
            "square_size": self.square_size,
            "aggregation": self.aggregation.as_dict() 
                if hasattr(self.aggregation, "as_dict")
                else self.aggregation.__class__.__name__
        }