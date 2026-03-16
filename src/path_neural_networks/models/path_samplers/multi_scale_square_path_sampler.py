import torch
import warnings

from path_neural_networks.models.path_samplers import PathSampler
from path_neural_networks.models.path_samplers.sampling_aggregation_method import SamplingAggregationMethod, SamplingMaxAggregation


class MultiScaleSquarePathSampling(PathSampler):
    def __init__(self, 
                 in_channels: int, 
                 square_sizes: list[int] = [1, 3, 5], 
                 aggregation: SamplingAggregationMethod = SamplingMaxAggregation, 
                 **kwargs
    ):
        super().__init__(in_channels)
        if square_sizes is None:
            raise ValueError(f"Sampling square sizes list must be set, got {square_sizes}")
        self.n_scale = len(square_sizes)
        if self.n_scale < 1:
            raise ValueError(f"Sampling square sizes list must at least contain one element, got {square_sizes}")
        for s in square_sizes:
            if s <= 0:
                raise ValueError(f"Sampling square size must be a non zero positive integer (preferably an odd number), got {s}")
            if s % 2 == 0:
                warnings.warn(f"Sampling square size should preferably be an odd number, got {s}")
        self.out_channels = in_channels * self.n_scale

        self.square_sizes = square_sizes
        self.aggregation = aggregation

        self.half_sizes = [s // 2 for s in self.square_sizes]

        for i, (square_size, half_size) in enumerate(zip(self.square_sizes, self.half_sizes)):
            offsets = torch.stack([
                torch.arange(-half_size, half_size + 1).repeat_interleave(square_size),
                torch.arange(-half_size, half_size + 1).repeat(square_size)
            ], dim=1)
            self.register_buffer(f'offsets_{i}', offsets)

    def forward(self,
                feature_maps: torch.Tensor,
                path: torch.Tensor
     ) -> torch.Tensor:
        """
        Args:
            feature_maps: (batch, channels, height, width)
            path: (1, path_length, 2) or (path_length, 2)
        
        Returns:
            path_features: (batch, channels * n_scale, path_length)
        """
        path = path.type(torch.long).squeeze(dim=0)  # (path_length, 2)
        batch_size, channels, height, width = feature_maps.shape
        path_length = path.shape[0]
        
        # Expand path coordinates with offsets
        path_expanded = path.unsqueeze(1)  # (path_length, 1, 2)
        offsets: list[torch.Tensor] = [getattr(self, f'offsets_{i}') for i in range(self.n_scale)] # list[(square_size^2, 2)]
        offsets_expanded = [offset.unsqueeze(0) for offset in offsets] # list[(1, square_size^2, 2)]
        
        # Broadcast and add: list[(path_length, square_size^2, 2)]
        samples_coords = [path_expanded + offset_expanded for offset_expanded in offsets_expanded]
        
        # Flatten to list[(path_length * square_size^2, 2)]
        samples_coords = [sample_coords.reshape(-1, 2) for sample_coords in samples_coords]
        
        # Clamp coordinates to valid range
        for sample_coords in samples_coords:
            sample_coords[:, 0] = torch.clamp(sample_coords[:, 0], 0, height - 1)
            sample_coords[:, 1] = torch.clamp(sample_coords[:, 1], 0, width - 1)
        
        # Sample features: list[(batch, channels, path_length * square_size^2)]
        sampleds = [feature_maps[:, :, sample_coords[:, 0], sample_coords[:, 1]] for sample_coords in samples_coords]
        
        # Reshape to list[(batch, channels, path_length, square_size^2)]
        sampleds = [sampled.reshape(batch_size, channels, path_length, -1) for sampled in sampleds]
        
        # Aggregate over the squares
        path_features = [self.aggregation(sampled) for sampled in sampleds]  # list[(batch, channels, path_length)]

        # Now that all scales samples are reduced to same shape, we can concat them channel-wise
        final_path_features = torch.cat(path_features, dim=1)
        
        return final_path_features
    
    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "square_sizes": self.square_sizes,
            "n_scale": self.n_scale,
            "aggregation": self.aggregation.as_dict() 
                if hasattr(self.aggregation, "as_dict")
                else self.aggregation.__class__.__name__
        }