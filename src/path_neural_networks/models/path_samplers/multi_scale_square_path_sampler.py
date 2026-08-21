import torch
import warnings

from path_neural_networks.models.path_samplers import PathSampler
from path_neural_networks.models.path_samplers import SamplingAggregationMethod
from path_neural_networks.models.path_samplers import SamplingMaxAggregation


class MultiScaleSquarePathSampler(PathSampler):
    def __init__(self, 
                 in_channels: int, 
                 square_sizes: list[int] = [1, 3, 5], 
                 ndim: int = 2,
                 aggregation: SamplingAggregationMethod = SamplingMaxAggregation, 
                 **kwargs
    ):
        super().__init__(in_channels, ndim=ndim, **kwargs)

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

        # Pre-compute relative offsets for each square
        # shape: (square_size^ndim, ndim) x n_scale
        for i, half_size in enumerate(self.half_sizes):
            offsets = self._create_ndim_square_offsets(half_size=half_size, ndim=ndim)
            self.register_buffer(f'offsets_{i}', offsets)

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
            path_features: (1, channels * n_scale, path_length)
        """
        assert path.shape[-1] == self.ndim
        assert feature_maps.ndim == self.ndim + 2

        path = path.long().squeeze(dim=0) # (path_length, ndim)
        channels = feature_maps.shape[1]
        path_length = path.shape[0]
        
        # Expand path coordinates with offsets
        path_expanded = path.unsqueeze(1)  # (path_length, 1, ndim)
        offsets = [getattr(self, f'offsets_{i}') for i in range(self.n_scale)] # list[(square_size^ndim, ndim)]
        offsets_expanded = [offset.unsqueeze(0) for offset in offsets] # list[(1, square_size^ndim, ndim)]
        
        # Broadcast and add:
        samples_coords = [path_expanded + offset_expanded for offset_expanded in offsets_expanded] # list[(path_length, square_size^ndim, ndim)]

        # Clamp coordinates to valid range
        spatial_shape = torch.tensor(feature_maps.shape[2:], device=feature_maps.device) # (ndim)
        for scale_i in range(self.n_scale):
            samples_coords[scale_i] = torch.minimum(samples_coords[scale_i], spatial_shape - 1)
            samples_coords[scale_i].clamp_(min=0)
        
        # Flatten to list[(path_length * square_size^ndim, ndim)]
        samples_coords_flatten = [sample_coords.reshape(-1, self.ndim) for sample_coords in samples_coords]
        samples_coords_tuple = [list(sample_coords_flatten.unbind(dim=1)) for sample_coords_flatten in samples_coords_flatten]  # list[tuple[(path_length * square_size^ndim,)]]
    
        # Sample features: list[(1, channels, path_length * square_size^ndim)]
        sampleds = [feature_maps[:, :, *sample_coords_tuple] for sample_coords_tuple in samples_coords_tuple]
        
        # Reshape to list[(1, channels, path_length, square_size^ndim)]
        sampleds = [sampled.reshape(1, channels, path_length, -1) for sampled in sampleds]

        # Aggregate over the squares / cubes

        if return_coords:
            path_features_and_idx = [self.aggregation(sampled, return_coords=True) for sampled in sampleds]  # list[(1, channels, path_length), (1, channels, path_length)]
            path_features, selected_idx = zip(*path_features_and_idx)  # list[(1, channels, path_length)], list[(1, channels, path_length)]

            final_path_features = torch.cat(path_features, dim=1) # (1, channels * n_scale, path_length)

            coords = [sample_coords.reshape(path_length, -1, self.ndim) for sample_coords in samples_coords]
            coords = [coord[torch.arange(path_length, device=feature_maps.device).view(1, 1, -1), idx] for coord, idx in zip(coords, selected_idx)]  # list[(1, channels, path_length, ndim)]
            final_coords = torch.cat(coords, dim=1)  # (1, channels * n_scale, path_length, ndim)

            return final_path_features, final_coords

        path_features = [self.aggregation(sampled) for sampled in sampleds]  # list[(1, channels, path_length)]

        # Now that all scales samples are reduced to same shape, we can concat them channel-wise
        final_path_features = torch.cat(path_features, dim=1) # (1, channels * n_scale, path_length)

        return final_path_features
    
    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "ndim": self.ndim,
            "square_sizes": self.square_sizes,
            "n_scale": self.n_scale,
            "aggregation": self.aggregation.as_dict() 
                if hasattr(self.aggregation, "as_dict")
                else self.aggregation.__class__.__name__
        }