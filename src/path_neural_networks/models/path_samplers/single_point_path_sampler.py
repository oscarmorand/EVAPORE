import torch

from path_neural_networks.models.path_samplers import PathSampler


class SinglePointPathSampler(PathSampler):
    def __init__(self, 
                 in_channels: int,
                 n_dim: int = 2,
                 **kwargs):
        
        super().__init__(in_channels, ndim=n_dim, **kwargs)
        
        self.out_channels = in_channels

    def forward(
        self,
        feature_maps: torch.Tensor,
        path: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feature_maps:
                2D: [1, C, H, W]
                3D: [1, C, D, H, W]

            path:
                2D: [1, L, 2]
                3D: [1, L, 3]

        Returns:
            [1, C, L]
        """

        path = path.long().squeeze(0)  # [L, ndim] ([L, 2] or [L, 3])

        # Split coordinates
        coords = path.long().unbind(dim=1) # [L, 2] -> [L], [L] or [L, 3] -> [L], [L], [L]

        # Dynamic indexing
        path_features = feature_maps[:, :, *coords]

        return path_features


    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels
        }