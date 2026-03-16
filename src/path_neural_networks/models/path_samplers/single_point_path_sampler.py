import torch

from path_neural_networks.models.path_samplers import PathSampler

class SinglePointPathSampler(PathSampler):
    def __init__(self, in_channels: int, **kwargs):
        super().__init__(in_channels)
        self.out_channels = in_channels

    def forward(self,
                feature_maps: torch.Tensor,
                path: torch.Tensor
    ) -> torch.Tensor:
        path = path.type(torch.long).squeeze(dim=0)  # shape (path_length, 2)
        path_features = feature_maps[:, :, path[:, 0], path[:, 1]]  # shape (1, channels, path_length)
        return path_features
    
    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels
        }