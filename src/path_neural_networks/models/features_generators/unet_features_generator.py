import torch
import torch.nn as nn

from path_neural_networks.models.features_generators import FeaturesGenerator

class UnetFeaturesGenerator(FeaturesGenerator):
    def __init__(self,
                 net: nn.Module,
                 out_channels: int,
                 skip_connection: bool = False,
                 *args, **kwargs):
        super().__init__(out_channels=out_channels, net=net, *args, **kwargs)
        self.skip_connection = skip_connection

        self.out_channels = out_channels
        if self.skip_connection:
            self.out_channels += 3

    def forward(self, 
                img: torch.Tensor # shape (1, 3, H, W)
    ) -> torch.Tensor:
        feature_maps = self.net(img) # # shape (1, out_channels, H, W)

        if self.skip_connection:
            feature_maps = torch.cat([feature_maps, img], dim=1) # shape (1, out_channels + 3, H, W)
        
        return feature_maps

    def as_dict(self):
        return {
            **super().as_dict(),
            "skip_connection": self.skip_connection
        }