import torch
import torch.nn as nn

class FeaturesGenerator(nn.Module):
    def __init__(self, out_channels: int = None, net: nn.Module = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
        self.net = net

    def forward(self, 
                img: torch.Tensor # shape (1, 3, H, W)
    ) -> torch.Tensor:
        return self.net(img) # shape (1, out_channels, H, W)

    def as_dict(self) -> dict:
        return {
            "cls": self.__class__.__name__,
            "out_channels": self.out_channels
        }