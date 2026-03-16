import torch
import torch.nn as nn

from image_segmentation.models.binary_segmentator import BinarySegmentator
from path_neural_networks.models.features_generators import UnetFeaturesGenerator

class PretrainedUnetFeaturesGenerator(UnetFeaturesGenerator):
    def __init__(self,
                 ckpt_path: str,
                 device: str,
                 out_channels: int = None,
                 freeze_pretrained: bool = False,
                 skip_connection: bool = False,
                 *args, **kwargs):
    
        self.ckpt_path = ckpt_path
        self.device = device
        self.freeze_pretrained = freeze_pretrained
        self.skip_connection = skip_connection

        net = BinarySegmentator.load_from_checkpoint(ckpt_path, map_location=device).net

        if self.freeze_pretrained:
            for param in net.parameters():
                param.requires_grad = False

        last_conv = net.layers[-1]
        assert isinstance(last_conv, nn.Conv2d), "Last layer is not Conv2d"
        last_conv_in_channels = last_conv.in_channels

        if out_channels is None:
            net.layers[-1] = nn.Identity()
            out_channels = last_conv_in_channels
        else:
            net.layers[-1] = nn.Conv2d(in_channels=last_conv_in_channels, out_channels=out_channels, kernel_size=1, stride=1)

        super().__init__(net, out_channels, skip_connection=skip_connection, *args, **kwargs)

    def as_dict(self):
        return {
            **super().as_dict(),
            "ckpt_path": self.ckpt_path,
            "device": self.device,
            "freeze_pretrained": self.freeze_pretrained,
            "skip_connection": self.skip_connection
        }