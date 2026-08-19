import torch
import torch.nn as nn

from path_neural_networks.models.features_generators import FeaturesGenerator


class NoFeaturesGenerator(FeaturesGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(out_channels=3, net=nn.Identity(), *args, **kwargs)