import torch
import torch.nn as nn

from path_neural_networks.models.features_generators import UnetFeaturesGenerator

class UntrainedUnetFeaturesGenerator(UnetFeaturesGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass # TODO
