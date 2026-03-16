import torch
import torch.nn as nn
from torch.nn import functional as F

from path_neural_networks.models.losses import PathClassificationLoss

class BCEWithLogitsLoss(PathClassificationLoss):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            input,
            target
        )

    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
        }