import torch
import torch.nn as nn
from torch.nn import functional as F

from path_neural_networks.models.losses import PathClassificationLoss

class WeightedBCEWithLogitsLoss(PathClassificationLoss):
    def __init__(self, classes_ratio: list[float], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes_ratio = classes_ratio
        neg_ratio, pos_ratio = classes_ratio[0], classes_ratio[1]
        self.pos_weight_f = neg_ratio / pos_ratio
        pos_weight =  torch.tensor(self.pos_weight_f, dtype=torch.float32)
        
        self.register_buffer("pos_weight", pos_weight)
        self.pos_weight: torch.Tensor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            pos_weight=self.pos_weight,
        )

    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "classes_ratio": self.classes_ratio,
            "pos_weight": self.pos_weight_f,
        }