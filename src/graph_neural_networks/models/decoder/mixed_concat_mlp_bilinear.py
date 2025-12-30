import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_neural_networks.models import ConcatMLPDecoder
from graph_neural_networks.models import BilinearDecoder

class MixedConcatMLPBilinearDecoder(nn.Module):
    def __init__(self,
                 concat_mlp: ConcatMLPDecoder,
                 bilinear: BilinearDecoder,
                 concat_mlp_ratio: float = 0.5
    ) -> None:
        super().__init__()
        self.concat_mlp = concat_mlp
        self.bilinear = bilinear
        self.concat_mlp_ratio = concat_mlp_ratio

    def forward(self, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        concat_mlp_scores = self.concat_mlp(y, edge_index)
        bilinear_scores = self.bilinear(y, edge_index)
        combined_scores = (concat_mlp_scores * self.concat_mlp_ratio + bilinear_scores * (1 - self.concat_mlp_ratio))
        return combined_scores