from abc import ABC
import torch
from torch_geometric.data import Data

class DataTarget(ABC):
    def __init__(self):
        pass

    def compute_edge_label(self, data: Data, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError