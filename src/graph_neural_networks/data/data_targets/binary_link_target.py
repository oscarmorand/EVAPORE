import torch
from torch_geometric.data import Data

from graph_neural_networks.data.data_targets.data_target import DataTarget

class BinaryLinkTarget(DataTarget):
    def __init__(self):
        super().__init__()

    def compute_edge_label(self, data: Data, edge_index: torch.Tensor) -> torch.Tensor:
        if hasattr(data, 'edge_label'):
            return data.edge_label
        
        if hasattr(data, 'geodesic_distance_matrix'):
            distance_matrix = data.geodesic_distance_matrix
            binary_distance_matrix = (distance_matrix <= 0.0).float()
            return binary_distance_matrix[edge_index[0], edge_index[1]]
        