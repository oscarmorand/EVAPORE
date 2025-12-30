import torch
from torch_geometric.data import Data

from graph_neural_networks.data.data_targets.data_target import DataTarget

class GeodesicDistanceTarget(DataTarget):
    def __init__(self):
        super().__init__()

    def compute_edge_label(self, data: Data, edge_index: torch.Tensor) -> torch.Tensor:
        if not hasattr(data, 'geodesic_distance_matrix'):
            raise ValueError("Data object does not have 'geodesic_distance_matrix' attribute.")

        distance_matrix = data.geodesic_distance_matrix
        return distance_matrix[edge_index[0], edge_index[1]]