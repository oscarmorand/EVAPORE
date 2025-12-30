import torch
import numpy as np
from torch_geometric.data import Data

from graph_neural_networks.data.edge_splits.edge_split import EdgeSplit
from graph_neural_networks.data.data_targets.data_target import DataTarget

class CustomEdgeSplit(EdgeSplit):
    def __init__(self, geodesic_euclidean_ratio_sampling_ratio: float = 1.0,
                 negative_neighbor_sampling: bool = True,
                 data_target: DataTarget = None) -> None:
        super().__init__()
        self.geodesic_euclidean_ratio_sampling_ratio = geodesic_euclidean_ratio_sampling_ratio
        self.negative_neighbor_sampling = negative_neighbor_sampling
        self.data_target = data_target

    def compute_not_in_pred_edges(self, data) -> torch.Tensor:
        not_in_pred_edge_index = data.not_in_pred_edge_index[:, ::2] # take only one direction of undirected edges, avoid sampling two times the same edge
        return not_in_pred_edge_index

    def compute_neighbors_samples_edges(self, data) -> torch.Tensor:
        distance_matrix = data.geodesic_distance_matrix

        if not self.negative_neighbor_sampling:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=distance_matrix.dtype)
        
        in_pred_node_neighbors = {}
        for u, v in data.in_pred_edge_index.t().tolist():
            if u not in in_pred_node_neighbors:
                in_pred_node_neighbors[u] = set()
            if v not in in_pred_node_neighbors:
                in_pred_node_neighbors[v] = set()
            in_pred_node_neighbors[u].add(v)
            in_pred_node_neighbors[v].add(u)

        not_in_pred_edge_index = data.not_in_pred_edge_index
        neighbors_samples_index = []
        for u, v in zip(not_in_pred_edge_index[0], not_in_pred_edge_index[1]):
            v_neighbors = torch.tensor(list(in_pred_node_neighbors.get(v.item(), set())))
            if len(v_neighbors) > 0:
                picked_v_neighbors = v_neighbors[torch.randint(0, len(v_neighbors), (1,))]
                for picked_v in picked_v_neighbors:
                    neighbors_samples_index.append((u, picked_v.item()))

        neighbors_samples_index = torch.tensor(neighbors_samples_index).t()
        return neighbors_samples_index

    def compute_geodesic_euclidean_ratio_edges(self, data) -> torch.Tensor:
        yx_pos = data.pos
        distance_matrix = data.geodesic_distance_matrix
        euclidean_distance_matrix = torch.cdist(yx_pos, yx_pos, p=2)

        n_positive_edges = data.not_in_pred_edge_index.shape[1] // 2
        n_geodesic_euclidean_ratio_samples = int(n_positive_edges * self.geodesic_euclidean_ratio_sampling_ratio)

        distance_ratio_matrix = distance_matrix.detach().clone()
        pos_euclid = euclidean_distance_matrix > 0
        distance_ratio_matrix[pos_euclid] = distance_ratio_matrix[pos_euclid] / euclidean_distance_matrix[pos_euclid]

        distance_ratio_matrix_masked = distance_ratio_matrix * torch.tril(torch.ones_like(distance_ratio_matrix))
        sorted_y, sorted_x = torch.unravel_index(torch.argsort(distance_ratio_matrix_masked.flatten(), descending=True), distance_ratio_matrix_masked.shape)
        top_sorted_y, top_sorted_x = sorted_y[:n_geodesic_euclidean_ratio_samples], sorted_x[:n_geodesic_euclidean_ratio_samples]

        geodesic_euclidean_ratio_index = torch.stack([top_sorted_y, top_sorted_x], dim=0)
        return geodesic_euclidean_ratio_index

    def __call__(self, data):
        self.check_basic_data_attrs(data)
        data_has_all_attr, missing_attrs = self.all_attrs_in_data(data, ['geodesic_distance_matrix', 'pos', 'in_pred_edge_index', 'not_in_pred_edge_index'])
        if not data_has_all_attr:
            raise ValueError("Data object is missing required attributes for custom data split: "
                                f"{', '.join(missing_attrs)}")
        
        if isinstance(data.geodesic_distance_matrix, np.ndarray):
            data.geodesic_distance_matrix = torch.tensor(data.geodesic_distance_matrix)

        not_in_pred_edge_index = self.compute_not_in_pred_edges(data)
        neighbors_samples_index = self.compute_neighbors_samples_edges(data)
        geodesic_euclidean_ratio_index = self.compute_geodesic_euclidean_ratio_edges(data)
        edge_label_index = torch.cat([not_in_pred_edge_index, neighbors_samples_index, geodesic_euclidean_ratio_index], axis=1)

        edge_label = self.data_target.compute_edge_label(data, edge_label_index)

        graph = Data(
            x = data.x,
            edge_index = data.edge_index,
            edge_label = edge_label,
            edge_label_index = edge_label_index,
            pos = data.pos,
            graph_id = data.graph_id,
        )

        return graph