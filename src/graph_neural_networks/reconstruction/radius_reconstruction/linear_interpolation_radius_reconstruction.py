import torch
import numpy as np
from networkx import Graph

from graph_neural_networks.reconstruction.reconstruction_method import RadiusReconstructionMethod

class LinearInterpolationRadiusReconstructionMethod(RadiusReconstructionMethod):
    def __init__(self) -> None:
        super().__init__()

    def reconstruct(self, 
                    map: torch.Tensor,
                    graph: Graph,
                    new_edges: torch.Tensor,
                    paths: list[np.array]
                    ) -> list[np.array]:
        
        radius_paths = []

        new_edges_list = new_edges.t().tolist()
        for new_edge, path in zip(new_edges_list, paths):
            radius_path = []
            path_length = len(path)

            u, v = new_edge 
            radius_u = graph.nodes[u]['radius']
            radius_v = graph.nodes[v]['radius']
            for i in range(len(path)):
                t = i / (path_length - 1) if path_length > 1 else 0.0
                radius = (1 - t) * radius_u + t * radius_v
                radius_path.append(radius)

            radius_paths.append(radius_path)

        return radius_paths