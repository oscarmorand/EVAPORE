import torch
import numpy as np
from networkx import Graph

from utils.reconstruction.reconstruction_method import RadiusReconstructionMethod

class SmallestRadiusReconstructionMethod(RadiusReconstructionMethod):
    def __init__(self) -> None:
        super().__init__()

    def reconstruct_one(self, 
                        starting_radius: float, 
                        ending_radius: float, 
                        path: np.array
    ) -> np.array:
        path_length = len(path)
        radius = min(starting_radius, ending_radius)
        radius_path = np.full(path_length, radius)
        return radius_path

    def reconstruct(self, 
                    graph: Graph,
                    new_edges: torch.Tensor,
                    paths: list[np.array]
                    ) -> list[np.array]:
        
        radius_paths = []

        new_edges_list = new_edges.t().tolist()
        for new_edge, path in zip(new_edges_list, paths):
            u, v = new_edge 
            radius_u = graph.nodes[u]['radius']
            radius_v = graph.nodes[v]['radius']
            radius_path = self.reconstruct_one(radius_u, radius_v, path)
            radius_paths.append(radius_path)

        return radius_paths