import torch
import numpy as np
from networkx import Graph

from graph_neural_networks.reconstruction.reconstruction_method import RadiusReconstructionMethod

class OnePixelRadiusReconstructionMethod(RadiusReconstructionMethod):
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
            path_length = len(path)
            radius_path = [1.0] * path_length
            radius_paths.append(radius_path)

        return radius_paths