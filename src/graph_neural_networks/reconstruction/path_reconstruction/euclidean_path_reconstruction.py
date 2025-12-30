import numpy as np
import torch
from networkx import Graph

from graph_neural_networks.reconstruction.reconstruction_method import PathReconstructionMethod
from graph.graph_visualization import get_virtual_centerline

class EuclideanPathReconstructionMethod(PathReconstructionMethod):
    def __init__(self) -> None:
        super().__init__(height_related=False)

    def reconstruct_one(self,
                        map: torch.Tensor,
                        pos_start: tuple[int, int],
                        pos_goal: tuple[int, int]
    ) -> list[np.array]:
        y0, x0 = pos_start
        y1, x1 = pos_goal
        euclidian_path = get_virtual_centerline(x0, y0, x1, y1)
        euclidian_path = [(pos[1], pos[0]) for pos in euclidian_path.tolist()] # Convert (x, y) to (y, x)
        euclidian_path = np.array(euclidian_path)
        return euclidian_path