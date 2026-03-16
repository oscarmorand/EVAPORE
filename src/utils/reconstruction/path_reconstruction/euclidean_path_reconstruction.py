import numpy as np
import torch
from networkx import Graph

from utils.reconstruction.reconstruction_method import PathReconstructionMethod
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
        euclidean_path = get_virtual_centerline(x0, y0, x1, y1) # shape: (len, 2)
        euclidean_path = np.flip(euclidean_path, axis=1)  # Convert (x, y) to (y, x)
        return euclidean_path