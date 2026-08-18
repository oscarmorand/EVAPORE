import numpy as np
import torch
from networkx import Graph
from typing import List, Tuple

from utils.reconstruction.reconstruction_method import PathReconstructionMethod
from graph.graph_visualization import get_virtual_centerline

class EuclideanPathReconstructionMethod(PathReconstructionMethod):
    def __init__(self) -> None:
        super().__init__(height_related=False)

    def reconstruct_one(self,
                         map: torch.Tensor,
                         pos_start: Tuple[int, ...],
                         pos_goal: Tuple[int, ...]
                         ) -> np.ndarray:
        p0 = np.array(pos_start)
        p1 = np.array(pos_goal)
        euclidean_path = get_virtual_centerline(p0, p1)  # shape: (len, D)
        return euclidean_path