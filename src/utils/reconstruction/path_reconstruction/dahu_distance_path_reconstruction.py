import torch
import numpy as np
from skimage.graph import route_through_array
from numba import njit
from typing import Optional
from collections.abc import Sequence
import pylena as pln
from networkx import Graph

from utils.reconstruction.reconstruction_method import PathReconstructionMethod

@njit
def _dahu_tos_mask_indices(parent: np.ndarray, n1: int, n2: int, depth: np.ndarray) -> np.ndarray:
    M = np.zeros_like(parent)

    while depth[n1] > depth[n2]:
        M[n1] = 1
        n1 = parent[n1]
    while depth[n2] > depth[n1]:
        M[n2] = 1
        n2 = parent[n2]
    while n1 != n2:
        M[n1] = 1
        M[n2] = 1
        n1 = parent[n1]
        n2 = parent[n2]
    M[n1] = 1

    return M

def dahu_tos_mask(t: pln.morpho.ComponentTree, n1: int, n2: int, depth: Optional[np.ndarray] = None) -> np.ndarray:
    if depth is None:
        depth = t.compute_depth()
    mask_nodes = _dahu_tos_mask_indices(t.parent, n1, n2, depth).astype(bool)
    return t.reconstruct(mask_nodes).astype(bool)

def dahu_shortest_path(
    t: pln.morpho.ComponentTree,
    p1: Sequence[int],
    p2: Sequence[int],
    depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    assert len(p1) == len(p2) == t.nodemap.ndim
    if depth is None:
        depth = t.compute_depth()
    p1 = tuple([p1[i] * 2 - 1 for i in range(len(p1))])
    p2 = tuple([p2[i] * 2 - 1 for i in range(len(p2))])

    n1 = t.nodemap[tuple(p1)]
    n2 = t.nodemap[tuple(p2)]
    ROI = dahu_tos_mask(t, n1, n2, depth)
    w = np.ones_like(t.nodemap, dtype=np.uint8) * 255
    w[ROI] = 1
    path, _ = route_through_array(w, p1, p2, geometric=False, fully_connected=False)

    return np.asarray(path)

class DahuDistancePathReconstructionMethod(PathReconstructionMethod):
    def __init__(self) -> None:
        super().__init__(height_related = True)

    def reconstruct_one(self,
                        map: torch.Tensor,
                        pos_start: tuple[int, int],
                        pos_goal: tuple[int, int],
                        tree: pln.morpho.ComponentTree = None
    ) -> list[np.array]:
        if tree is None:
            map_max_val = map.max().item()
            if self.height_related and map_max_val <= 1.0:
                map = map * 255.0
            img = map.numpy().astype(np.uint8)
            tree = pln.morpho.tos(img, padding="median", subsampling="full")

        path = dahu_shortest_path(tree, pos_start, pos_goal)
        path = (path + 1) // 2  # Convert back to original image coordinates
        return path

    def reconstruct(self, 
                    map: torch.Tensor,
                    graph: Graph,
                    new_edges: torch.Tensor
    ) -> torch.Tensor:
        paths = []

        map_max_val = map.max().item()
        if self.height_related and map_max_val <= 1.0:
            map = map * 255.0
        img = map.numpy().astype(np.uint8)

        tree = pln.morpho.tos(img, padding="median", subsampling="full")

        new_edges_list = new_edges.t().tolist()
        for edge in new_edges_list:
            u, v = edge
            start = graph.nodes[u]['pos']
            goal = graph.nodes[v]['pos']
            start = (int(round(start[0])), int(round(start[1])))
            goal = (int(round(goal[0])), int(round(goal[1])))
            
            path = self.reconstruct_one(map, start, goal, tree)
            paths.append(path)

        return paths