from abc import ABC
import torch
import numpy as np
from networkx import Graph
from typing import Tuple, List

class PathReconstructionMethod(ABC):
    def __init__(self, height_related: bool) -> None:
        super().__init__()
        self.height_related = height_related

    def reconstruct_one(self,
                         map: torch.Tensor,
                         pos_start: Tuple[int, ...],
                         pos_goal: Tuple[int, ...]
                         ) -> np.ndarray:
        raise NotImplementedError

    def reconstruct(self,
                     map: torch.Tensor,
                     graph: Graph,
                     new_edges: torch.Tensor
                     ) -> List[np.ndarray]:
        paths = []

        if self.height_related:
            map_max_val = map.max().item()
            if map_max_val <= 1.0:
                map = map * 255.0

        new_edges_list = new_edges.t().tolist()
        for edge in new_edges_list:
            u, v = edge
            start = graph.nodes[u]['pos']
            goal = graph.nodes[v]['pos']
            start = tuple(int(round(c)) for c in start)
            goal = tuple(int(round(c)) for c in goal)

            path = self.reconstruct_one(map, start, goal)
            paths.append(path)

        return paths


class RadiusReconstructionMethod(ABC):
    def __init__(self) -> None:
        super().__init__()

    def reconstruct_one(self,
                        starting_radius: float,
                        ending_radius: float,
                        path: np.array
    ) -> np.array:
        raise NotImplementedError


    def reconstruct(self, 
                    graph: Graph,
                    new_edges: torch.Tensor,
                    paths: list[np.array]
    ) -> list[np.array]:
        raise NotImplementedError
    

class ReconstructionMethod():
    def __init__(self,
                 path_reconstruction: PathReconstructionMethod,
                 radius_reconstruction: RadiusReconstructionMethod
    ) -> None:
        super().__init__()
        self.path_reconstruction = path_reconstruction
        self.radius_reconstruction = radius_reconstruction

    @classmethod
    def get_reconstruction_mask(cls,
                                mask: torch.Tensor,
                                paths: list[np.array],
                                radius_paths: list[np.array]
    ) -> torch.Tensor:
        H, W = mask.shape
        reconstruction_map = torch.zeros((H, W), dtype=mask.dtype)

        for path, radius_path in zip(paths, radius_paths):
            for (y, x), radius in zip(path, radius_path):
                y = int(round(y))
                x = int(round(x))
                r = int(round(radius))
                y_min = max(0, y - r)
                y_max = min(H, y + r + 1)
                x_min = max(0, x - r)
                x_max = min(W, x + r + 1)

                for yy in range(y_min, y_max):
                    for xx in range(x_min, x_max):
                        if (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2:
                            reconstruction_map[yy, xx] = 1.0

        return reconstruction_map
    
    @classmethod
    def draw_reconstruction(cls,       
                            mask: torch.Tensor,
                            paths: list[np.array],
                            radius_paths: list[np.array],
                            old_edges_color: np.array = np.array([255, 255, 255]),
                            new_edges_color: np.array = np.array([255, 0, 0])
    ) -> torch.Tensor:
        reconstruction_mask = cls.get_reconstruction_mask(mask, paths, radius_paths)
        img = torch.zeros((mask.shape[0], mask.shape[1], 3), dtype=torch.uint8)
        img[mask.bool()] = torch.tensor(old_edges_color, dtype=torch.uint8)
        img[reconstruction_mask.bool()] = torch.tensor(new_edges_color, dtype=torch.uint8)
        full_mask = torch.logical_or(mask.bool(),reconstruction_mask.bool())
        return img, full_mask

    def reconstruct(self, 
                    map: torch.Tensor,
                    graph: Graph,
                    new_edges: torch.Tensor
                    ) -> tuple[list[np.array], list[np.array]]:
        paths = self.path_reconstruction.reconstruct(map, graph, new_edges)
        radius_paths = self.radius_reconstruction.reconstruct(graph, new_edges, paths)
        return paths, radius_paths