from abc import ABC
from typing import List, Tuple

import torch
import numpy as np
from networkx import Graph


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
                         path: np.ndarray
                         ) -> np.ndarray:
        raise NotImplementedError

    def reconstruct(self,
                     graph: Graph,
                     new_edges: torch.Tensor,
                     paths: List[np.ndarray]
                     ) -> List[np.ndarray]:
        raise NotImplementedError


class ReconstructionMethod:
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
                                 paths: List[np.ndarray],
                                 radius_paths: List[np.ndarray]
                                 ) -> torch.Tensor:
        shape = tuple(mask.shape)  # (H, W) or (D, H, W)
        ndim = len(shape)
        reconstruction_map = torch.zeros(shape, dtype=mask.dtype)

        for path, radius_path in zip(paths, radius_paths):
            for coords, radius in zip(path, radius_path):
                center = np.array([int(round(c)) for c in coords])
                r = int(round(radius))

                # bounding box, clipped to volume bounds
                mins = np.maximum(0, center - r)
                maxs = np.minimum(np.array(shape), center + r + 1)
                if np.any(mins >= maxs):
                    continue

                # local grid of indices relative to center, N-dimensional
                ranges = [np.arange(mins[d], maxs[d]) for d in range(ndim)]
                grids = np.meshgrid(*ranges, indexing='ij')
                sq_dist = sum((g - center[d]) ** 2 for d, g in enumerate(grids))
                sphere_mask = sq_dist <= r ** 2

                slices = tuple(slice(mins[d], maxs[d]) for d in range(ndim))
                region = reconstruction_map[slices]
                region[sphere_mask] = 1.0
                reconstruction_map[slices] = region

        return reconstruction_map

    @classmethod
    def draw_reconstruction(cls,
                             mask: torch.Tensor,
                             paths: List[np.ndarray],
                             radius_paths: List[np.ndarray],
                             old_edges_color: np.ndarray = np.array([255, 255, 255]),
                             new_edges_color: np.ndarray = np.array([255, 0, 0])
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        reconstruction_mask = cls.get_reconstruction_mask(mask, paths, radius_paths)
        img = torch.zeros((*mask.shape, 3), dtype=torch.uint8)
        img[mask.bool()] = torch.tensor(old_edges_color, dtype=torch.uint8)
        img[reconstruction_mask.bool()] = torch.tensor(new_edges_color, dtype=torch.uint8)
        full_mask = torch.logical_or(mask.bool(), reconstruction_mask.bool())
        return img, full_mask

    def reconstruct(self,
                     map: torch.Tensor,
                     graph: Graph,
                     new_edges: torch.Tensor
                     ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        paths = self.path_reconstruction.reconstruct(map, graph, new_edges)
        radius_paths = self.radius_reconstruction.reconstruct(graph, new_edges, paths)
        return paths, radius_paths