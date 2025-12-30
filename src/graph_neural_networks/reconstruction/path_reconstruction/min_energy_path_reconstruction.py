import torch
import numpy as np
import heapq
from networkx import Graph
from abc import ABC
from enum import Enum

from graph_neural_networks.reconstruction.reconstruction_method import PathReconstructionMethod

class Connexity(Enum):
    CONNEX_4 = 0
    CONNEX_8 = 1

class MinEnergyPathReconstructionMethod(PathReconstructionMethod, ABC):
    def __init__(self, connexity = Connexity.CONNEX_8) -> None:
        super().__init__(height_related=True)
        self.connexity = connexity

    def reconstruct_path(self,
                         prev: np.array, 
                         start: tuple[int, int], 
                         goal: tuple[int, int]
    ) -> list[tuple[int, int]] | None:
        path = []
        current = goal

        while current != start:
            path.append(current)
            py, px = prev[current]
            if py == -1:
                return None
            current = (py, px)

        path.append(start)
        path.reverse()
        path = np.array(path)
        return path
    
    def energy_function(self, 
                        heightmap: np.array,
                        x: int, y: int,
                        nx: int, ny: int
    ) -> np.array:
        raise NotImplementedError("Energy function not implemented.")

    def dijkstra_heightmap(self,
                           heightmap: np.array,
                           start: tuple[int, int],  # (y, x) coordinates of the start point
                           goal: tuple[int, int]  # (y, x) coordinates of the goal point
    ) -> tuple[np.array, np.array]:
        H, W = heightmap.shape

        dist = np.full((H, W), np.inf)
        prev = np.full((H, W, 2), -1, dtype=int)
        visited = np.zeros((H, W), dtype=bool)

        dist[start] = 0.0

        pq = []
        heapq.heappush(pq, (0.0, start))

        if self.connexity == Connexity.CONNEX_8:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                         (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while pq:
            current_dist, (y, x) = heapq.heappop(pq)

            if visited[y, x]:
                continue

            visited[y, x] = True

            if (y, x) == goal:
                break

            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx

                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue

                if visited[ny, nx]:
                    continue

                cost = self.energy_function(heightmap, x, y, nx, ny)

                new_dist = current_dist + cost

                if new_dist < dist[ny, nx]:
                    dist[ny, nx] = new_dist
                    prev[ny, nx] = [y, x]
                    heapq.heappush(pq, (new_dist, (ny, nx)))

        return dist, prev

    def reconstruct_one(self,
                        map: torch.Tensor,
                        pos_start: tuple[int, int],
                        pos_goal: tuple[int, int]
    ) -> list[np.array]:
        map_max_val = map.max().item()
        if self.height_related and map_max_val <= 1.0:
            map = map * 255.0
        dist, prev = self.dijkstra_heightmap(map.numpy(), pos_start, pos_goal)
        path = self.reconstruct_path(prev, pos_start, pos_goal)
        return path
    
class ClassicMinEnergyPathReconstructionMethod(MinEnergyPathReconstructionMethod):
    def energy_function(self, 
                        heightmap: np.array,
                        x: int, y: int,
                        nx: int, ny: int
    ) -> np.array:
        xy_dist = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
        dh = heightmap[ny, nx] - heightmap[y, x]
        cost = xy_dist + max(0.0, dh)
        return cost

class SquaredMinEnergyPathReconstructionMethod(MinEnergyPathReconstructionMethod):
    def __init__(self):
        super().__init__()
        self.up_height_factor = 2.0
        self.down_height_factor = 1.0

    def energy_function(self, 
                        heightmap: np.array,
                        x: int, y: int,
                        nx: int, ny: int
    ) -> np.array:
        xy_dist = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
        dh = heightmap[ny, nx] - heightmap[y, x]
        cost = xy_dist + (self.up_height_factor * (max(0.0, dh) ** 2)) + (self.down_height_factor * (min(0.0, dh) ** 2))
        #print(f"Cost from ({x},{y}) to ({nx},{ny}): {cost} (dh: {dh}, xy_dist: {xy_dist})")
        return cost