import torch
import torch.nn as nn
from abc import ABC

class EdgePredictor(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, edge_scores: torch.Tensor, query_edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
class EdgePredictorThresholded(EdgePredictor):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, edge_scores: torch.Tensor, query_edge_index: torch.Tensor) -> torch.Tensor:
        above_threshold_mask = edge_scores >= self.threshold
        selected_edge_index = query_edge_index[:, above_threshold_mask]
        selected_edge_scores = edge_scores[above_threshold_mask]
        return selected_edge_index, selected_edge_scores

class EdgePredictorReciprocalMaxThresholded(EdgePredictor):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, edge_scores: torch.Tensor, query_edge_index: torch.Tensor) -> torch.Tensor:
        nodes = query_edge_index.flatten().unique().tolist()

        node_best_friend = {node: None for node in nodes}
        node_best_friend_score = {node: float('-inf') for node in nodes}

        above_threshold_mask = edge_scores >= self.threshold
        above_threshold_scores = edge_scores[above_threshold_mask]
        above_threshold_edges = query_edge_index[:, above_threshold_mask]

        for i, (u, v) in enumerate(above_threshold_edges.t().tolist()):
            score = above_threshold_scores[i].item()
            if score > node_best_friend_score[u]:
                node_best_friend_score[u] = score
                node_best_friend[u] = v
            if score > node_best_friend_score[v]:
                node_best_friend_score[v] = score
                node_best_friend[v] = u

        selected_edge_index = []
        selected_edge_scores = []
        for i, (node, best_friend) in enumerate(zip(nodes, node_best_friend.values())):
            if best_friend is not None:
                best_friend_best_friend = node_best_friend[best_friend]
                if best_friend_best_friend == node:
                    # best friend are reciprocal
                    selected_edge_index.append((min(node, best_friend), max(node, best_friend)))
                    selected_edge_scores.append(node_best_friend_score[node])

        selected_edge_index = torch.tensor(selected_edge_index, dtype=torch.long).t()
        selected_edge_scores = torch.tensor(selected_edge_scores, dtype=edge_scores.dtype)
        return selected_edge_index, selected_edge_scores
    