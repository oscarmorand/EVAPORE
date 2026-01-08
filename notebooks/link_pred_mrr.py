# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: graph-neural-networks
#     language: python
#     name: python3
# ---

# %%
import torch

# %%
logits = torch.randn(5, 16)  # Example node embeddings for 5 nodes and 16 features
edge_label_index = torch.tensor([[2, 4],
                                [3, 1]])
edge_label = torch.tensor([1, 0])

M = torch.matmul(logits, logits.t()) # M.shape = [num_nodes, num_nodes]
M = torch.sigmoid(M) # normalize to [0, 1]
M = M * (1 - torch.eye(M.size(0), device=M.device)) # zero out diagonal entries
print(M)

pos_edges = edge_label_index[:, edge_label == 1] # shape = [2, num_pos_edges]
print(pos_edges)

_, pred_index_mat = torch.topk(M, k=logits.shape[0], dim=0)
print(pred_index_mat)

pos_edges = torch.concat([pos_edges, torch.flip(pos_edges, dims=[0])], dim=1)
print(pos_edges)

source_nodes, target_nodes = pos_edges
print(source_nodes)
print(target_nodes)

mrr = 0.0
for source_node, target_node in zip(source_nodes, target_nodes):
    print(f"Source node: {source_node}, Target node: {target_node}")
    print(f"Predicted ranking of target nodes: {pred_index_mat[:, source_node]}")
    rank = torch.where(pred_index_mat[:, source_node] == target_node)[0].item() + 1  # +1 for rank starting at 1
    print(rank)
    mrr += 1 / rank

mrr /= source_nodes.size(0)
print(f"Mean Reciprocal Rank (MRR): {mrr}")

# %%
from torchmetrics import Metric
from torch import Tensor

class LinkPredMRR(Metric):
    r"""An abstract class for computing link prediction retrieval metrics.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: bool | None = None

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

        self.accum: Tensor
        self.total: Tensor
        
        self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')

    def update(
        self,
        logits: Tensor,
        edge_label_index: Tensor | tuple[Tensor, Tensor],
        edge_label: Tensor
    ) -> None:

        M = torch.matmul(logits, logits.t()) # M.shape = [num_nodes, num_nodes]
        M = torch.sigmoid(M) # normalize to [0, 1]
        M = M * (1 - torch.eye(M.size(0), device=M.device)) # zero out diagonal entries

        pos_edges = edge_label_index[:, edge_label == 1] # shape = [2, num_pos_edges]

        k = self.k if self.k is not None else logits.shape[0]
        _, pred_index_mat = torch.topk(M, k=k, dim=0)

        pos_edges = torch.concat([pos_edges, torch.flip(pos_edges, dims=[0])], dim=1)

        source_nodes, target_nodes = pos_edges

        mrr = 0.0
        for source_node, target_node in zip(source_nodes, target_nodes):
            rank = torch.where(pred_index_mat[:, source_node] == target_node)[0].item() + 1  # +1 for rank starting at 1
            mrr += 1 / rank
        
        self.accum += mrr
        self.total += source_nodes.size(0)

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.zeros_like(self.accum)
        return self.accum / self.total

    def reset(self) -> None:
        r"""Resets metric state variables to their default value."""
        super().reset()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'


# %%
metric = LinkPredMRR(k=None)

metric.update(logits, edge_label_index, edge_label)

print(f"Mean Reciprocal Rank (MRR): {metric.compute().item()}")

# %%
