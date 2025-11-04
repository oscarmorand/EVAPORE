import torch
import torch.nn as nn

class InnerProductDecoder(nn.Module):
    """Decoder that computes edge scores using the inner product of node embeddings."""

    def __init__(self) -> None:
        """Initializes the `InnerProductDecoder`."""
        super().__init__()

    def forward(self, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Computes edge scores based on the inner product of node embeddings.

        Args:
            y: Node embeddings of shape `(num_nodes, embedding_dim)`.
            edge_index: Edge indices of shape `(2, num_edges)`.

        Returns:
            Edge scores of shape `(num_edges,)`.
        """
        src, dst = edge_index
        edge_scores = (y[src] * y[dst]).sum(dim=-1)
        return edge_scores