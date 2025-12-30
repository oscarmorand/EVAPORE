import torch
import torch.nn as nn

class NoEncoder(nn.Module):
    """No encoder, just passes the input through."""

    def __init__(self, out_channels: int) -> None:
        """Initializes the `NoEncoder`."""
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Passes the input features through unchanged.

        Args:
            x: Input node features of shape `(num_nodes, num_features)`.
            edge_index: Edge indices of shape `(2, num_edges)`.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            The same input node features `x`.
        """
        return x