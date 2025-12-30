import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatMLPDecoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int | list[int], 
                 symmetric: bool = True,
                 normalize_embeddings: bool = True, 
                 dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.normalize_embeddings = normalize_embeddings
        self.dropout = dropout

        dims = []
        if symmetric:
            dims.append(4 * input_dim)
        else:
            dims.append(2 * input_dim)

        if isinstance(hidden_dim, int):
            dims.append(hidden_dim)
        elif isinstance(hidden_dim, list):
            dims.extend(hidden_dim)
        else:
            raise ValueError("hidden_dim must be an int or a list of ints.")

        layers = []
        for i in range(len(dims) - 1):
            first_layer_dim = dims[i]
            second_layer_dim = dims[i + 1]
            layers.append(nn.Linear(first_layer_dim, second_layer_dim))
            layers.append(nn.ReLU())

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], 1))

        self.layers = nn.Sequential(*layers)


    def forward(self, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            y = F.normalize(y, p=2, dim=-1)

        src, dst = edge_index
        src_embedding = y[src]
        dst_embedding = y[dst]

        if self.symmetric:
            fcn_input = torch.cat([
                src_embedding,
                dst_embedding,
                torch.abs(src_embedding - dst_embedding),
                src_embedding * dst_embedding
            ], dim=-1)
        else:
            fcn_input = torch.cat([src_embedding, dst_embedding], dim=-1)

        edge_scores = self.layers(fcn_input).squeeze(-1)
        
        return edge_scores