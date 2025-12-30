import torch
import torch.nn as nn
import torch.nn.functional as F

class HadamardMLPDecoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,  
                 normalize_embeddings: bool = True, 
                 dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.dropout = dropout

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))

        layers.append(nn.ReLU())

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.Sequential(*layers)


    def forward(self, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            y = F.normalize(y, p=2, dim=-1)

        src, dst = edge_index
        src_embedding = y[src]
        dst_embedding = y[dst]

        fcn_input = src_embedding * dst_embedding

        edge_scores = self.layers(fcn_input).squeeze(-1)
        
        return edge_scores