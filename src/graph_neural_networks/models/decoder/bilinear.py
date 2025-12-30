import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearDecoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 symmetric: bool = True, 
                 normalize_embeddings: bool = True, 
                 dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.normalize_embeddings = normalize_embeddings
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else None

        self.W = nn.Parameter(torch.Tensor(input_dim, input_dim))
        nn.init.xavier_uniform_(self.W)


    def forward(self, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            y = F.normalize(y, p=2, dim=-1)

        src, dst = edge_index
        src_embedding = y[src]
        dst_embedding = y[dst]

        if self.dropout_layer is not None:
            src_embedding = self.dropout_layer(src_embedding)
            dst_embedding = self.dropout_layer(dst_embedding)

        edge_scores = torch.sum(src_embedding @ self.W * dst_embedding, dim=-1, keepdim=True)
        if self.symmetric:
            edge_scores += torch.sum(dst_embedding @ self.W * src_embedding, dim=-1, keepdim=True)
            edge_scores = edge_scores * 0.5
        
        edge_scores = edge_scores.squeeze(-1)

        return edge_scores