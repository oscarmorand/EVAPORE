import torch
import torch.nn as nn
from path_neural_networks.models.path_encoders.pooling.pooling_operation import PoolingOperation

class AttentionPooling(PoolingOperation):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (T, C)
        weights = torch.softmax(self.att(x), dim=0)  # (T, 1)
        return (weights * x).sum(dim=0) # (C,)