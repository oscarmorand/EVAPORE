import torch
import torch.nn as nn
from path_neural_networks.models.path_encoders.pooling.pooling_operation import PoolingOperation


class NonLearnedPooling(nn.Module):
    def __init__(self, out_channels_factor: int = 1):
        super(NonLearnedPooling, self).__init__()
        self.out_channels_factor = out_channels_factor

    def forward(self, x):
        # x: (1, C, T)
        raise NotImplementedError("NonLearnedPooling is an abstract class and cannot be instantiated directly.")
    
    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "out_channels_factor": self.out_channels_factor
        }


class MaxPooling(NonLearnedPooling):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (1, C, T)
        return torch.max(x, dim=2).values # (1, C)
    
class MinPooling(NonLearnedPooling):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (1, C, T)
        return torch.min(x, dim=2).values # (1, C)
    
class MeanPooling(NonLearnedPooling):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (1, C, T)
        return torch.mean(x, dim=2) # (1, C)
    
class SumPooling(NonLearnedPooling):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (1, C, T)
        return torch.sum(x, dim=2) # (1, C)
    
class MedianPooling(NonLearnedPooling):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (1, C, T)
        return torch.median(x, dim=2).values # (1, C)
    
class StdPooling(NonLearnedPooling):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (1, C, T)
        return torch.std(x, dim=2) # (1, C)
    
class MultiStatPooling(NonLearnedPooling):
    def __init__(self):
        super().__init__(out_channels_factor=3)

    def forward(self, x):
        # x: (1, C, T)
        max_pool = torch.max(x, dim=2).values # (1, C)
        min_pool = torch.min(x, dim=2).values # (1, C)
        mean_pool = torch.mean(x, dim=2) # (1, C)

        return torch.cat([max_pool, min_pool, mean_pool], dim=1) # (1, 3*C)