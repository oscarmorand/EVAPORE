import torch

from path_neural_networks.models.path_encoders import PathEncoder
from path_neural_networks.models.path_encoders.pooling.non_learned_pooling import *
from path_neural_networks.models.path_encoders.pooling.attention_pooling import AttentionPooling
from path_neural_networks.models.path_encoders.pooling.pooling_operation import PoolingOperation


class PoolingPathEncoder(PathEncoder):
    def __init__(self, 
                 pooling_operation: PoolingOperation,
                 in_channels: int):
        super(PoolingPathEncoder, self).__init__()
        self.pooling_operation = pooling_operation

    def forward(self, 
                path: torch.Tensor # shape (num_features, path_length) or (1, num_features, path_length)
    ) -> torch.Tensor:
        path = super(PoolingPathEncoder, self).forward(path) # shape (1, num_features, path_length)
        return self.pooling_operation(path) # shape (1, num_features)


class NonLearnedPoolingPathEncoder(PoolingPathEncoder):
    def __init__(self, pooling_operation: NonLearnedPooling, in_channels: int):
        super(NonLearnedPoolingPathEncoder, self).__init__(pooling_operation, in_channels)
        self.out_channels_factor = pooling_operation.out_channels_factor
        self.out_channels = in_channels * self.out_channels_factor

    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "pooling_operation": (
                self.pooling_operation.as_dict()
                if hasattr(self.pooling_operation, "as_dict")
                else self.pooling_operation.__class__.__name__
            ),
            "out_channel_factor": self.out_channels_factor,
            "out_channels": self.out_channels
        }

class MaxPoolingPathEncoder(NonLearnedPoolingPathEncoder):
    def __init__(self, in_channels: int):
        super(MaxPoolingPathEncoder, self).__init__(MaxPooling(), in_channels)

class MinPoolingPathEncoder(NonLearnedPoolingPathEncoder):
    def __init__(self, in_channels: int):
        super(MinPoolingPathEncoder, self).__init__(MinPooling(), in_channels)

class MeanPoolingPathEncoder(NonLearnedPoolingPathEncoder):
    def __init__(self, in_channels: int):
        super(MeanPoolingPathEncoder, self).__init__(MeanPooling(), in_channels)

class SumPoolingPathEncoder(NonLearnedPoolingPathEncoder):
    def __init__(self, in_channels: int):
        super(SumPoolingPathEncoder, self).__init__(SumPooling(), in_channels)

class MedianPoolingPathEncoder(NonLearnedPoolingPathEncoder):
    def __init__(self, in_channels: int):
        super(MedianPoolingPathEncoder, self).__init__(MedianPooling(), in_channels)

class StdPoolingPathEncoder(NonLearnedPoolingPathEncoder):
    def __init__(self, in_channels: int):
        super(StdPoolingPathEncoder, self).__init__(StdPooling(), in_channels)

class MultiStatPoolingPathEncoder(NonLearnedPoolingPathEncoder):
    def __init__(self, in_channels: int):
        super(MultiStatPoolingPathEncoder, self).__init__(MultiStatPooling(), in_channels)
