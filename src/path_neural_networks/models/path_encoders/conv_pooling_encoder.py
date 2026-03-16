import torch
import torch.nn as nn
from typing import Sequence

from path_neural_networks.models.path_encoders import PathEncoder
from path_neural_networks.models.path_encoders.pooling.pooling_operation import PoolingOperation
from path_neural_networks.models.path_encoders.pooling.non_learned_pooling import *
from path_neural_networks.models.path_encoders.convolutional_blocks import ResidualDouble1DConvBlock, Double1DConvBlock

class ConvPoolingPathEncoder(PathEncoder):
    def __init__(self, 
                 in_channels: int,
                 hidden_layers: list[int],
                 pooling_operation: PoolingOperation,
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual_blocks: bool = False,
                 skip_connection: bool = False
    ):
        super(ConvPoolingPathEncoder, self).__init__()

        self.in_channels = in_channels

        if isinstance(hidden_layers, int):
            hidden_layers = [in_channels * (2**l) for l in range(hidden_layers + 1)]
        elif isinstance(hidden_layers, Sequence):
            hidden_layers = [in_channels * (2**l) if n is None else n for l, n in enumerate(hidden_layers)]

        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.residual_blocks = residual_blocks
        self.skip_connection = skip_connection

        layers = []
        for i in range(len(hidden_layers) - 1):
            prev_features, next_features = hidden_layers[i], hidden_layers[i+1]
            if residual_blocks:
                layers.append(ResidualDouble1DConvBlock(prev_features, next_features, kernel_size=kernel_size, padding=padding))
            else:
                layers.append(Double1DConvBlock(prev_features, next_features, kernel_size=kernel_size, padding=padding, final_activation=True))
        
        self.conv_layers = nn.Sequential(*layers)
        self.skip_connection = skip_connection
        if skip_connection:
            self.skip_norm = nn.GroupNorm(1, in_channels)
            self.out_channels = next_features + in_channels
        else:
            self.out_channels = next_features

        if isinstance(pooling_operation, NonLearnedPooling):
            self.out_channels *= pooling_operation.out_channels_factor

        self.pooling_operation = pooling_operation

    def forward(self, 
                path: torch.Tensor # shape (num_features, path_length) or (1, num_features, path_length)
    ) -> torch.Tensor:
        path = super(ConvPoolingPathEncoder, self).forward(path) # shape (1, num_features, path_length)
        
        # Apply 1D convolution followed by ReLU activation
        conv_output = self.conv_layers(path)  # shape (1, out_channels, path_length)

        if self.skip_connection:
            skip = self.skip_norm(path)
            conv_output = torch.cat([conv_output, skip], dim=1)  # shape (1, out_channels + num_features, path_length)

        # Pooling operation (e.g., max pooling) to get fixed-size representation
        pooled_output = self.pooling_operation(conv_output) # shape (1, out_channels)
        return pooled_output

    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "in_channels": self.in_channels,
            "hidden_layers": self.hidden_layers,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "residual_blocks": self.residual_blocks,
            "skip_connection": self.skip_connection,
            "pooling_operation": (
                self.pooling_operation.as_dict()
                if hasattr(self.pooling_operation, "as_dict")
                else self.pooling_operation.__class__.__name__
            ),
            "out_channels": self.out_channels
        }


class ConvMaxPoolingPathEncoder(ConvPoolingPathEncoder):
    def __init__(self, 
                 in_channels: int,
                 hidden_layers: int | list[int],
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual_blocks: bool = False,
                 skip_connection: bool = False
    ):
        super(ConvMaxPoolingPathEncoder, self).__init__(
            in_channels=in_channels,
            hidden_layers=hidden_layers,
            pooling_operation=MaxPooling(),
            kernel_size=kernel_size,
            padding=padding,
            residual_blocks=residual_blocks,
            skip_connection=skip_connection
        )

class ConvMeanPoolingPathEncoder(ConvPoolingPathEncoder):
    def __init__(self, 
                 in_channels: int,
                 hidden_layers: int | list[int],
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual_blocks: bool = False,
                 skip_connection: bool = False
    ):
        super(ConvMeanPoolingPathEncoder, self).__init__(
            in_channels=in_channels,
            hidden_layers=hidden_layers,
            pooling_operation=MeanPooling(),
            kernel_size=kernel_size,
            padding=padding,
            residual_blocks=residual_blocks,
            skip_connection=skip_connection
        )

class ConvMultiStatsPoolingPathEncoder(ConvPoolingPathEncoder):
    def __init__(self, 
                 in_channels: int,
                 hidden_layers: int | list[int],
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual_blocks: bool = False,
                 skip_connection: bool = False
    ):
        super(ConvMultiStatsPoolingPathEncoder, self).__init__(
            in_channels=in_channels,
            hidden_layers=hidden_layers,
            pooling_operation=MultiStatPooling(),
            kernel_size=kernel_size,
            padding=padding,
            residual_blocks=residual_blocks,
            skip_connection=skip_connection
        )