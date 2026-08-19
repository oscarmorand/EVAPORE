import torch
import torch.nn as nn

from path_neural_networks.models.path_encoders import PathEncoder
from path_neural_networks.models.path_encoders.convolutional_blocks import ResidualDouble1DConvBlock, Double1DConvBlock

class ConvMaxPathEncoder(PathEncoder):
    def __init__(self, 
                 in_channels: int,
                 n_hidden_layers: list[int],
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual_blocks: bool = False,
                 skip_connection: bool = False,
                 out_channels: int = None
    ):
        super(ConvMaxPathEncoder, self).__init__()

        layers = []
        prev_features = in_channels
        for hidden_layer_i in range(n_hidden_layers):
            if out_channels is not None and hidden_layer_i == n_hidden_layers - 1:
                next_features = out_channels
            else:
                next_features = prev_features * 2
            if residual_blocks:
                layers.append(ResidualDouble1DConvBlock(prev_features, next_features, kernel_size=kernel_size, padding=padding))
            else:
                layers.append(Double1DConvBlock(prev_features, next_features, kernel_size=kernel_size, padding=padding, final_activation=True))
            prev_features = next_features
        
        self.conv_layers = nn.Sequential(*layers)
        self.skip_connection = skip_connection
        if skip_connection:
            self.skip_norm = nn.GroupNorm(1, in_channels)
            self.out_channels = next_features + in_channels
        else:
            self.out_channels = next_features

    def forward(self, 
                path: torch.Tensor # shape (num_features, path_length) or (1, num_features, path_length)
    ) -> torch.Tensor:
        path = super(ConvMaxPathEncoder, self).forward(path) # shape (1, num_features, path_length)
        
        # Apply 1D convolution followed by ReLU activation
        conv_output = self.conv_layers(path)  # shape (1, out_channels, path_length)

        if self.skip_connection:
            skip = self.skip_norm(path)
            conv_output = torch.cat([conv_output, skip], dim=1)  # shape (1, out_channels + num_features, path_length)

        # Pooling operation (e.g., max pooling) to get fixed-size representation
        pooled_output = torch.max(conv_output, dim=2).values  # shape (1, out_channels + num_features if skip_connection else out_channels)
        return pooled_output
