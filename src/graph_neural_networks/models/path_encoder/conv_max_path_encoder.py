import torch
import torch.nn as nn

class ConvMaxPathEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvMaxPathEncoder, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, 
                paths: torch.Tensor # shape (num_paths, num_features, path_length)
    ) -> torch.Tensor:
        # Apply 1D convolution followed by ReLU activation
        conv_output = self.conv1d(paths)
        activated_output = self.relu(conv_output)
        # Pooling operation (e.g., max pooling) to get fixed-size representation
        pooled_output = torch.max(activated_output, dim=2)[0]
        return pooled_output