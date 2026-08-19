import torch
import torch.nn as nn

class ResidualDouble1DConvBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size: int = 3,
                 padding: int = 1
    ):
        super().__init__()

        self.base_conv = Simple1DConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, activation=True)
        self.res_conv = Double1DConvBlock(out_channels, out_channels, kernel_size=kernel_size, padding=padding, final_activation=False)
        self.activation = nn.ReLU()

    def forward(self, 
                path: torch.Tensor
    ) -> torch.Tensor:
        base_path = self.base_conv(path)
        res_path = self.res_conv(base_path)
        sum_path = base_path + res_path
        final_path = self.activation(sum_path)
        return final_path

class Double1DConvBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size: int = 3,
                 padding: int = 1,
                 final_activation: bool = True
    ):
        super().__init__()

        self.net = nn.Sequential(
            Simple1DConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, activation=True),
            Simple1DConvBlock(out_channels, out_channels, kernel_size=kernel_size, padding=padding, activation=final_activation),
        )

    def forward(self, 
                path: torch.Tensor
    ) -> torch.Tensor:
        return self.net(path)
    

class Simple1DConvBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size: int = 3,
                 padding: int = 1,
                 activation: bool = True
    ):
        super().__init__()

        if activation:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.GroupNorm(num_groups=1, num_channels=out_channels),
                nn.ReLU()
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.GroupNorm(num_groups=1, num_channels=out_channels)
            )

    def forward(self, 
                path: torch.Tensor
    ) -> torch.Tensor:
        return self.net(path)