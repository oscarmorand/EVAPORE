import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        norm_op: str = 'batch',
        dropout: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        layers = [DoubleConv(input_channels, features_start, norm_op=norm_op, kernel_size=kernel_size)]

        feats = features_start
        # Encoder
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2, norm_op=norm_op, kernel_size=kernel_size))
            feats *= 2

        # Decoder
        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear, norm_op=norm_op, dropout=dropout, kernel_size=kernel_size))
            feats //= 2

        # Final conv
        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        enc_features = []
        out = x
        # Encoder forward
        for i in range(self.num_layers):
            out = self.layers[i](out)
            enc_features.append(out)

        # Decoder forward
        out = enc_features[-1]  # bottleneck
        if self.dropout > 0:
            out = F.dropout2d(out, p=self.dropout, training=self.training)

        for i, up_layer in enumerate(self.layers[self.num_layers:-1]):
            skip = enc_features[-2 - i]
            out = up_layer(out, skip)

        out = self.layers[-1](out)
        return out


class DoubleConv(nn.Module):
    """ [Conv2d => Norm => ReLU] x2 """

    def __init__(self, in_ch: int, out_ch: int, norm_op: str = 'batch', kernel_size: int = 3):
        super().__init__()

        if norm_op == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_op == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm_op == 'layer':
            norm_layer = lambda c: nn.LayerNorm([c, 1, 1])
        elif norm_op == 'group':
            norm_layer = lambda c: nn.GroupNorm(num_groups=4, num_channels=c)
        elif norm_op == 'none':
            norm_layer = None
        else:
            raise ValueError(f"Unsupported norm_op: {norm_op}")

        if norm_layer is not None:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
                norm_layer(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
                norm_layer(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """ MaxPool => DoubleConv """

    def __init__(self, in_ch: int, out_ch: int, norm_op: str = 'batch', kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, norm_op=norm_op, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """ Upsample => Concat skip => DoubleConv """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False, norm_op: str = 'batch', dropout: float = 0.0, kernel_size: int = 3):
        super().__init__()
        self.dropout = dropout

        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch, norm_op=norm_op, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad if needed
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        if self.dropout > 0:
            x1 = F.dropout2d(x1, p=self.dropout, training=self.training)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
