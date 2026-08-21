import torch
import torch.nn as nn

from image_segmentation.models.spatial_ops import SpatialOps

class DoubleConv(nn.Module):
    """ [Conv => Norm => ReLU] x2 """

    def __init__(self, 
                 ops: SpatialOps, 
                 in_ch: int, 
                 out_ch: int, 
                 norm_op: str = 'instance', 
                 kernel_size: int = 3):
        super().__init__()

        self.net = nn.Sequential(
            ops.Conv(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            ops.Norm(out_ch, norm_op),
            nn.ReLU(inplace=True),

            ops.Conv(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            ops.Norm(out_ch, norm_op),
            nn.ReLU(inplace=True)
        )

    def forward(self, 
                x: torch.Tensor
    ) -> torch.Tensor:
        
        return self.net(x)


class Down(nn.Module):
    """ MaxPool => DoubleConv """

    def __init__(self, 
                 ops: SpatialOps, 
                 in_ch: int, 
                 out_ch: int, 
                 norm_op: str = 'batch', 
                 kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            ops.MaxPool(2),
            DoubleConv(ops, in_ch, out_ch, norm_op=norm_op, kernel_size=kernel_size)
        )

    def forward(self, 
                x: torch.Tensor
    ) -> torch.Tensor:
        
        return self.net(x)


class Up(nn.Module):
    """
    Upsample -> Concat skip -> DoubleConv
    """

    def __init__(
        self,
        ops: SpatialOps,
        in_ch: int,
        out_ch: int,
        bilinear: bool = False,
        norm_op: str = "instance",
        dropout: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.ops = ops
        self.dropout = dropout

        if bilinear:
            self.upsample = nn.Sequential(
                ops.Upsample(scale_factor=2),
                ops.Conv(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = ops.ConvTranspose(
                in_ch,
                in_ch // 2,
                kernel_size=2,
                stride=2,
            )

        self.dropout_layer = ops.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv = DoubleConv(
            ops=ops,
            in_ch=in_ch,
            out_ch=out_ch,
            norm_op=norm_op,
            kernel_size=kernel_size,
        )

    def forward(self,
                x: torch.Tensor,
                skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.upsample(x)

        x = self.ops.pad_to_match(x, skip)

        x = self.dropout_layer(x)

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)
