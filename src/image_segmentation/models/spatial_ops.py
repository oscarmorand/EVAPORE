import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialOps:
    """
    Factory class returning spatial operators for 2D or 3D networks.

    Example
    -------
    ops = SpatialOps(3)

    conv = ops.Conv(16, 32, kernel_size=3, padding=1)
    norm = ops.Norm(32, "instance")
    pool = ops.MaxPool(2)
    up = ops.ConvTranspose(64, 32, kernel_size=2, stride=2)
    """
    def __init__(self, 
                 spatial_dims: int):
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")

        self.spatial_dims = spatial_dims

        self.Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.ConvTranspose = (
            nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
        )

        self.MaxPool = (
            nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d
        )

        self.BatchNorm = (
            nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d
        )

        self.InstanceNorm = (
            nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d
        )

        self.Dropout = (
            nn.Dropout2d if spatial_dims == 2 else nn.Dropout3d
        )

        self.UpsampleMode = (
            "bilinear" if spatial_dims == 2 else "trilinear"
        )


    def Norm(self, 
             channels: int, 
             norm_type: str
    ) -> nn.Module:
        
        norm_type = norm_type.lower()

        if norm_type == "batch":
            return self.BatchNorm(channels)

        elif norm_type == "instance":
            return self.InstanceNorm(channels)

        elif norm_type == "group":
            return nn.GroupNorm(
                num_groups=min(8, channels),
                num_channels=channels,
            )

        elif norm_type == "layer":
            # LayerNorm over channels only
            return nn.GroupNorm(1, channels)

        elif norm_type == "none":
            return nn.Identity()

        raise ValueError(f"Unsupported normalization: {norm_type}")


    def Upsample(self, 
                 scale_factor=2
    ) -> nn.Module:
        
        return nn.Upsample(
            scale_factor=scale_factor,
            mode=self.UpsampleMode,
            align_corners=True,
        )


    def pad_to_match(self, 
        x: torch.Tensor, 
        ref: torch.Tensor
    ) -> torch.Tensor:
        """
        Pads x so that it matches ref spatial dimensions.
        """
        pads = []

        # iterate from the LAST spatial dim to the first, since F.pad
        # expects padding ordered starting from the last dimension
        for i in range(self.spatial_dims):
            diff = ref.shape[-(i + 1)] - x.shape[-(i + 1)]
            pads.extend([diff // 2, diff - diff // 2])

        return F.pad(x, pads)
