import torch
import torch.nn as nn
import torch.nn.functional as F

from image_segmentation.models.spatial_ops import SpatialOps
from image_segmentation.models.blocks import DoubleConv, Down, Up

class UNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1,
        spatial_dims: int = 2,
        channels: list[int] = None,
        bilinear: bool = False,
        norm_op: str = "instance",
        dropout: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.ops = SpatialOps(spatial_dims)
        self.kernel_size = kernel_size

        # ---------- Encoder ----------
        self.encoder = nn.ModuleList()

        self.encoder.append(
            DoubleConv(
                ops=self.ops,
                in_ch=input_channels,
                out_ch=channels[0],
                norm_op=norm_op,
                kernel_size=kernel_size,
            )
        )

        for channel_i in range(0, len(channels) - 1):
            self.encoder.append(
                Down(
                    ops=self.ops,
                    in_ch=channels[channel_i],
                    out_ch=channels[channel_i + 1],
                    norm_op=norm_op,
                    kernel_size=kernel_size,
                )
            )

        # ---------- Bottleneck dropout ----------
        self.dropout = (
            self.ops.Dropout(dropout)
            if dropout > 0
            else nn.Identity()
        )

        # ---------- Decoder ----------
        self.decoder = nn.ModuleList()

        for channel_i in range(len(channels) - 1, 0, -1):
            self.decoder.append(
                Up(
                    ops=self.ops,
                    in_ch=channels[channel_i],
                    out_ch=channels[channel_i - 1],
                    bilinear=bilinear,
                    norm_op=norm_op,
                    dropout=dropout,
                    kernel_size=kernel_size,
                )
            )

        # ---------- Segmentation head ----------
        self.head = self.ops.Conv(
            channels[0],
            num_classes,
            kernel_size=1,
        )

    @property
    def receptive_field(self) -> int:
        """
        Theoretical (isotropic) receptive field, in input pixels/voxels,
        of a single output pixel/voxel.

        Computed along the deepest path through the network (input ->
        encoder -> bottleneck -> decoder -> output). At each decoder
        stage the DoubleConv sees both the skip connection and the
        upsampled branch coming from the bottleneck; the skip branch
        always has a *smaller* receptive field than the deep branch, so
        the combined receptive field is that of the deep branch alone --
        walking that single path is enough to get the network's true
        receptive field.

        Assumes stride-1, non-dilated convolutions using the model's
        configured kernel_size, plus the fixed kernel_size=2/stride=2
        MaxPool downsampling and ConvTranspose (or Upsample+1x1 conv)
        upsampling used by this implementation. The bilinear Upsample
        path is approximated as adding no context of its own (it is,
        for a plain ConvTranspose(k=2, s=2)); in practice it blends 2
        neighboring pixels per axis, so treat this as a lower bound
        by a couple pixels when bilinear=True.

        This is a theoretical upper bound on how far an input pixel can
        influence an output pixel, not the *effective* receptive field:
        gradients from distant pixels decay with depth and can vanish
        numerically well before this radius. It also ignores that
        BatchNorm/InstanceNorm layers (norm_op="batch"/"instance")
        compute statistics over the whole spatial extent of a feature
        map, which technically makes every output pixel depend on the
        entire input.
        """

        receptive_field = 1
        jump = 1

        def conv(kernel_size: int, stride: int = 1):
            nonlocal receptive_field, jump
            receptive_field += (kernel_size - 1) * jump
            jump *= stride

        num_down_stages = len(self.decoder)  # == num_layers - 1

        # ---------- initial DoubleConv ----------
        conv(self.kernel_size)
        conv(self.kernel_size)

        # ---------- encoder: MaxPool(2) + DoubleConv per stage ----------
        for _ in range(num_down_stages):
            conv(kernel_size=2, stride=2)
            conv(self.kernel_size)
            conv(self.kernel_size)

        # ---------- decoder: Upsample(x2) + DoubleConv per stage ----------
        for _ in range(num_down_stages):
            # ConvTranspose(k=2,s=2), or Upsample+1x1 conv: both
            # reconstruct each output position from a single, disjoint
            # input position, so upsampling itself adds no context --
            # only the jump halves back down.
            jump //= 2
            conv(self.kernel_size)
            conv(self.kernel_size)

        return receptive_field

    def weights_memory_bytes(self) -> int:
        """
        Estimated memory footprint of the model's weights, in bytes.

        Sums the storage of every parameter (weights, biases, learnable
        norm affine params) and buffer (e.g. running mean/var of
        BatchNorm), each counted as numel * element_size for its own
        dtype. This is a static property of the model -- it does not
        depend on the input, batch size, or spatial resolution, so it
        excludes activations, gradients, and optimizer state, all of
        which scale with the data being passed through the model.
        """

        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        return param_bytes + buffer_bytes

    def estimate_memory_bytes(self, example_batch: torch.Tensor) -> int:
        """
        Estimated total memory footprint (weights + activations), in
        bytes, for a forward pass on `example_batch`.

        Unlike `weights_memory_bytes`, this *does* depend on the data:
        it runs one real forward pass (under torch.no_grad(), in eval
        mode) with forward hooks on every leaf submodule, summing the
        byte size of each hooked module's output tensor(s) to estimate
        the activation memory, then adds `weights_memory_bytes()`.

        Caveats:
          - This sums every intermediate feature map as if they were
            all alive simultaneously. In practice PyTorch frees
            activations that are no longer referenced, so this is an
            upper bound on the true instantaneous peak, not a live
            (e.g. CUDA) memory measurement.
          - It covers inference only: no gradients or optimizer state,
            both of which would add further memory during training
            (gradients roughly double the weight memory; optimizer
            state such as Adam's moments can add more on top).
        """

        activation_bytes = 0

        def record_output(_module, _inputs, output):
            nonlocal activation_bytes
            outputs = output if isinstance(output, (tuple, list)) else [output]
            for out in outputs:
                if isinstance(out, torch.Tensor):
                    activation_bytes += out.numel() * out.element_size()

        # Leaf modules only, so a container's output isn't counted on
        # top of the sub-outputs that make it up.
        hooks = [
            module.register_forward_hook(record_output)
            for module in self.modules()
            if next(module.children(), None) is None
        ]

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                self(example_batch)
        finally:
            for hook in hooks:
                hook.remove()
            self.train(was_training)

        input_bytes = example_batch.numel() * example_batch.element_size()
        return self.weights_memory_bytes() + input_bytes + activation_bytes

    def forward(self,
                x: torch.Tensor
    ) -> torch.Tensor:

        skips = []

        # Encoder
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        # Bottleneck
        x = self.dropout(skips.pop())

        # Decoder
        for block in self.decoder:
            x = block(x, skips.pop())

        # Segmentation head
        return self.head(x)