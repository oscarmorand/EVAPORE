import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

from path_neural_networks.models.features_generators import FeaturesGenerator


class SlidingWindowFeaturesGenerator(nn.Module):
    """
    Generate dense features from large volumes using non-overlapping
    patches with contextual halo.

    Example:
        patch_size=128
        context_margin=32

        extracted patch:
            192³

        returned features:
            128³

    Notes:
        - size_divisor: some networks (e.g. U-Nets with skip connections)
          require the spatial size fed into them to be a multiple of
          2**depth, or the encoder/decoder feature maps won't line up at
          concatenation time. Since patch_size + 2*context_margin isn't
          guaranteed to satisfy this, each extracted patch is padded up
          to the nearest valid size right before inference, and the
          output is cropped back down afterwards. Set to 1 (default) if
          the underlying model has no such constraint (e.g. it already
          re-aligns internally via same-padding + skip cropping/padding).
    """

    def __init__(
        self,
        features_generator: FeaturesGenerator,
        patch_size: int | tuple[int, ...],
        context_margin: int = 0,
        size_divisor: int = 1,
    ):
        super().__init__()

        if features_generator is None:
            raise ValueError(
                "features_generator cannot be None"
            )

        if size_divisor < 1:
            raise ValueError(
                f"size_divisor must be >= 1, got {size_divisor}"
            )

        self.features_generator = features_generator
        self.patch_size = patch_size
        self.context_margin = context_margin
        self.size_divisor = size_divisor


    def resolve_patch_size(
        self,
        ndim: int
    ) -> tuple[int, ...]:

        if isinstance(self.patch_size, int):
            return (self.patch_size,) * ndim

        patch_size = tuple(self.patch_size)

        if len(patch_size) != ndim:
            raise ValueError(
                f"patch_size has {len(patch_size)} dimensions but the "
                f"volume has {ndim} spatial dimensions"
            )

        return patch_size


    @torch.no_grad()
    def forward(
        self,
        volume: torch.Tensor
    ):

        """
        volume:
            (B,C,H,W)
            or
            (B,C,D,H,W)

        returns:
            dense feature map
        """

        if volume.dim() not in (4, 5):
            raise ValueError(
                f"Expected volume of shape (B,C,H,W) or (B,C,D,H,W), "
                f"got {volume.dim()} dimensions with shape {tuple(volume.shape)}."
            )

        ndim = volume.dim() - 2
        patch_size = self.resolve_patch_size(ndim)
        margin = self.context_margin

        padded_volume, remove_padding = self.pad_volume(
            volume,
            patch_size
        )

        # locations are generated over the "core" region only, i.e.
        # padded_volume with the context halo excluded on each side.
        # Note: the core still includes the alignment padding added in
        # pad_volume -- that's intentional, it's what lets patches tile
        # it exactly with no remainder.
        core_shape = tuple(
            size - 2*margin
            for size in padded_volume.shape[2:]
        )

        output_shape = (
            volume.shape[0],
            self.features_generator.out_channels,
            *padded_volume.shape[2:]
        )

        features_volume = None

        for location in self.generate_locations(
            core_shape,
            patch_size
        ):

            # location of useful patch: offset by the halo, since
            # padded_volume starts with a margin-wide context border
            output_slices = self.location_to_slices(
                tuple(
                    x + margin
                    for x in location
                ),
                patch_size
            )

            # extract bigger patch with context; no offset needed here,
            # the halo already supplies the margin on both sides so this
            # never reads out of bounds
            input_slices = self.location_to_slices(
                location,
                tuple(
                    x + 2*margin
                    for x in patch_size
                )
            )

            patch = padded_volume[input_slices]

            if self.is_patch_empty(patch):
                continue

            # some models (e.g. U-Nets with skip connections) require
            # the spatial size to be a multiple of size_divisor, which
            # patch_size + 2*margin isn't guaranteed to satisfy -- pad
            # up to a valid size here, then crop the output back down
            model_input, model_crop = self.pad_to_divisor(
                patch,
                self.size_divisor
            )

            patch_features = None
            with torch.no_grad():
                patch_features = self.features_generator(
                    model_input
                )

            patch_features = patch_features[
                (slice(None), slice(None)) + model_crop
            ]

            # remove context from features
            if margin > 0:
                patch_features = self.remove_context(
                    patch_features,
                    patch_size
                )

            if features_volume is None:
                features_volume = torch.zeros(
                    output_shape,
                    device=patch_features.device,
                    dtype=patch_features.dtype
                )

            features_volume[output_slices] = patch_features


        if features_volume is None:
            # every patch was empty: nothing was generated
            features_volume = torch.zeros(
                output_shape,
                device=volume.device,
                dtype=volume.dtype
            )


        # remove global padding (margin + alignment together)
        return features_volume[
            (slice(None), slice(None))
            + remove_padding
        ]


    # --------------------------------------------------
    # Patch locations
    # --------------------------------------------------

    def generate_locations(
        self,
        volume_shape,
        patch_size
    ):

        axes = []

        for size, patch in zip(
            volume_shape,
            patch_size
        ):

            positions = list(
                range(
                    0,
                    size,
                    patch
                )
            )

            axes.append(positions)

        return product(*axes)


    def location_to_slices(
        self,
        location,
        size
    ):

        return (
            slice(None),
            slice(None),
            *[
                slice(
                    start,
                    start + length
                )
                for start, length in zip(
                    location,
                    size
                )
            ]
        )


    # --------------------------------------------------
    # Context handling
    # --------------------------------------------------

    def remove_context(
        self,
        features,
        patch_size
    ):

        slices = [
            slice(None),
            slice(None),
        ]

        for _ in patch_size:
            slices.append(
                slice(
                    self.context_margin,
                    -self.context_margin
                )
            )

        return features[
            tuple(slices)
        ]


    # --------------------------------------------------
    # Model-size alignment (e.g. U-Net divisibility by 2**depth)
    # --------------------------------------------------

    def pad_to_divisor(
        self,
        patch: torch.Tensor,
        divisor: int
    ) -> tuple[torch.Tensor, tuple]:

        """
        Pads `patch` on the trailing edge of each spatial dimension so
        every spatial size becomes a multiple of `divisor`. Padding is
        added only "after" (not centered) so the crop to recover the
        original size is a simple [0:size] slice on each dimension.
        """

        if divisor == 1:
            identity_crop = tuple(
                slice(0, size)
                for size in patch.shape[2:]
            )
            return patch, identity_crop

        padding = []
        crop = []

        for size in reversed(patch.shape[2:]):
            remainder = size % divisor
            extra = (divisor - remainder) % divisor

            padding.extend([0, extra])
            crop.insert(0, slice(0, size))

        if all(p == 0 for p in padding):
            return patch, tuple(crop)

        padded = self.replicate_pad_unbounded(patch, padding)

        return padded, tuple(crop)


    # --------------------------------------------------
    # Padding
    # --------------------------------------------------

    def pad_volume(
        self,
        volume,
        patch_size
    ):

        """
        Pad volume so each dimension is divisible by patch size, plus a
        context_margin-wide halo on every side so that context patches
        can always be extracted without reading out of bounds.

        Returns the padded volume and the crop (tuple of slices) needed
        to recover the original volume region -- this crop removes the
        alignment padding and the margin halo together, in one step.
        """

        padding = []
        crop = []

        for size, patch in reversed(
            list(
                zip(
                    volume.shape[2:],
                    patch_size
                )
            )
        ):

            remainder = size % patch
            align_pad = (patch - remainder) % patch

            align_before = align_pad // 2
            align_after = align_pad - align_before

            before = align_before + self.context_margin
            after = align_after + self.context_margin

            padding.extend([before, after])
            crop.insert(0, slice(before, before + size))

        volume = self.replicate_pad_unbounded(
            volume,
            padding
        )

        return volume, tuple(crop)


    # --------------------------------------------------
    # Utils
    # --------------------------------------------------

    def is_patch_empty(
        self,
        patch
    ):

        return torch.all(patch == 0).item()


    def replicate_pad_unbounded(
        self,
        volume: torch.Tensor,
        padding: list[int]
    ) -> torch.Tensor:

        """
        Equivalent to F.pad(volume, padding, mode="replicate"), but
        without PyTorch's restriction that padding must be smaller than
        the corresponding dimension size.
        """

        ndim = volume.dim() - 2
        pads_per_dim = [
            (padding[2*i], padding[2*i + 1])
            for i in range(ndim)
        ]
        pads_per_dim = list(reversed(pads_per_dim))

        result = volume

        for dim_offset, (before, after) in enumerate(pads_per_dim):
            dim = 2 + dim_offset
            size = result.shape[dim]

            idx = torch.arange(
                -before,
                size + after,
                device=volume.device
            )
            idx = idx.clamp(0, size - 1)

            result = result.index_select(dim, idx)

        return result

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

        if example_batch.dim() not in (4, 5):
            raise ValueError(
                f"Expected volume of shape (B,C,H,W) or (B,C,D,H,W), "
                f"got {example_batch.dim()} dimensions with shape {tuple(example_batch.shape)}."
            )
        
        ndim = example_batch.dim() - 2
        B, C = example_batch.shape[0], example_batch.shape[1]

        example_patch = torch.zeros(
            size=[B, C] + [self.patch_size + self.context_margin * 2] * ndim,
            dtype=example_batch.dtype
        )

        res = self.features_generator.net.estimate_memory_bytes(example_patch)
        return res
    