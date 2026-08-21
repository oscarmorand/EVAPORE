"""Train/val augmentation pipelines for 2D images (albumentations) and 3D
volumes (TorchIO), exposing the *same* dict-based callable interface so
`ImageDataset` doesn't need to know which one it's using:

    transform(image=img, mask=gt, fg_mask=fg_mask) -> {"image":..., "mask":..., "fg_mask":...}

`albumentations` only supports 2D arrays - it cannot be reused as-is on 3D
volumes. For those, the standard library is TorchIO (`pip install torchio`),
which mirrors most of what you were doing:

    HorizontalFlip / VerticalFlip   -> tio.RandomFlip (one instance per axis,
                                        so each axis keeps its own probability)
    Affine(rotate, scale, translate)-> tio.RandomAffine
    CropNonEmptyMaskIfExists        -> tio.CropOrPad(patch_size, mask_name=...)
                                        (centers the patch on the mask instead
                                        of a random position that merely
                                        contains it - see note below)
    RandomBrightnessContrast        -> no built-in equivalent, reimplemented
                                        manually (simple mean-centered
                                        contrast + additive brightness)
    Lambda(AddGaussNoise)           -> tio.RandomNoise
    Normalize                       -> manual (image - mean) / std, so it
                                        uses the exact same dataset stats as
                                        the 2D pipeline (from
                                        `ImageDataset.get_dataset_stats`)

Note on axes for `tio.RandomFlip`: 3D "horizontal"/"vertical" isn't a fixed
convention like it is for a 2D photo - it depends on how your volumes are
oriented (e.g. RAS+ after `nib.as_closest_canonical`). Adjust
`horizontal_axis`/`vertical_axis` below to match your data.

Note on `CropOrPad(..., mask_name=...)`: unlike `CropNonEmptyMaskIfExists`,
which picks a *random* crop position among those containing the mask, this
centers the crop on the mask's center of mass (deterministic position, but
still combined with the random flip/rotation applied beforehand, so you
still get positional variety epoch to epoch).
"""
from __future__ import annotations

import random
from typing import Optional, Sequence, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

try:
    import torchio as tio
except ImportError:  # only needed for 3D volumes
    tio = None
    

# ----------------------------------------------------------------------
# 2D (images) - unchanged from your original pipeline, just parameterized
# ----------------------------------------------------------------------

class AddGaussNoise:
    def __init__(self, std: float | tuple[float, float] = 0.01):
        self.std = std

    def __call__(self, image, **kwargs):
        if isinstance(self.std, tuple):
            std = np.random.uniform(self.std[0], self.std[1])
        else:
            std = self.std
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        return np.clip(image + noise, 0.0, 1.0)

def build_image_train_transform(
    mean: Sequence[float],
    std: Sequence[float],
    patch_size: int,
) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Affine(
                rotate=(-5, 5),
                scale=1.0,
                translate_percent=(0, 0),
                shear=0,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.CropNonEmptyMaskIfExists(height=patch_size, width=patch_size, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.15),
                contrast_limit=(-0.15, 0.15),
                p=0.5,
            ),
            A.Lambda(image=AddGaussNoise(std=(0.005, 0.015)), p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ],
        additional_targets={"fg_mask": "mask"},
    )


def build_image_val_transform(mean: Sequence[float], std: Sequence[float]) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ],
        additional_targets={"fg_mask": "mask"},
    )


# ----------------------------------------------------------------------
# 3D (volumes) - TorchIO-based equivalent, same dict interface
# ----------------------------------------------------------------------

def _random_brightness_contrast(
    img: torch.Tensor, brightness_limit: float = 0.15, contrast_limit: float = 0.15, p: float = 0.5
) -> torch.Tensor:
    if random.random() > p:
        return img
    brightness = random.uniform(-brightness_limit, brightness_limit)
    contrast = 1.0 + random.uniform(-contrast_limit, contrast_limit)
    mean = img.mean()
    return (img - mean) * contrast + mean + brightness


class VolumeTransform:
    """Wraps a TorchIO transform to expose the same dict interface as an
    albumentations `A.Compose`: `transform(image=, mask=, fg_mask=)`.

    Assumes single-channel volumes shaped (W, H, D). Manual brightness/
    contrast and normalization are applied after the TorchIO transform,
    since TorchIO has no exact equivalent and to reuse your dataset stats.
    """

    def __init__(
        self,
        tio_transform,
        mean: Sequence[float],
        std: Sequence[float],
        brightness_contrast_p: float = 0.0,
        brightness_limit: float = 0.15,
        contrast_limit: float = 0.15,
    ):
        if tio is None:
            raise ImportError("torchio is required for 3D augmentations. Install it with `pip install torchio`.")
        self.tio_transform = tio_transform
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1, 1)
        self.brightness_contrast_p = brightness_contrast_p
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def __call__(self, image: np.ndarray, mask: np.ndarray, fg_mask: Optional[np.ndarray] = None) -> dict:
        subject_dict = {
            "image": tio.ScalarImage(tensor=self._add_channel(image)),
            "mask": tio.LabelMap(tensor=self._add_channel(mask)),
        }
        if fg_mask is not None:
            subject_dict["fg_mask"] = tio.LabelMap(tensor=self._add_channel(fg_mask))
        subject = tio.Subject(**subject_dict)

        subject = self.tio_transform(subject)

        img = subject["image"].data.float()
        img = _random_brightness_contrast(
            img, self.brightness_limit, self.contrast_limit, p=self.brightness_contrast_p
        )
        img = (img - self.mean) / self.std

        out = {"image": img, "mask": subject["mask"].data.squeeze(0).float()}
        if fg_mask is not None:
            out["fg_mask"] = subject["fg_mask"].data.squeeze(0).float()
        return out

    @staticmethod
    def _add_channel(array: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        tensor = torch.as_tensor(array).float()
        return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor


def build_volume_train_transform(
    mean: Sequence[float],
    std: Sequence[float],
    patch_size: Union[int, Sequence[int]],
    horizontal_axis: int = 2,
    vertical_axis: int = 1,
) -> VolumeTransform:
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)

    tio_transform = tio.Compose(
        [
            tio.RandomFlip(axes=(horizontal_axis,), flip_probability=0.5),
            tio.RandomFlip(axes=(vertical_axis,), flip_probability=0.1),
            tio.RandomAffine(
                scales=0,
                degrees=5,
                translation=0,
                default_pad_value=0,
                default_pad_label=0,
                image_interpolation="linear",
                label_interpolation="nearest",
                p=0.5,
            ),
            tio.CropOrPad(patch_size, mask_name="mask"),
            tio.RandomNoise(std=(0.005, 0.015), p=0.5, include=["image"]),
        ]
    )
    return VolumeTransform(
        tio_transform, mean=mean, std=std, brightness_contrast_p=0.5,
        brightness_limit=0.15, contrast_limit=0.15,
    )


def build_volume_val_transform(mean: Sequence[float], std: Sequence[float]) -> VolumeTransform:
    # No augmentation, no crop needed (val_batch_size=1 -> no shape mismatch
    # across a batch), just normalize with the same dataset stats.
    return VolumeTransform(tio.Compose([]), mean=mean, std=std, brightness_contrast_p=0.0)


# ----------------------------------------------------------------------
# General functions to call
# ----------------------------------------------------------------------


def build_val_transform(ndim: int,
                        mean: Sequence[float], 
                        std: Sequence[float]
) -> A.Compose | VolumeTransform:
    
    if ndim == 2:
        return build_image_val_transform(mean, std)
    elif ndim == 3:
        return build_volume_val_transform(mean, std)
    raise ValueError(
        f"Unsupported number of dimensions {ndim}. Expected 2 or 3 dimensions."
    )

def build_train_transform(ndim: int,
                          mean: Sequence[float],
                          std: Sequence[float],
                          patch_size: Union[int, Sequence[int]]
) -> A.Compose | VolumeTransform:
    
    if ndim == 2:
        return build_image_train_transform(mean, std, patch_size)
    elif ndim == 3:
        return build_volume_train_transform(mean, std, patch_size)
    raise ValueError(
        f"Unsupported number of dimensions {ndim}. Expected 2 or 3 dimensions."
    )