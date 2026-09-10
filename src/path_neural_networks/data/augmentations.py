from __future__ import annotations

import random
from typing import Optional, Sequence, Union
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import torchio as tio

from image_segmentation.data.augmentations import VolumeTransform, AddGaussNoise

# ----------------------------------------------------------------------
# 2D (images) 
# ----------------------------------------------------------------------

def build_image_train_transform(
    mean: Sequence[float],
    std: Sequence[float]
) -> A.Compose:
    return A.Compose(
        [
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
# 3D (volumes) 
# ----------------------------------------------------------------------

def build_volume_train_transform(
    mean: Sequence[float],
    std: Sequence[float]
) -> VolumeTransform:

    return VolumeTransform(
        pre_crop_transform=tio.RandomNoise(std=(0.005, 0.015), p=0.5, include=["image"]),
        mean=mean, std=std,
        brightness_contrast_p=0.5,
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
                          std: Sequence[float]
) -> A.Compose | VolumeTransform:
    
    if ndim == 2:
        return build_image_train_transform(mean, std)
    elif ndim == 3:
        return build_volume_train_transform(mean, std)
    raise ValueError(
        f"Unsupported number of dimensions {ndim}. Expected 2 or 3 dimensions."
    )