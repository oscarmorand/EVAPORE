from __future__ import annotations

import random
from typing import Optional, Sequence, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

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

def build_train_transform(
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


def build_val_transform(
        mean: Sequence[float], 
        std: Sequence[float]
) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ],
        additional_targets={"fg_mask": "mask"},
    )