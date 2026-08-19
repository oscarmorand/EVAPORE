from __future__ import annotations

import random
from typing import Optional, Sequence, Union
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from image_segmentation.data.augmentations import AddGaussNoise

def build_train_transform(
    mean: Sequence[float],
    std: Sequence[float],
    use_data_augmentation: bool = True,
) -> A.Compose:
    if use_data_augmentation:
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.15),
                contrast_limit=(-0.15, 0.15),
                p=0.5
            ),
            A.Lambda(image=AddGaussNoise(std=(0.005, 0.015)), p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2()
        ])


def build_val_transform(
        mean: Sequence[float], 
        std: Sequence[float]
) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        ToTensorV2()
    ])