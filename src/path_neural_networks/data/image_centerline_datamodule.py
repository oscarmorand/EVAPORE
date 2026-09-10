from torch.utils.data import Subset
import torch
import albumentations as A
from pathlib import Path
from typing import Sequence, Union

from path_neural_networks.data.image_centerline_dataset import ImageCenterlineDataset
from image_segmentation.data.image_datamodule import ImageDatamodule
from image_segmentation.data.io_utils import true_stem
from image_segmentation.data.augmentations import VolumeTransform

class ImageCenterlineDatamodule(ImageDatamodule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        split_file_path: Union[str, Path],
        centerline_dirname: str,
        train_split_name: str = 'train',
        val_split_ratio: float = 0.2,
        train_transforms: Union[A.Compose, VolumeTransform] = None,
        val_transforms: Union[A.Compose, VolumeTransform] = None,
        test_transforms: Union[A.Compose, VolumeTransform] = None,
        seed: int = 42,
        shuffle_train: bool = True,
        save_resolved_split: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split_file_path=split_file_path,
            train_split_name=train_split_name,
            val_split_ratio=val_split_ratio,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=1,
            train_batch_size=1,
            val_batch_size=1,
            seed=seed,
            shuffle_train=shuffle_train,
            save_resolved_split=save_resolved_split,
            *args,
            **kwargs
        )

        self.centerline_dirname = centerline_dirname

    def setup(self, stage: str = None):
        # No transforms needed here - reused to list files, match ids, and
        # compute dataset stats (see `get_dataset_stats` on `self.dataset`).
        self.dataset = ImageCenterlineDataset(
            self.data_dir, 
            self.centerline_dirname,
            transforms=None
        )

        self.compute_indices()

        self.train_dataset = Subset(
            ImageCenterlineDataset(
                self.data_dir, 
                self.centerline_dirname, 
                transforms=self.train_transforms
            ), self.train_indices
        )
        self.val_dataset = Subset(
            ImageCenterlineDataset(
                self.data_dir, 
                self.centerline_dirname,
                transforms=self.val_transforms
            ), self.val_indices
        )
        self.test_dataset = Subset(
            ImageCenterlineDataset(
                self.data_dir,
                self.centerline_dirname,
                transforms=self.test_transforms
            ), self.test_indices
        )