import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, Subset
import albumentations as A

from path_neural_networks.data.image_centerline_dataset import ImageCenterlineDataset

class ImageCenterlineDatamodule(LightningDataModule):
    def __init__(
            self,
            dataset: ImageCenterlineDataset,
            val_split: float = 0.2,
            test_split: float = 0.1,
            train_transforms: A.Compose = None,
            val_transforms: A.Compose = None,
            test_transforms: A.Compose = None,
            num_workers: int = 16,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.full_dataset = dataset
        self.val_split = val_split
        self.test_split = test_split

        self.num_workers = num_workers
        self.seed = seed

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def setup(self, stage=None):
        total_len = len(self.full_dataset)
        val_len = int(total_len * self.val_split)
        test_len = int(total_len * self.test_split)
        train_len = total_len - val_len - test_len
        print(f"Dataset split: Train={train_len}, Val={val_len}, Test={test_len}: Total={total_len}")

        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(total_len, generator=generator)

        self.train_indices = indices[:train_len]
        self.val_indices = indices[train_len:train_len+val_len]
        self.test_indices = indices[train_len+val_len:]

        # Create separate dataset objects for each split
        self.train_dataset = Subset(ImageCenterlineDataset(data_dir=self.full_dataset.data_dir, transforms=self.train_transforms, compute_stats_only_on_foreground=self.full_dataset.compute_stats_only_on_foreground, centerline_dir=self.full_dataset.CENTERLINE_PATH), self.train_indices)
        self.val_dataset = Subset(ImageCenterlineDataset(data_dir=self.full_dataset.data_dir, transforms=self.val_transforms, compute_stats_only_on_foreground=self.full_dataset.compute_stats_only_on_foreground, centerline_dir=self.full_dataset.CENTERLINE_PATH), self.val_indices)
        self.test_dataset = Subset(ImageCenterlineDataset(data_dir=self.full_dataset.data_dir, transforms=self.test_transforms,  compute_stats_only_on_foreground=self.full_dataset.compute_stats_only_on_foreground, centerline_dir=self.full_dataset.CENTERLINE_PATH), self.test_indices)

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)
        return loader
    
    def predict_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)
        return loader