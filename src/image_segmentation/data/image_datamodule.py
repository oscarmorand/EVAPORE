import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, Subset
import albumentations as A
import json

from image_segmentation.data.image_dataset import ImageDataset

class ImageDatamodule(LightningDataModule):
    def __init__(
            self,
            dataset: ImageDataset,
            split_file_path: str = None,
            train_split_name: str = 'train',
            val_split_ratio: float = 0.2,
            train_transforms: A.Compose = None,
            val_transforms: A.Compose = None,
            test_transforms: A.Compose = None,
            num_workers: int = 16,
            train_batch_size: int = 16,
            val_batch_size: int = 1,
            seed: int = 42,
            shuffle_train: bool = True,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.split_file_path = split_file_path
        if not os.path.exists(self.split_file_path):
            raise ValueError(f"Split file not found at {self.split_file_path}. Please provide a valid path to the split file.")
        with open(self.split_file_path, 'r') as f:
            self.split_info = json.load(f)

        self.full_dataset = dataset
        self.val_split_ratio = val_split_ratio
        self.train_split_name = train_split_name

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.shuffle_train = shuffle_train

    def setup(self, stage=None):
        self.train_val_split_idx = self.split_info[self.train_split_name]
        self.test_split_idx = self.split_info['test']

        train_val_split_i = torch.tensor([int(id.split('_')[1]) - 1 for id in self.train_val_split_idx])
        test_split_i = torch.tensor([int(id.split('_')[1]) - 1 for id in self.test_split_idx])

        train_val_len = len(self.train_val_split_idx)
        val_len = int(train_val_len * self.val_split_ratio)
        train_len = train_val_len - val_len
        test_len = len(self.test_split_idx)
        print(f"Dataset split: Train={train_len}, Val={val_len}, Test={test_len}")

        generator = torch.Generator().manual_seed(self.seed)
        train_val_perm = torch.randperm(train_val_len, generator=generator)
        train_perm = train_val_perm[:train_len]
        val_perm = train_val_perm[train_len:train_len+val_len]

        self.train_indices = train_val_split_i[train_perm]
        self.val_indices = train_val_split_i[val_perm]
        self.test_indices = test_split_i

        # Create separate dataset objects for each split
        self.train_dataset = Subset(
            ImageDataset(
                data_dir=self.full_dataset.data_dir,
                transforms=self.train_transforms
            ),
            self.train_indices
        )
        self.val_dataset = Subset(
            ImageDataset(
                data_dir=self.full_dataset.data_dir,
                transforms=self.val_transforms
            ),
            self.val_indices
        )
        self.test_dataset = Subset(
            ImageDataset(
                data_dir=self.full_dataset.data_dir,
                transforms=self.test_transforms
            ),
            self.test_indices
        )

        self.split_info[f"seed_{self.seed}"] = {
            "train": [self.train_val_split_idx[i] for i in train_perm],
            "val": [self.train_val_split_idx[i] for i in val_perm]
        }
        with open(self.split_file_path, 'w') as f:
            json.dump(self.split_info, f, indent=4)

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.train_batch_size,
                            shuffle=self.shuffle_train,
                            num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=self.val_batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=self.val_batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader