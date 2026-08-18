"""LightningDataModule for ImageDataset, with per-split transforms."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union

import albumentations as A
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from image_segmentation.data.image_dataset import ImageDataset
from image_segmentation.data.io_utils import true_stem


class ImageDatamodule(LightningDataModule):
    """Wraps ``ImageDataset`` into train/val/test splits.

    A separate ``ImageDataset`` instance is created for each split so that
    each one can carry its own transform pipeline (e.g. augmentations for
    train, resize-only for val/test) - the underlying files on disk are
    identical, only ``transforms`` differs. ``Subset`` is then used to
    restrict each instance to its split's indices.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split_file_path: Union[str, Path],
        train_split_name: str = "train",
        val_split_ratio: float = 0.2,
        train_transforms: A.Compose = None,
        val_transforms: A.Compose = None,
        test_transforms: A.Compose = None,
        num_workers: int = 16,
        train_batch_size: int = 16,
        val_batch_size: int = 1,
        seed: int = 42,
        shuffle_train: bool = True,
        save_resolved_split: bool = True,
        mask_input: bool = False,
        background_fill_value: float = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.data_dir = Path(data_dir)
        self.split_file_path = Path(split_file_path)
        if not self.split_file_path.exists():
            raise ValueError(f"Split file not found at {self.split_file_path}.")
        self.split_info = json.loads(self.split_file_path.read_text())

        self.train_split_name = train_split_name
        self.val_split_ratio = val_split_ratio

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.seed = seed
        self.shuffle_train = shuffle_train
        self.save_resolved_split = save_resolved_split

        self.mask_input = mask_input
        self.background_fill_value = background_fill_value

    def setup(self, stage: str = None):
        # No transforms needed here - reused to list files, match ids, and
        # compute dataset stats (see `get_dataset_stats` on `self.dataset`).
        self.dataset = ImageDataset(self.data_dir, mask_input=self.mask_input, background_fill_value=self.background_fill_value)
        stem_to_index = {true_stem(p): i for i, p in enumerate(self.dataset.img_paths)}

        train_val_ids = self.split_info[self.train_split_name]
        test_ids = self.split_info["test"]

        train_val_indices = self._resolve_indices(train_val_ids, stem_to_index)
        test_indices = self._resolve_indices(test_ids, stem_to_index)

        val_len = int(len(train_val_indices) * self.val_split_ratio)
        train_len = len(train_val_indices) - val_len
        print(f"Dataset split: Train={train_len}, Val={val_len}, Test={len(test_indices)}")

        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(train_val_indices), generator=generator).tolist()
        train_perm, val_perm = perm[:train_len], perm[train_len:]

        self.train_indices = [train_val_indices[i] for i in train_perm]
        self.val_indices = [train_val_indices[i] for i in val_perm]
        self.test_indices = test_indices

        self.train_dataset = Subset(
            ImageDataset(self.data_dir, transforms=self.train_transforms), self.train_indices
        )
        self.val_dataset = Subset(
            ImageDataset(self.data_dir, transforms=self.val_transforms), self.val_indices
        )
        self.test_dataset = Subset(
            ImageDataset(self.data_dir, transforms=self.test_transforms), self.test_indices
        )

        if self.save_resolved_split:
            self._save_resolved_split(train_val_ids, train_perm, val_perm)

    @staticmethod
    def _resolve_indices(ids: Sequence[str], stem_to_index: dict) -> list:
        """Match split ids (e.g. 'name_003') to dataset indices by filename
        stem, rather than assuming a numeric id -> index relationship.
        Adjust this if your split file uses a different id convention.
        """
        try:
            return [stem_to_index[i] for i in ids]
        except KeyError as e:
            raise ValueError(f"Split id {e} has no matching file on disk.") from e

    def _save_resolved_split(self, train_val_ids, train_perm, val_perm):
        # Written to a side file so the original split file is never mutated.
        out_path = self.split_file_path.with_suffix(f".seed_{self.seed}.json")
        resolved = {
            "train": [train_val_ids[i] for i in train_perm],
            "val": [train_val_ids[i] for i in val_perm],
        }
        out_path.write_text(json.dumps(resolved, indent=4))

    def _dataloader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, self.train_batch_size, self.shuffle_train)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, self.val_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, self.val_batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()