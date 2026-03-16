import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

class ImageDataset(Dataset):
    IMAGE_PATH = 'img'
    GT_PATH = 'gt'
    FOREGROUND_MASK_PATH = 'foreground_masks'

    def __init__(self,
                 data_dir: str,
                 transforms=None,
                 compute_stats_only_on_foreground: bool = True
    ):
        super().__init__()
        self.transforms = transforms
        self.compute_stats_only_on_foreground = compute_stats_only_on_foreground

        self.data_dir = data_dir
        self.img_path = os.path.join(self.data_dir, self.IMAGE_PATH)
        self.gt_path = os.path.join(self.data_dir, self.GT_PATH)
        self.foreground_mask_path = os.path.join(self.data_dir, self.FOREGROUND_MASK_PATH)

        self.img_list = self.get_filenames(self.img_path)
        self.gt_list = self.get_filenames(self.gt_path)
        self.foreground_mask_list = self.get_filenames(self.foreground_mask_path)

        self.stats = self.get_dataset_stats()

    def __len__(self):
        return len(self.img_list)
    
    def _get_foreground_mask(self, idx, gt_shape):
        if len(self.foreground_mask_list) == 0:
            return None

        mask_path = (
            self.foreground_mask_list[idx]
            if len(self.foreground_mask_list) > 1
            else self.foreground_mask_list[0]
        )

        if mask_path.endswith(".pt"):
            fg_mask = torch.load(mask_path)

            fg_mask = F.interpolate(
                fg_mask.unsqueeze(0).unsqueeze(0).float(),
                size=gt_shape,
                mode="nearest",
            )

            fg_mask = fg_mask.squeeze()
            fg_mask = (fg_mask > 0).cpu().numpy()

        else:
            fg_mask = np.array(Image.open(mask_path).convert("L"))
            fg_mask = fg_mask > 0

        assert fg_mask.shape == gt_shape
        return fg_mask.astype(bool)


    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_list[idx])).astype(np.float32)
        img = img / 255.0  # Normalize to [0, 1]
        gt = np.array(Image.open(self.gt_list[idx]).convert("L"))
        gt = (gt > 0).astype(np.float32)

        fg_mask = None
        if len(self.foreground_mask_list) > 0:
            fg_mask = self._get_foreground_mask(idx, gt.shape).astype(np.float32)

            img = img * fg_mask[..., None]
            gt = gt * fg_mask

        if self.transforms:
            if fg_mask is not None:
                augmented = self.transforms(image=img, mask=gt, fg_mask=fg_mask)
                fg_mask = augmented["fg_mask"]
            else:
                augmented = self.transforms(image=img, mask=gt)
            img = augmented["image"]
            gt = augmented["mask"].unsqueeze(0).float()

        return img, gt


    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(os.path.join(path, filename))
        return files_list
    
    def get_dataset_stats(self):
        filename = 'image_stats.json'
        if self.compute_stats_only_on_foreground and len(self.foreground_mask_list) > 0:
            filename = 'image_stats_foreground_only.json'
        stats_filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(stats_filepath):
            with open(stats_filepath, 'r') as f:
                stats = json.load(f)
        else:
            print("Computing dataset stats...")
            stats = self.compute_dataset_stats()
            with open(stats_filepath, 'w') as f:
                json.dump(stats, f, indent=4)
        return stats
    
    def compute_dataset_stats(self):
        sum_ = np.zeros(3, dtype=np.float64)
        sum_sq = np.zeros(3, dtype=np.float64)
        n_pixels = 0

        for i, img_file in enumerate(tqdm(self.img_list, desc="Computing dataset stats on images")):
            img = np.array(Image.open(img_file), dtype=np.float32) / 255.0
            fg_mask = None
            if self.compute_stats_only_on_foreground and len(self.foreground_mask_list) > 0:
                fg_mask = self._get_foreground_mask(i, img.shape[:2])
            if fg_mask is not None:
                pixels = img[fg_mask > 0]
            else:
                pixels = img.reshape(-1, 3)

            sum_ += pixels.sum(axis=0)
            sum_sq += (pixels ** 2).sum(axis=0)
            n_pixels += pixels.shape[0]

        mean = sum_ / n_pixels
        std = np.sqrt(sum_sq / n_pixels - mean ** 2)

        res = {
            'mean': mean.tolist(),
            'std': std.tolist(),
        }
        return res