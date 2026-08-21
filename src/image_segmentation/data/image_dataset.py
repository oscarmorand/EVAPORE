"""Dataset for image/volume segmentation.

Supports both 2D images (png, jpg, ...) and 3D volumes (nii, nii.gz, mgz)
transparently, based on file extension, via ``io_utils.load_array``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from image_segmentation.data.io_utils import load_array

_SUPPORTED_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".nii", ".nii.gz", ".mgz", ".mgh",
    ".pt",
)


class ImageDataset(Dataset):
    """Segmentation dataset: image/volume + ground-truth (+ optional foreground mask).

    Transforms are applied here, inside ``__getitem__``, not in the
    DataLoader. This is the standard PyTorch pattern, for two reasons:

    1. Most augmentations (random crop, flip, elastic deformation, ...) are
       per-sample operations that must happen *before* samples are stacked
       into a batch - the DataLoader/collate_fn only sees already-batched
       tensors, which is too late for that.
    2. It lets each split use a different transform pipeline (e.g. augment
       at train time, resize-only at val/test time) while sharing the exact
       same Dataset implementation - only the ``transforms`` object attached
       to each instance changes. That's exactly what ``ImageDatamodule``
       does below: it builds one ``ImageDataset`` per split with its own
       transforms, on the same ``data_dir``.

    The DataLoader's job is only to batch, shuffle and parallelize loading -
    not to transform data.
    """

    IMAGE_DIR = "img"
    GT_DIR = "gt"
    FOREGROUND_MASK_DIR = "foreground_masks"

    def __init__(
        self,
        data_dir: Union[str, Path],
        transforms=None,
        mask_input: bool = False,
        background_fill_value: float = 0.0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_name = self.data_dir.name.split("_")[0]
        self.transforms = transforms
        self.mask_input = mask_input
        self.background_fill_value = background_fill_value

        self.img_dir = self.data_dir / self.IMAGE_DIR
        self.gt_dir = self.data_dir / self.GT_DIR
        self.foreground_mask_dir = self.data_dir / self.FOREGROUND_MASK_DIR

        self.img_paths = self._list_files(self.img_dir)
        self.gt_paths = self._list_files(self.gt_dir)
        self.foreground_mask_paths = self._list_files(self.foreground_mask_dir)
        self.foreground_available = len(self.foreground_mask_paths) > 0

        if self.gt_paths and len(self.gt_paths) != len(self.img_paths):
            raise ValueError(
                f"Found {len(self.img_paths)} images but {len(self.gt_paths)} "
                f"ground truths in {self.data_dir}"
            )
        if self.foreground_available and len(self.foreground_mask_paths) not in (1, len(self.img_paths)):
            raise ValueError(
                f"Expected either 1 shared foreground mask or one per image "
                f"({len(self.img_paths)}), found {len(self.foreground_mask_paths)}."
            )

    def __len__(self):
        return len(self.img_paths)

    def _list_files(self, directory: Path) -> list:
        if not directory.exists():
            return []
        files = [
            f for f in directory.iterdir()
            if f.name.startswith(self.dataset_name)
            and any("".join(f.suffixes).lower().endswith(ext) for ext in _SUPPORTED_EXTENSIONS)
        ]
        return sorted(files)

    @staticmethod
    def _as_tensor(array) -> torch.Tensor:
        return array if torch.is_tensor(array) else torch.as_tensor(array)

    @staticmethod
    def _to_unit_scale(array: np.ndarray) -> np.ndarray:
        """8-bit images -> [0, 1]. Volumes (CT/MRI, ...) keep their native
        intensity scale, since it isn't bounded to 255. Standardize those
        using the dataset mean/std from ``get_dataset_stats`` in your
        transform pipeline instead (e.g. ``A.Normalize(mean=..., std=...)``).
        """
        if array.dtype == np.uint8:
            return array.astype(np.float32) / 255.0
        return array.astype(np.float32)
    
    def _load_foreground_mask(self, idx: int, target_shape: tuple) -> np.ndarray:
        mask_path = (
            self.foreground_mask_paths[idx]
            if len(self.foreground_mask_paths) == len(self.img_paths)
            else self.foreground_mask_paths[0]
        )

        if mask_path.suffix == ".pt":
            fg_mask = torch.load(mask_path).float()
            fg_mask = F.interpolate(
                fg_mask[None, None], size=tuple(int(s) for s in target_shape), mode="nearest"
            ).squeeze()
            fg_mask = (fg_mask > 0).numpy()
        else:
            fg_mask = load_array(mask_path, grayscale=True) > 0

        if fg_mask.shape != tuple(target_shape):
            raise ValueError(
                f"Foreground mask shape {fg_mask.shape} != target shape {target_shape} for {mask_path}"
            )
        return fg_mask.astype(np.float32)


    def __getitem__(self, idx: int):
        img = self._to_unit_scale(load_array(self.img_paths[idx]))
        gt = (load_array(self.gt_paths[idx], grayscale=True) > 0).astype(np.float32)

        fg_mask = None
        if self.foreground_available:
            fg_mask = self._load_foreground_mask(idx, gt.shape)
            gt = gt * fg_mask

            if self.mask_input:
                broadcast_mask = fg_mask[..., None] if img.ndim == fg_mask.ndim + 1 else fg_mask
                img = np.where(broadcast_mask > 0, img, self.background_fill_value).astype(np.float32)

        sample = {"image": img, "mask": gt}
        if fg_mask is not None:
            sample["fg_mask"] = fg_mask

        if self.transforms is not None:
            # NB: if you pass a fg_mask, your albumentations `Compose` needs
            # `additional_targets={"fg_mask": "mask"}` so it receives the
            # same spatial transform as the ground truth.
            sample = self.transforms(**sample)

        img = self._as_tensor(sample["image"]).float()
        gt = self._as_tensor(sample["mask"]).float()
        if gt.ndim == img.ndim - 1:
            gt = gt.unsqueeze(0)

        return img, gt
    
    def get_dataset_stats(self, 
                          split_name: str = None, 
                          split_indices: list[int] = None, 
                          fov: float = None
    ) -> dict:
        filename = 'image_stats.json'
        stats_filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(stats_filepath):
            with open(stats_filepath, 'r') as f:
                all_stats = json.load(f)
        else:
            fg_mask = load_array(mask_path, grayscale=True) > 0

        if fg_mask.shape != tuple(target_shape):
            raise ValueError(
                f"Foreground mask shape {fg_mask.shape} != target shape {target_shape} for {mask_path}"
            )
        return fg_mask.astype(np.float32)

    def _list_files(self, directory: Path) -> list:
        if not directory.exists():
            return []
        files = [
            f for f in directory.iterdir()
            if f.name.startswith(self.dataset_name)
            and any("".join(f.suffixes).lower().endswith(ext) for ext in _SUPPORTED_EXTENSIONS)
        ]
        return sorted(files)

    # ------------------------------------------------------------------
    # Dataset statistics (for normalization via transforms)
    # ------------------------------------------------------------------

    def get_dataset_stats(
        self, split_name: Optional[str] = None, split_indices: Optional[Sequence[int]] = None
    ) -> dict:
        stats_filepath = self.data_dir / "image_stats.json"
        all_stats = json.loads(stats_filepath.read_text()) if stats_filepath.exists() else {}

        split_name = split_name or "full_dataset"
        if split_name in all_stats:
            print(f"Loading dataset stats for split '{split_name}' from {stats_filepath}...")
            stats = all_stats[split_name]
        else:
            if split_indices is None:
                split_indices = list(range(len(self.img_list)))
                
            stats = {}

            width, height, n_channels = self.compute_dataset_image_stats(split_indices)
            stats['image_width'] = width
            stats['image_height'] = height
            stats['n_channels'] = n_channels

            full_img_pixel_values_stats = self.compute_dataset_pixel_values_stats(split_indices, use_foreground_mask=False)
            stats['full_image'] = full_img_pixel_values_stats

            if self.foreground_available:
                foreground_pixel_values_stats = self.compute_dataset_pixel_values_stats(split_indices, use_foreground_mask=True)
                stats['foreground'] = foreground_pixel_values_stats

            if fov is not None and self.foreground_available:
                pixel_res, median_width = self.compute_dataset_resolution(split_indices, fov)
                stats['field_of_view_degrees'] = fov
                stats['estimated_resolution_mm_per_pixel'] = pixel_res
                stats['median_foreground_width_pixels'] = median_width

            all_stats[split_name] = stats
            print(f"Saving dataset stats for split '{split_name}' to {stats_filepath}...")
            with open(stats_filepath, 'w') as f:
                json.dump(all_stats, f, indent=4)

        all_stats[split_name] = stats
        print(f"Saving dataset stats for split '{split_name}' to {stats_filepath}...")
        stats_filepath.write_text(json.dumps(all_stats, indent=4))
        return stats
    
    def compute_dataset_image_stats(self, 
                                    split_indices: list[int]
    ) -> tuple[int, int, int]:
        img_path = self.img_list[split_indices[0]]
        img = np.array(Image.open(img_path), dtype=np.float32)
        width, height = img.shape[:2]
        n_channels = img.shape[2] if len(img.shape) == 3 else 1
        return width, height, n_channels

    def compute_dataset_pixel_values_stats(self, 
                                           split_indices: list[int], 
                                           use_foreground_mask: bool = True
    ) -> dict:
        sum_ = np.zeros(3, dtype=np.float64)
        sum_sq = np.zeros(3, dtype=np.float64)
        n_pixels = 0

        desc_suffix = "foreground pixels" if use_foreground_mask else "all pixels"
        for i in tqdm(split_indices, desc=f"Computing dataset pixel values statistics on images using {desc_suffix}"):
            img_path = self.img_list[i]
            img = np.array(Image.open(img_path), dtype=np.float32) / 255.0

            if self.foreground_available and use_foreground_mask:
                fg_mask = self._load_foreground_mask(i, spatial_shape)
                pixels = pixels[fg_mask.reshape(-1) > 0]

            if channel_sum is None:
                channel_sum = np.zeros(pixels.shape[1], dtype=np.float64)
                channel_sum_sq = np.zeros(pixels.shape[1], dtype=np.float64)

            channel_sum += pixels.sum(axis=0)
            channel_sum_sq += (pixels.astype(np.float64) ** 2).sum(axis=0)
            n_pixels += pixels.shape[0]

        res = {
            'mean': mean.tolist(),
            'std': std.tolist(),
        }
        return res
    
    def compute_dataset_resolution(self, 
                                   split_indices: list[int], 
                                   fov: float
    ) -> float:
        """
        Compute the resolution of the dataset based on the foreground masks and field of view.
        This method is only an estimate of the reel resolution of the dataset, and should be used with caution. 
        It assumes that the foreground masks are accurate and that the field of view is correctly specified.

        Args:
            split_indices (list[int]): The indices of the images to use for computing the resolution.
            fov (float): The field of view of the images in degrees. This should be provided by the user based on the imaging setup used to capture the dataset.

        Returns:
            float: The estimated resolution of the dataset in mm/pixel.
        """
        if fov is None or fov <= 0 or fov > 180:
            raise ValueError(f"Invalid FOV provided: {fov}, must be in the range (0, 180) degrees.")
        if not self.foreground_available:
            raise ValueError("Foreground masks are required to compute dataset resolution, but none are available.")
            
        widths = []
        for i in tqdm(split_indices, desc="Computing dataset resolution statistics on images"):
            fg_mask = self._get_foreground_mask(i, None)
            if fg_mask is not None:
                bounding_box_width = np.sum(np.any(fg_mask, axis=0))
                widths.append(bounding_box_width)
        median_width = int(np.median(widths))
        fov_rad = np.deg2rad(fov)
        pixel_res = 2 * fov_rad * 12 / float(median_width) # Assuming an eyeball radius of 12mm, 24mm diameter being a common approximation for the human eye
        return pixel_res, median_width
    
    def get_dataset_distance_hparams(self):
        hparams_filepath = os.path.join(self.data_dir, 'distances_hparams.json')
        print(f"Loading dataset distance hyperparameters from {hparams_filepath}...")
        if os.path.exists(hparams_filepath):
            with open(hparams_filepath, 'r') as f:
                hparams = json.load(f)
        else:
            raise FileNotFoundError(f"Distance hyperparameters file not found at {hparams_filepath}. Please compute the distance hyperparameters by running the 11_centerline_length_study.ipynb notebook.")
        return hparams
    
    @classmethod
    def length_in_pixels_to_mm(cls, length_in_pixels: float, pixel_res_mm_per_pixel: float):
        return length_in_pixels * pixel_res_mm_per_pixel
    
    @classmethod
    def length_in_mm_to_pixels(cls, length_in_mm: float, pixel_res_mm_per_pixel: float):
        return length_in_mm / pixel_res_mm_per_pixel
    
    @classmethod
    def length_in_pixels_to_another_res(cls, length_in_pixels: float, original_res: float, target_res: float):
        length_in_mm = cls.length_in_pixels_to_mm(length_in_pixels, original_res)
        return cls.length_in_mm_to_pixels(length_in_mm, target_res)
    
    @classmethod
    def res_to_another_res(cls, original_res: float, target_res: float):
        scale_factor = original_res / target_res
        return scale_factor
