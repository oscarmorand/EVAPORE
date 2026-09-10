import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

from image_segmentation.data import ImageDataset
from image_segmentation.data.io_utils import load_array

class ImageCenterlineDataset(ImageDataset):
    CENTERLINE_DIR = 'centerlines'
    PRED_DIR = 'pred'

    def __init__(self,
                 data_dir: str,
                 centerline_dirname: str,
                 transforms = None,
                 mask_input: bool = False,
                 background_fill_value: float = 0.0,
    ):
        super().__init__(data_dir=data_dir, transforms=transforms, mask_input=mask_input, background_fill_value=background_fill_value)

        self.centerline_dirname = centerline_dirname
        self.centerline_dir = self.data_dir / self.CENTERLINE_DIR / centerline_dirname
        self.pred_dir = self.data_dir / self.PRED_DIR

        self.centerline_paths = self._list_files(self.centerline_dir)
        self.pred_paths = self._list_files(self.pred_dir)

    def __getitem__(self, idx):
        img, gt = super().__getitem__(idx)

        pred = (load_array(self.pred_paths[idx], grayscale=True) > 0).astype(np.float32)

        fg_mask = None
        if len(self.foreground_mask_paths) > 0:
            fg_mask = self._load_foreground_mask(idx, pred.shape)
            pred = pred * fg_mask

        with open(self.centerline_paths[idx], 'r') as f:
            centerlines_data = json.load(f)
        train_path_centerlines = centerlines_data["path_centerlines"]
        train_edges_classes = centerlines_data["edges_classes"]

        pred = self._as_tensor(pred).float()
        train_path_centerlines = [self._as_tensor(path).float() for path in train_path_centerlines]
        train_edges_classes = self._as_tensor(train_edges_classes).long()

        return img, (train_path_centerlines, train_edges_classes), (gt, pred)


    def get_dataset_classes_stats(self):
        classes_stats_filepath = os.path.join(self.centerline_dir, 'classes_stats.json')
        if os.path.exists(classes_stats_filepath):
            with open(classes_stats_filepath, 'r') as f:
                classes_stats = json.load(f)
        else:
            print("Computing dataset stats...")
            classes_stats = self.compute_dataset_classes_stats()
            with open(classes_stats_filepath, 'w') as f:
                json.dump(classes_stats, f, indent=4)

        return classes_stats
    
    def compute_dataset_classes_stats(self):
        classes_count = None
        for centerline_file in tqdm(self.centerline_paths, desc="Computing dataset stats on centerline classes"):
            with open(centerline_file, 'r') as f:
                centerlines_data = json.load(f)
            classes = centerlines_data["edges_classes"]
            if len(classes) == 0:
                count = np.array([0, 0])
            else:
                count = np.bincount(classes)
            if classes_count is None:
                classes_count = count
            else:
                classes_count = classes_count + count
        classes_ratio = classes_count / classes_count.sum()

        return {'classes_count': classes_count.tolist(), 'classes_ratio': classes_ratio.tolist()}