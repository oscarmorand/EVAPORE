import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

from image_segmentation.data import ImageDataset

class ImageCenterlineDataset(ImageDataset):
    CENTERLINE_PATH = 'centerlines'
    PRED_PATH = 'pred'

    def __init__(self,
                 data_dir: str,
                 centerline_dirname: str,
                 transforms = None,
    ):
        super().__init__(data_dir=data_dir, transforms=transforms)
        self.centerline_dirname = centerline_dirname
        self.centerline_path = os.path.join(self.data_dir, self.CENTERLINE_PATH, centerline_dirname)
        self.pred_path = os.path.join(self.data_dir, self.PRED_PATH)

        self.centerline_list = self.get_filenames(self.centerline_path, extension="json")
        self.pred_list = self.get_filenames(self.pred_path, extension="png")

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_list[idx])).astype(np.float32)
        img = img / 255.0  # Normalize to [0, 1]
        gt = np.array(Image.open(self.gt_list[idx]).convert('L'))
        pred = np.array(Image.open(self.pred_list[idx]).convert('L'))

        fg_mask = None
        if len(self.foreground_mask_list) > 0:
            fg_mask = self._get_foreground_mask(idx, gt.shape).astype(np.float32)

            img = img * fg_mask[..., None]
            gt = gt * fg_mask
            pred = pred * fg_mask

        with open(self.centerline_list[idx], 'r') as f:
            centerlines_data = json.load(f)
        train_path_centerlines = centerlines_data["path_centerlines"]
        train_edges_classes = centerlines_data["edges_classes"]

        if self.transforms:
            img = self.transforms(image=img)['image']
        else:
            img = torch.tensor(img).permute(2, 0, 1)  # Convert to C,H,W

        gt = torch.tensor(gt, dtype=torch.float32)
        pred = torch.tensor(pred, dtype=torch.float32)
        train_path_centerlines = [torch.tensor(path, dtype=torch.float32) for path in train_path_centerlines]
        train_edges_classes = torch.tensor(train_edges_classes, dtype=torch.long)

        return img, (train_path_centerlines, train_edges_classes), (gt, pred)

    def get_dataset_classes_stats(self):
        classes_stats_filepath = os.path.join(self.centerline_path, 'classes_stats.json')
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
        for centerline_file in tqdm(self.centerline_list, desc="Computing dataset stats on centerline classes"):
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