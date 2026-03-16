import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

class ImageCenterlineDataset(Dataset):
    IMAGE_PATH = 'img'
    CENTERLINE_PATH = 'centerlines'
    GT_PATH = 'gt'
    PRED_PATH = 'pred'
    FOREGROUND_MASK_PATH = 'foreground_masks'

    def __init__(self,
                 data_dir: str,
                 transforms=None,
                 compute_stats_only_on_foreground: bool = True,
                 centerline_dir: str = None
    ):
        super().__init__()

        if centerline_dir is not None:
            self.CENTERLINE_PATH = centerline_dir

        self.data_dir = data_dir
        self.dataset_name = data_dir.split("/")[-1].split("_")[0]
        self.img_path = os.path.join(self.data_dir, self.IMAGE_PATH)
        self.centerline_path = os.path.join(self.data_dir, self.CENTERLINE_PATH)
        self.gt_path = os.path.join(self.data_dir, self.GT_PATH)
        self.pred_path = os.path.join(self.data_dir, self.PRED_PATH)
        self.foreground_mask_path = os.path.join(self.data_dir, self.FOREGROUND_MASK_PATH)

        self.img_list = self.get_filenames(self.img_path, extension="png")
        self.centerline_list = self.get_filenames(self.centerline_path, extension="json")
        self.gt_list = self.get_filenames(self.gt_path, extension="png")
        self.pred_list = self.get_filenames(self.pred_path, extension="png")
        self.foreground_mask_list = self.get_filenames(self.foreground_mask_path, extension="pt")

        self.transforms = transforms
        self.compute_stats_only_on_foreground = compute_stats_only_on_foreground

        self.stats = self.get_dataset_stats()

    def __len__(self):
        return len(self.centerline_list)
    
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
        train_path_centerlines = centerlines_data["train"]["path_centerlines"]
        train_edges_classes = centerlines_data["train"]["edges_classes"]

        eval_edges = centerlines_data["eval"]["edges"]
        nodes_radius = centerlines_data["eval"]["nodes_radius"] 
        eval_nodes_radius = []
        for edge in eval_edges:
            n1, n2 = edge
            r1, r2 = nodes_radius[str(n1)], nodes_radius[str(n2)]
            eval_nodes_radius.append([r1, r2])
        eval_path_centerlines = centerlines_data["eval"]["path_centerlines"]
        eval_full_path_centerlines = centerlines_data["eval"].get("full_path_centerlines", None)

        if self.transforms:
            img = self.transforms(image=img)['image']
        else:
            img = torch.tensor(img).permute(2, 0, 1)  # Convert to C,H,W

        gt = torch.tensor(gt, dtype=torch.float32)
        pred = torch.tensor(pred, dtype=torch.float32)
        train_path_centerlines = [torch.tensor(path, dtype=torch.float32) for path in train_path_centerlines]
        train_edges_classes = torch.tensor(train_edges_classes, dtype=torch.long)
        eval_edges = torch.tensor(eval_edges, dtype=torch.long)
        eval_nodes_radius = torch.tensor(eval_nodes_radius, dtype=torch.float32)
        eval_path_centerlines = [torch.tensor(path, dtype=torch.float32) for path in eval_path_centerlines]
        if eval_full_path_centerlines is not None:
            eval_full_path_centerlines = [torch.tensor(path, dtype=torch.float32) for path in eval_full_path_centerlines]

        return img, (train_path_centerlines, train_edges_classes), (eval_edges, eval_nodes_radius, eval_path_centerlines, eval_full_path_centerlines), (gt, pred)

    def get_dataset_stats(self):
        image_stats_filename = 'image_stats.json'
        if self.compute_stats_only_on_foreground and len(self.foreground_mask_list) > 0:
            image_stats_filename = 'image_stats_foreground_only.json'
        image_stats = {}
        image_stats_filepath = os.path.join(self.data_dir, image_stats_filename)
        if os.path.exists(image_stats_filepath):
            with open(image_stats_filepath, 'r') as f:
                image_stats = json.load(f)
        else:
            print("Computing dataset stats...")
            image_stats = self.compute_dataset_stats()
            with open(image_stats_filepath, 'w') as f:
                json.dump(image_stats, f, indent=4)

        classes_stats_filepath = os.path.join(self.centerline_path, 'classes_stats.json')
        if os.path.exists(classes_stats_filepath):
            with open(classes_stats_filepath, 'r') as f:
                classes_stats = json.load(f)
        else:
            print("Computing dataset stats...")
            classes_stats = self.compute_centerline_stats()
            with open(classes_stats_filepath, 'w') as f:
                json.dump(classes_stats, f, indent=4)

        stats = image_stats | classes_stats
        return stats
    
    def compute_centerline_stats(self):
        classes_count = None
        for centerline_file in tqdm(self.centerline_list, desc="Computing dataset stats on centerline classes"):
            with open(centerline_file, 'r') as f:
                centerlines_data = json.load(f)
            classes = centerlines_data["train"]["edges_classes"]
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

        return {'mean': mean.tolist(), 'std': std.tolist()}


    def get_filenames(self, path: str, extension: str):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        filenames = os.listdir(path)
        filenames = [filename for filename in filenames if (filename.split(".")[0].split("_")[0] == self.dataset_name and filename.split(".")[1] == extension)]
        filenames.sort()
        for filename in filenames:
            files_list.append(os.path.join(path, filename))
        return files_list