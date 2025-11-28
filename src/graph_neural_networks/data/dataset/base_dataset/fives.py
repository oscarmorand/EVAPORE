import os
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from typing import Any, Callable
import numpy as np
import torch
from graph_neural_networks.utils import RankedLogger
from graph_neural_networks.data.dataset.graph_dataset import GraphDataset

log = RankedLogger(__name__, rank_zero_only=True)

class FIVESGraphDataset(GraphDataset):
    def __init__(self,
                 root: str,
                 name: str = "FIVES",
                 height: int = 2048,
                 width: int = 2048,
                 raw_dir_name: str = "raw",
                 transform: Callable | None = None,
                 node_attrs_filter: list[str] | None = None,
                 edge_attrs_filter: list[str] | None = None,
                 graph_attrs_filter: list[str] | None = None,
                 json_to_nx_kwargs: dict[str, Any] | None = None,
                 n_debug : int = -1
        ) -> None:
            super().__init__(
                root=root,
                name=name,
                height=height,
                width=width,
                raw_dir_name=raw_dir_name,
                transform=transform,
                node_attrs_filter=node_attrs_filter,
                edge_attrs_filter=edge_attrs_filter,
                graph_attrs_filter=graph_attrs_filter,
                json_to_nx_kwargs=json_to_nx_kwargs,
                n_debug=n_debug
            )


class FIVESImgDataset(TorchDataset):
    def __init__(self, 
                 root: str,
                 name: str = "FIVES"
    ) -> None:
        self.name = name
        self.root = root

        self.img_dir = os.path.join(self.root, "Original")
        self.gt_dir = os.path.join(self.root, "Ground truth")

        self.img_paths = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith('.png')])
        self.gt_paths = sorted([os.path.join(self.gt_dir, f) for f in os.listdir(self.gt_dir) if f.endswith('.png')])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Any:
        img_path = self.img_paths[idx]
        gt_path = self.gt_paths[idx]

        img = Image.open(img_path)
        gt = Image.open(gt_path)

        img_arr = np.array(img).astype(np.float32) / 255.0
        gt_arr = np.array(gt).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_arr)
        gt_tensor = torch.from_numpy(gt_arr)

        return img_tensor, gt_tensor
