import os
import torch
from torch_geometric.data import Dataset
from typing import Any, Callable
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.io import fs
from PIL import Image
import numpy as np

from graph_neural_networks.data.utils.io import json_to_networkx
from graph_neural_networks.data.utils.io import json_to_pyg
from graph_neural_networks.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class GraphDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 name: str,
                 height: int,
                 width: int,
                 raw_dir_name: str,
                 infer_dir_name: str,
                 transform: Callable | None = None, 
                 node_attrs_filter: list[str] | None = None,
                 edge_attrs_filter: list[str] | None = None,
                 graph_attrs_filter: list[str] | None = None,
                 json_to_nx_kwargs: dict[str, Any] | None = None,
                 n_debug : int = -1
    ) -> None:
        self.classic_dataset = True
        self.dynamic_dir = None

        self._json_to_nx_kwargs = json_to_nx_kwargs

        self._nx_to_pyg_kwargs = {
            "group_node_attrs": node_attrs_filter,
            "group_edge_attrs": edge_attrs_filter,
            "group_graph_attrs": graph_attrs_filter,
        }

        self.n_debug = n_debug
        self._iter_index = 0

        root = Path(root) / name if root is not None else None

        self.name = name

        self.height = height
        self.width = width

        self.raw_dir_name = raw_dir_name
        self.infer_dir_name = infer_dir_name

        super().__init__(root, transform)

        self.common_foreground_mask = None
        if len(os.listdir(self.foreground_masks_dir)) == 1:
            common_mask_path = os.path.join(self.foreground_masks_dir, os.listdir(self.foreground_masks_dir)[0])
            self.common_foreground_mask = fs.torch_load(common_mask_path)
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_dir_name)

    @property
    def processed_dir(self) -> str:
        if self.classic_dataset or self.dynamic_dir is None:
            return os.path.join(self.root, 'processed')
        return self.dynamic_dir
    
    @property
    def gt_dir(self) -> str:
        return os.path.join(self.root, 'gt')

    @property
    def img_dir(self) -> str:
        return os.path.join(self.root, 'img')
    
    @property
    def pred_dir(self) -> str:
        return os.path.join(self.root, 'pred')

    @property
    def feature_maps_dir(self) -> str:
        return os.path.join(self.root, 'feature_maps')
    
    @property
    def probability_maps_dir(self) -> str:
        return os.path.join(self.root, 'probability_maps')
    
    @property
    def foreground_masks_dir(self) -> str:
        return os.path.join(self.root, 'foreground_masks')

    @property
    def raw_file_names(self) -> list[str]:
        return [f.name for f in sorted(Path(self.raw_dir).glob("*.json"))]
    
    @property
    def processed_file_names(self) -> list[str]:
        return [f"{Path(f).stem}.pt" for f in self.raw_file_names]
    
    def process(self) -> None:
        raw_paths = self.raw_paths
        processed_path = self.processed_paths
        if self.n_debug >= 0:
            raw_paths = self.raw_paths[: self.n_debug]
            processed_path = self.processed_paths[: self.n_debug]
            
        data_list = [
            json_to_pyg(
                json_path,
                target_attr=None,
                target_dtype=None,
                line_graph=False,
                json_to_nx_kwargs=self._json_to_nx_kwargs,
                nx_to_pyg_kwargs=self._nx_to_pyg_kwargs,
            )
            for json_path in raw_paths
        ]

        for processed_path, data in zip(processed_path, data_list):
            fs.torch_save(data, processed_path)

            self.processed_paths.append(processed_path)

    def len(self) -> int:
        return len(self.processed_file_names)
    
    def get(self, idx: int, **kwargs) -> Data:
        processed_path = self.processed_paths[idx]
        data = fs.torch_load(processed_path)
        return data

    def get_raw(self, idx: int, **kwargs) -> Any:
        raw_path = self.raw_paths[idx]
        nx_graph = json_to_networkx(raw_path, False, {})
        return nx_graph
    
    def get_nx(self, idx: int, **kwargs) -> Any:
        if self.classic_dataset or self.dynamic_dir is None:
            return self.get_raw(idx)
        nx_path = os.path.join(self.dynamic_dir, f"{Path(self.raw_file_names[idx]).stem}.json")
        nx_graph = json_to_networkx(nx_path, False, {})
        return nx_graph
    
    def get_probability_map(self, idx: int, foreground_mask: torch.Tensor | None = None) -> Any:
        probability_map_path = os.path.join(self.probability_maps_dir, f"{Path(self.raw_file_names[idx]).stem}.pt")
        probability_map = fs.torch_load(probability_map_path)[1]
        if foreground_mask is not None:
            min_val = probability_map[foreground_mask].min()
            probability_map[~foreground_mask] = min_val
        return probability_map
    
    def get_gt(self, idx: int, **kwargs) -> Any:
        gt_path = os.path.join(self.gt_dir, f"{Path(self.raw_file_names[idx]).stem}.png")
        gt = np.array(Image.open(gt_path))
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        return gt
    
    def get_img(self, idx: int, **kwargs) -> Any:
        img_path = os.path.join(self.img_dir, f"{Path(self.raw_file_names[idx]).stem}.png")
        img = np.array(Image.open(img_path))
        return img
    
    def get_pred(self, idx: int, **kwargs) -> Any:
        pred_path = os.path.join(self.pred_dir, f"{Path(self.raw_file_names[idx]).stem}.png")
        pred = np.array(Image.open(pred_path))
        return pred
    
    def get_foreground_mask(self, idx: int) -> Any:
        if self.common_foreground_mask is not None:
            return self.common_foreground_mask
        foreground_mask_path = os.path.join(self.foreground_masks_dir, f"{Path(self.raw_file_names[idx]).stem}.pt")
        foreground_mask = fs.torch_load(foreground_mask_path)
        return foreground_mask

    def get_all_from_keys(self, idx: int, keys: list[str] = None, apply_foreground_mask: bool = False) -> dict[str, Any]:
        foreground_mask = None
        if apply_foreground_mask:
            foreground_mask = self.get_foreground_mask(idx)
        keys_to_getters = {
            "processed": self.get,
            "nx": self.get_nx,
            "probability_map": self.get_probability_map,
            "gt": self.get_gt,
            "img": self.get_img,
            "pred": self.get_pred,
        }
        if keys is None:
            keys = list(keys_to_getters.keys())
        result = {}
        for key in keys:
            if key in keys_to_getters:
                result[key] = keys_to_getters[key](idx, foreground_mask=foreground_mask)
            else:
                raise ValueError(f"Key '{key}' is not recognized.")
        return result

    def __iter__(self):
        return self
    
    def __next__(self) -> Data:
        if self._iter_index >= self.len():
            self._iter_index = 0
            raise StopIteration
        data = self.get(self._iter_index)
        self._iter_index += 1
        return data