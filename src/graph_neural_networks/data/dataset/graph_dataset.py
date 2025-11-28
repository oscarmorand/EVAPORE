import os
from torch_geometric.data import Dataset
from typing import Any, Callable
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.io import fs
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

        super().__init__(root, transform)
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_dir_name)

    @property
    def processed_dir(self) -> str:
        if self.classic_dataset or self.dynamic_dir is None:
            return os.path.join(self.root, 'processed')
        return self.dynamic_dir

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
    
    def get(self, idx: int) -> Data:
        processed_path = self.processed_paths[idx]
        data = fs.torch_load(processed_path)
        return data

    def __iter__(self):
        return self
    
    def __next__(self) -> Data:
        if self._iter_index >= self.len():
            self._iter_index = 0
            raise StopIteration
        data = self.get(self._iter_index)
        self._iter_index += 1
        return data