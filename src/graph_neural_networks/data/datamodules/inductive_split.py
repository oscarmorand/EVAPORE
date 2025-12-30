from torch_geometric.transforms import RandomLinkSplit
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import random_split
import numpy.random as random
import torch
from torch_geometric.data import Data
import numpy as np
import os
import json

from graph_neural_networks.data.dataset.graph_dataset_builder import GraphDatasetBuilder
from graph_neural_networks.data.data_targets.data_target import DataTarget
from graph_neural_networks.utils import RankedLogger
from graph_neural_networks.data.datamodules.split_datamodule import SplitDataModule
from graph_neural_networks.data.edge_splits.edge_split import EdgeSplit

log = RankedLogger(__name__, rank_zero_only=True)

class InductiveSplitDataModule(SplitDataModule):
    def __init__(self, 
                 dataset: GraphDatasetBuilder, 
                 edge_split: EdgeSplit,
                 regenerate_splits: bool = False,
                 batch_size: int = 32, 
                 num_workers: int = 7, 
                 val_split: float = 0.1, 
                 test_split: float = 0.1
    ) -> None:
        super().__init__(dataset, batch_size, num_workers)

        self.val_split = val_split
        self.test_split = test_split

        self.edge_split = edge_split

        self.regenerate_splits = regenerate_splits
        self.split_file_name = "custom_split_indices.json"

    def save_split_indices(self, train_indices, val_indices, test_indices) -> None:
        split_file_path = os.path.join(self.dataset.root, self.split_file_name)
        split_indices = {
            "train_indices": list(train_indices),
            "val_indices": list(val_indices),
            "test_indices": list(test_indices)
        }
        log.info(f"Saving split indices to {split_file_path}...")
        with open(split_file_path, 'w') as f:
            json.dump(split_indices, f, indent=4)

    def load_split_indices(self) -> tuple[list[int], list[int], list[int]]:
        split_file_path = os.path.join(self.dataset.root, self.split_file_name)
        log.info(f"Loading split indices from {split_file_path}...")
        with open(split_file_path, 'r') as f:
            split_indices = json.load(f)
        train_indices = split_indices["train_indices"]
        val_indices = split_indices["val_indices"]
        test_indices = split_indices["test_indices"]
        return train_indices, val_indices, test_indices

    def split_file_exists(self) -> bool:
        split_file_path = os.path.join(self.dataset.root, self.split_file_name)
        return os.path.exists(split_file_path)

    def compute_split_indices(self) -> tuple[list[int], list[int], list[int]]:
        N = len(self.dataset)
        len_val = int(N * self.val_split)
        len_test = int(N * self.test_split)
        len_train = N - len_val - len_test

        train_indices, val_indices, test_indices = random_split(range(len(self.dataset)), [len_train, len_val, len_test])
        return train_indices, val_indices, test_indices

    def setup(self, 
              stage: TrainerFn=None
    ) -> None:
        train_dataset, val_dataset, test_dataset, pred_dataset = [], [], [], []

        if self.regenerate_splits or not self.split_file_exists():
            train_indices, val_indices, test_indices = self.compute_split_indices()
            self.save_split_indices(train_indices, val_indices, test_indices)
        else:
            train_indices, val_indices, test_indices = self.load_split_indices()

        for i, data in enumerate(self.dataset):
            graph = self.edge_split(data)

            if i in train_indices:
                train_dataset.append(graph)
            elif i in val_indices:
                val_dataset.append(graph)
            elif i in test_indices:
                test_dataset.append(graph)
                pred_dataset.append(graph)

        if stage == TrainerFn.FITTING:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        elif stage == TrainerFn.VALIDATING:
            self.val_dataset = val_dataset
        elif stage == TrainerFn.TESTING:
            self.test_dataset = test_dataset
        elif stage == TrainerFn.PREDICTING:
            self.pred_dataset = pred_dataset