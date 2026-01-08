from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.utils.data import DataLoader as ImageDataLoader
from typing import Any
from lightning.pytorch.trainer.states import TrainerFn
from torch_geometric.data import Data
from abc import ABC
from enum import Enum

from graph_neural_networks.data.dataset.graph_dataset_builder import GraphDatasetBuilder
from graph_neural_networks.utils import RankedLogger
from graph_neural_networks.data.dataset.graph_dataset import GraphDataset

log = RankedLogger(__name__, rank_zero_only=True)

class SplitDataModule(LightningDataModule, ABC):
    def __init__(self, 
                 dataset_builder: GraphDatasetBuilder,
                 batch_size: int = 32,
                 num_workers: int = 7,
                 mode: str = "processed"
    ) -> None:
        super().__init__()

        log.info(f"Initializing {self.__class__.__name__}...")

        self.dataset_builder: GraphDatasetBuilder = dataset_builder
        self.dataset: GraphDataset = dataset_builder.get_dataset()

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.train_dataset: list[Data] = None
        self.val_dataset: list[Data] = None
        self.test_dataset: list[Data] = None
        self.pred_dataset: list[Data] = None

        self.mode: str = mode

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"batch_size={self.batch_size}, "
                f"num_workers={self.num_workers})")

    def setup(self, 
              stage: TrainerFn=None
    ) -> None:
        raise NotImplementedError

    def get_loader(self, dataset: list[Any]):
        if self.mode == "processed":
            return GraphDataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            return ImageDataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self) -> GraphDataLoader | ImageDataLoader:
        return self.get_loader(self.train_dataset)

    def val_dataloader(self) -> GraphDataLoader | ImageDataLoader:
        return self.get_loader(self.val_dataset)

    def test_dataloader(self) -> GraphDataLoader | ImageDataLoader:
        return self.get_loader(self.test_dataset)
    
    def predict_dataloader(self) -> GraphDataLoader | ImageDataLoader:
        return self.get_loader(self.pred_dataset)