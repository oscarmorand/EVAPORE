from lightning import LightningDataModule
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
from lightning.pytorch.trainer.states import TrainerFn
from graph_neural_networks.data.dataset.graph_dataset_builder import GraphDatasetBuilder
from graph_neural_networks.utils import RankedLogger
from graph_neural_networks.data.dataset.graph_dataset import GraphDataset
from torch_geometric.data import Data
from graph_neural_networks.data.edge_splits.edge_split import EdgeSplit
from abc import ABC

log = RankedLogger(__name__, rank_zero_only=True)

class SplitDataModule(LightningDataModule, ABC):
    def __init__(self, 
                 dataset_builder: GraphDatasetBuilder,
                 batch_size: int = 32,
                 num_workers: int = 7
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

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"batch_size={self.batch_size}, "
                f"num_workers={self.num_workers})")

    def setup(self, 
              stage: TrainerFn=None
    ) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, num_workers=self.num_workers)