from torch_geometric.transforms import RandomLinkSplit
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import random_split

from graph_neural_networks.data.dataset.graph_dataset_builder import GraphDatasetBuilder
from graph_neural_networks.utils import RankedLogger
from graph_neural_networks.data.datamodules.split_datamodule import SplitDataModule

log = RankedLogger(__name__, rank_zero_only=True)

class NormalSplitDataModule(SplitDataModule):
    def __init__(self, 
                 dataset: GraphDatasetBuilder, 
                 batch_size: int = 32, 
                 num_workers: int = 7, 
                 val_split: float = 0.1, 
                 test_split: float = 0.1,
                 remove_ratio: float = 0.1, 
                 negative_sampling_ratio: float = 1.0
    ) -> None:
        super().__init__(dataset, batch_size, num_workers)

        self.val_split = val_split
        self.test_split = test_split

        self.remove_ratio = remove_ratio
        self.negative_sampling_ratio = negative_sampling_ratio


    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"batch_size={self.batch_size}, "
                f"num_workers={self.num_workers}, "
                f"val_split={self.val_split}, "
                f"test_split={self.test_split}, "
                f"remove_ratio={self.remove_ratio}, "
                f"negative_sampling_ratio={self.negative_sampling_ratio})")

    def setup(self, 
              stage: TrainerFn=None
    ) -> None:
        if stage == TrainerFn.PREDICTING:
            self.pred_dataset = self.dataset
            return

        train_dataset = []
        val_dataset = []
        test_dataset = []

        N = len(self.dataset)
        len_val = int(N * self.val_split)
        len_test = int(N * self.test_split)
        len_train = N - len_val - len_test

        train_indices, val_indices, test_indices = random_split(range(len(self.dataset)), [len_train, len_val, len_test])

        for i, data in enumerate(self.dataset):
            transform = RandomLinkSplit(
                num_val=self.remove_ratio,
                num_test=0.0,
                is_undirected=True,
                add_negative_train_samples=True,
                neg_sampling_ratio=self.negative_sampling_ratio,
            )

            _, data, _ = transform(data)

            if i in train_indices:
                train_dataset.append(data)
            elif i in val_indices:
                val_dataset.append(data)
            elif i in test_indices:
                test_dataset.append(data)

        if stage == TrainerFn.FITTING:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        elif stage == TrainerFn.VALIDATING:
            self.val_dataset = val_dataset
        elif stage == TrainerFn.TESTING:
            self.test_dataset = test_dataset