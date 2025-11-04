from lightning import LightningDataModule
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
from lightning.pytorch.trainer.states import TrainerFn

class LinkSplitDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size=32, num_workers=4, val_split=0.1, test_split=0.1, negative_sampling_ratio=1.0):
        super().__init__()

        self.dataset = dataset
        print("self.dataset:", self.dataset)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.val_split = val_split
        self.test_split = test_split
        self.negative_sampling_ratio = negative_sampling_ratio

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pred_dataset = None

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"batch_size={self.batch_size}, "
                f"num_workers={self.num_workers}, "
                f"val_split={self.val_split}, "
                f"test_split={self.test_split})")

    def setup(self, stage: TrainerFn=None):
        if stage == TrainerFn.PREDICTING:
            self.pred_dataset = self.dataset
            return

        train_dataset = []
        val_dataset = []
        test_dataset = []
        for data in self.dataset:
            transform = RandomLinkSplit(
                num_val=self.val_split,
                num_test=self.test_split,
                is_undirected=True,
                add_negative_train_samples=True,
                neg_sampling_ratio=self.negative_sampling_ratio,
            )
            train_data, val_data, test_data = transform(data)
            train_dataset.append(train_data)
            val_dataset.append(val_data)
            test_dataset.append(test_data)

        if stage == TrainerFn.FITTING:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        elif stage == TrainerFn.VALIDATING:
            self.val_dataset = val_dataset
        elif stage == TrainerFn.TESTING:
            self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)