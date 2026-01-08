# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: graph-neural-networks
#     language: python
#     name: python3
# ---

# %%
from graph_neural_networks.data.dataset.fives import FIVESGraphDataset

# %%
root_path = "~/afs/EVAPORE/data/FIVES/"

fives_dataset = FIVESGraphDataset(root=root_path, edge_attrs_filter=['length','min_radius', 'max_radius', 'mean_radius'])

# %%
print(fives_dataset)
print(fives_dataset.raw_paths)
print(fives_dataset.processed_paths)
print(fives_dataset.raw_file_names)
print(fives_dataset.processed_file_names)
print(len(fives_dataset))

# %%
data = fives_dataset.get(0)
print(data)

# %%
print(data.edge_attr)

# %%
from torch_geometric.loader import DataLoader

loader = DataLoader(fives_dataset, batch_size=4, shuffle=True)

for batch in loader:
    print(batch)
    print(batch.batch)
    print(batch.ptr)
    break

# %%
import pytorch_lightning as pl
import torch
from torch_geometric.transforms import RandomLinkSplit
from torch.utils.data import random_split

class FIVESDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32, num_workers=4, val_split=0.1, test_split=0.1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        train_dataset = []
        val_dataset = []
        test_dataset = []

        for data in self.dataset:
            transform = RandomLinkSplit(
                num_val=self.val_split,
                num_test=self.test_split,
                is_undirected=True,
                add_negative_train_samples=True,
                neg_sampling_ratio=1.0,
            )
            train_data, val_data, test_data = transform(data)
            train_dataset.append(train_data)
            val_dataset.append(val_data)
            test_dataset.append(test_data)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


# %%
fives_data_module = FIVESDataModule(fives_dataset, batch_size=4)
fives_data_module.setup()

# %%
