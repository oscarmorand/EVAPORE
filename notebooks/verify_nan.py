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
import networkx as nx
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch import nn
import torch_geometric as pyg
import numpy as np
import PIL.Image as Image

# %%
import os

dir = "/home/morand/afs/EVAPORE/data/FIVES/processed/dynamic_dataset/dynamic_dataset_2/"

names = os.listdir(dir)
paths = [os.path.join(dir, name) for name in names if name.endswith(".pt")]

print(len(paths))

# %%
torch.set_printoptions(threshold=100000)

nan_count = 0

for path in paths:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if torch.isnan(data.x).any() or torch.isnan(data.edge_attr).any():
        print(data.x.shape)
        print(data.x)
        break
        print(f"NaN found in {path}")
        nan_count += 1

print(f"Total files with NaN: {nan_count}")

# %%
dir = "/home/morand/afs/EVAPORE/data/FIVES/feature_maps/qtseg/"
names = os.listdir(dir)
paths = [os.path.join(dir, name) for name in names if name.endswith(".pt")]

print(len(paths))

# %%
for path in paths:
    data = torch.load(path)
    print(data.shape)
    if torch.isnan(data).any():
        print(f"NaN found in {path}")

# %%
