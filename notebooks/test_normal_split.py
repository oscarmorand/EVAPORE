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
path = "/home/morand/afs/EVAPORE/data/FIVES/processed/dynamic_dataset/dynamic_dataset_0/FIVES_001.pt"

pyg_graph = torch.load(path, map_location=torch.device('cpu'), weights_only=False)

print(pyg_graph)

# %%
print(pyg_graph.num_edges, pyg_graph.num_nodes)

# %%
double_side_edges = True

nodes_1, nodes_2 = pyg_graph.edge_index.tolist()
print(len(nodes_1), len(nodes_2))

real_edges = set()

for node_1, node_2 in zip(nodes_1, nodes_2):
    real_edges.add(frozenset((node_1, node_2)))

print(len(real_edges))

# %%
remove_ratio = 0.2
negative_sampling_ratio = 1.0

# %%
from torch_geometric.transforms import RandomLinkSplit

transform = RandomLinkSplit(
                num_val=remove_ratio,
                num_test=0.0,
                is_undirected=True,
                add_negative_train_samples=True,
                neg_sampling_ratio=negative_sampling_ratio,
            )

# %%
train, val, test = transform(pyg_graph)
print(train)

# %%
print(val)
print(test)

# %%
print(val.edge_index)
print(val.edge_attr)
print(val.edge_label)
print(val.edge_label_index)
print(val.edge_label_index.dtype)

# %%
print(val.edge_index.t())

# %%
print(val.edge_label_index.t())

# %%
