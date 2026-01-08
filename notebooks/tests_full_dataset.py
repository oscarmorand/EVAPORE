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
from graph_neural_networks.data import FIVESGraphDataset

# %%
graph_dataset = FIVESGraphDataset(root='data/FIVES')
img_dataset = FIVESImgDataset(root='data/FIVES')

full_dataset = FIVESFullDataset(graph_dataset=graph_dataset, img_dataset=img_dataset)

# %%
import torch

# %%
