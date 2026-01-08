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
import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from graph.graph_creation import img_to_graph
from fire import Fire
import logging
import colorlog
from graph.graph_io import save_graph_to_json
import json
from graph_neural_networks.data.utils.pred_state import get_combined_graph
import matplotlib.pyplot as plt

# %%
pred_dir = "/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds"
gt_dir = "/home/morand/afs/EVAPORE/data/FIVES/gt"

pred_filenames = sorted(os.listdir(pred_dir))
gt_filenames = sorted(os.listdir(gt_dir))

pred_filenames = [f for f in pred_filenames if f.endswith('.png')]

pred_paths = [os.path.join(pred_dir, f) for f in pred_filenames]
gt_paths = [os.path.join(gt_dir, f) for f in gt_filenames]

print(f"Number of prediction files: {len(pred_paths)}")
print(f"Number of ground truth files: {len(gt_paths)}")

# %%
pred_path, gt_path = pred_paths[0], gt_paths[0]
print("Prediction path:", pred_path)
print("Ground Truth path:", gt_path)

# %%
gt = np.array(Image.open(gt_path))[:,:,0]
pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1], resample=Image.NEAREST))

plt.figure(figsize=(10,10))
plt.imshow(pred)
plt.title("Prediction")
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(gt)
plt.title("Ground Truth")
plt.show()

# %% [markdown]
# # Train pipeline

# %%
graph = get_combined_graph(gt, pred)
print(graph)

for u, v, data in graph.edges(data=True):
    print(data.keys())
    break

# %%
from graph.graph_visualization import display_graph_overlay
from skimage.morphology import skeletonize

skel_gt = skeletonize(gt > 0)
display_graph_overlay(skel_gt, graph, figsize=(100,100))

# %%
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper

graph_wrapper = GraphWrapper(graph)

# %%
from graph_neural_networks.data.dataset.dynamic.graph_transforms.oversample_nodes import OversampleNodesTransform

oversample_transform = OversampleNodesTransform(max_dist=50, remove_original_edges=True)

graph_oversampled_wrapper = oversample_transform(graph_wrapper)

display_graph_overlay(skel_gt, graph_oversampled_wrapper.get_graph(), figsize=(100,100))

# %%
print(graph_oversampled_wrapper.in_pred_graph)

# %%
display_graph_overlay(skel_gt, graph_oversampled_wrapper.in_pred_graph, figsize=(100,100))

# %%
from graph_neural_networks.data.dataset.dynamic.graph_transforms.add_edge_closest_cc import AddEdgeClosestCCTransform

add_edge_transform = AddEdgeClosestCCTransform(only_one_edge_per_cc=False)

graph_added_edges_wrapper = add_edge_transform(graph_oversampled_wrapper)

display_graph_overlay(skel_gt, graph_added_edges_wrapper.get_graph(), figsize=(100,100))

# %% [markdown]
# # Inference pipeline

# %%
from graph.graph_creation import img_to_graph

skel_pred = skeletonize(pred > 0)

pred_graph = img_to_graph(pred, clean=True, closing_radius=0, return_pixel_graph=False)

display_graph_overlay(skel_pred, pred_graph, figsize=(100,100))

# %%
pred_graph_wrapper = GraphWrapper(pred_graph)

oversampled_pred_graph_wrapper = oversample_transform(pred_graph_wrapper)

display_graph_overlay(skel_pred, oversampled_pred_graph_wrapper.get_graph(), figsize=(100,100))

# %%
virtual_edges_graph_wrapper = add_edge_transform(oversampled_pred_graph_wrapper)

display_graph_overlay(skel_pred, virtual_edges_graph_wrapper.get_graph(), figsize=(100,100))

# %%
