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
import logging
import numpy as np
import json
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt

# %%
gt = np.array(Image.open("/home/morand/afs/EVAPORE/data/FIVES/gt/1_A.png"))[:,:,0]
pred = np.array(Image.open("/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds/1_Aweight_best_iou.png").resize(gt.shape[1::-1], resample=Image.NEAREST))

print(gt.shape)
print(pred.shape)

plt.imshow(gt, cmap='gray')
plt.show()

plt.imshow(pred, cmap='gray')
plt.show()

# %%
from graph_neural_networks.data.utils.pred_state import get_combined_graph

graph = get_combined_graph(gt, pred)

# %%
from graph.graph_visualization import display_graph_overlay

black = np.zeros_like(gt, dtype=bool)

display_graph_overlay(black, graph, figsize=(100,100))

# %%
from graph_neural_networks.data.utils.pred_state import EdgePredState

lengths_not_in_prediction = []
for u, v, data in graph.edges(data=True):
    print(data)
    pred_state = data.get('edge_pred_state', EdgePredState.IN_PREDICTION)
    if pred_state in [EdgePredState.NOT_IN_PREDICTION, EdgePredState.NOT_IN_PREDICTION.value]:
        length = data.get('length', 0.0)
        lengths_not_in_prediction.append(length)

plt.hist(lengths_not_in_prediction, bins=50)
plt.title("Histogram of lengths of edges not in prediction")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()

# %%
import os

pred_dir = "/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds"
gt_dir = "/home/morand/afs/EVAPORE/data/FIVES/gt"

pred_filenames = sorted(os.listdir(pred_dir))
gt_filenames = sorted(os.listdir(gt_dir))

pred_filenames = [f for f in pred_filenames if f.endswith('.png')]

pred_paths = [os.path.join(pred_dir, f) for f in pred_filenames]
gt_paths = [os.path.join(gt_dir, f) for f in gt_filenames]

# %%
pred_paths = pred_paths
gt_paths = gt_paths

lengths_not_in_prediction = []

for pred_path, gt_path in zip(pred_paths, gt_paths):
    print(gt_path)
    pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))
    gt = np.array(Image.open(gt_path))[:,:,0]

    graph = get_combined_graph(gt, pred)

    for u, v, data in graph.edges(data=True):
        pred_state = data.get('edge_pred_state', EdgePredState.IN_PREDICTION)
        if pred_state in [EdgePredState.NOT_IN_PREDICTION, EdgePredState.NOT_IN_PREDICTION.value]:
            length = data.get('length', 0.0)
            lengths_not_in_prediction.append(length)

# %%
plt.hist(lengths_not_in_prediction, bins=50)
plt.title("Histogram of lengths of edges not in prediction")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()

# %%
cum_hist = np.cumsum(np.histogram(lengths_not_in_prediction, bins=50)[0])
cum_hist = cum_hist / cum_hist[-1]  # Normalize to get cumulative distribution
plt.plot(cum_hist)
plt.title("Cumulative histogram of lengths of edges not in prediction")
plt.xlabel("Length bin")
plt.ylabel("Cumulative Frequency")
plt.show()

# %%
pred_paths = pred_paths
gt_paths = gt_paths

dist_not_in_prediction = []

for pred_path, gt_path in zip(pred_paths, gt_paths):
    print(gt_path)
    pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))
    gt = np.array(Image.open(gt_path))[:,:,0]

    graph = get_combined_graph(gt, pred)

    for u, v, data in graph.edges(data=True):
        pred_state = data.get('edge_pred_state', EdgePredState.IN_PREDICTION)
        if pred_state in [EdgePredState.NOT_IN_PREDICTION, EdgePredState.NOT_IN_PREDICTION.value]:
            u_pos = graph.nodes[u]['pos']
            v_pos = graph.nodes[v]['pos']
            dist = np.linalg.norm(np.array(u_pos) - np.array(v_pos))
            dist_not_in_prediction.append(dist)

# %%
plt.hist(dist_not_in_prediction, bins=50)
plt.title("Histogram of lengths of edges not in prediction")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()

# %%
