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
img = np.array(Image.open("/home/morand/afs/datasets/FIVES/train/Ground truth/1_A.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/New Piskel.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/simple_loop_two.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/error_loop_branch.png"))[:,:,0]

print(img.shape)

plt.imshow(img, cmap='gray')
plt.show()

# %%
from graph.graph_creation import img_to_graph

graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)

# %%
from graph.graph_visualization import display_graph_overlay

display_graph_overlay(img, graph)

# %%
from scipy.spatial import Voronoi, voronoi_plot_2d

points = np.array([graph.nodes[n]['pos'] for n in graph.nodes()])
print(points)

vor = Voronoi(points)
voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2, point_size=2)

# %%
from scipy.spatial import cKDTree

h, w = img.shape

tree = cKDTree(points)
yy, xx = np.mgrid[0:h, 0:w]
coords = np.stack((yy.ravel(), xx.ravel()), axis=-1)
_, ids = tree.query(coords)
voronoi_ids = ids.reshape(h, w)

plt.imshow(voronoi_ids)
plt.scatter(points[:, 1], points[:, 0], c='white', s=50, edgecolors='black')
plt.title("Voronoi Regions (each color = closest point ID)")
plt.show()

# %%
voronoi_seg = voronoi_ids * (img > 0)

plt.imshow(voronoi_seg)
plt.title("Voronoi Segmentation of the Image")
plt.show()


# %%
def aggregate_node_features(features, positions, radiuses):
    """Aggregate features in circular regions around given positions.
    
    Args:
        features (np.ndarray): 3D array of shape (H, W, C)
        positions (list of tuple): List of (y, x) positions.
        radiuses (list of int): List of radiuses for each position.

    Returns:
        np.ndarray: Aggregated features of shape (N, C)
        np.ndarray: Boolean mask of the regions used for aggregation of shape (H, W)
    """
    aggregated_features = []
    region_masks = []
    for pos, radius in zip(positions, radiuses):
        y, x = pos
        y_min = np.floor(max(0, y - radius)).astype(int)
        y_max = np.ceil(min(features.shape[0], y + radius + 1)).astype(int)
        x_min = np.floor(max(0, x - radius)).astype(int)
        x_max = np.ceil(min(features.shape[1], x + radius + 1)).astype(int)

        print(f"Position: {pos}, Radius: {radius}, Region: y[{y_min}:{y_max}], x[{x_min}:{x_max}]")

        region = features[y_min:y_max, x_min:x_max]
        region_mask = np.zeros(features.shape[:2], dtype=bool)
        region_mask[y_min:y_max, x_min:x_max] = True
        region_masks.append(region_mask)

        aggregated_feature = np.mean(region, axis=(0, 1))
        aggregated_features.append(aggregated_feature)

    region_masks = np.logical_or.reduce(np.array(region_masks), axis=0)
    return np.array(aggregated_features), region_masks


# %%

positions = []
radiuses = []
for node, data in graph.nodes(data=True):
    positions.append(data['pos'])
    radiuses.append(data.get('radius', 1))

print(positions[:5])
print(radiuses[:5])

features = np.expand_dims(img, axis=-1)  # Make single channel
features = np.repeat(features, repeats=3, axis=-1)  # Make 3 channels
print(features.shape)

aggregated_features, region_masks = aggregate_node_features(features, positions, radiuses)


# %%
print(region_masks.shape)

# %%
plt.imshow(region_masks, cmap='gray')
plt.title("Regions used for Feature Aggregation")
plt.show()

# %%
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Aggregated features shape: {aggregated_features.shape}")


# %%
def aggregate_edge_features(features, centerlines, radiuses=None):
    aggregated_features = []
    region_masks = []

    if radiuses is None:
        for centerline in centerlines:
            edge_mask = np.zeros(features.shape[:2])
            edge_mask[tuple(np.array(centerline).T)] = 1
            region_masks.append(edge_mask)
            edge_mask = np.expand_dims(edge_mask, axis=-1)
            aggregated_feature = np.mean(features * edge_mask, axis=(0, 1))
    else:
        for centerline, radius in zip(centerlines, radiuses):
            edge_mask = np.zeros(features.shape[:2])
            for i, pos in enumerate(centerline):
                y, x = pos
                r = radius[i]
                y_min = np.floor(max(0, y - r)).astype(int)
                y_max = np.ceil(min(features.shape[0], y + r + 1)).astype(int)
                x_min = np.floor(max(0, x - r)).astype(int)
                x_max = np.ceil(min(features.shape[1], x + r + 1)).astype(int)
                edge_mask[y_min:y_max, x_min:x_max] = 1
            region_masks.append(edge_mask)
            edge_mask = np.expand_dims(edge_mask, axis=-1)
            aggregated_feature = np.mean(features * edge_mask, axis=(0, 1))
            aggregated_features.append(aggregated_feature)

    region_masks = np.logical_or.reduce(np.array(region_masks), axis=0)
    return np.array(aggregated_features), region_masks


# %%
centerlines = []
radiuses = []
for u, v, data in graph.edges(data=True):
    centerlines.append(data['centerline'])
    radiuses.append(data.get('radius', 1))

print(centerlines[:5])
print(radiuses[:5])

aggregated_features, region_masks = aggregate_edge_features(features, centerlines, radiuses=None)


# %%
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")

print(aggregated_features.shape)

# %%
plt.imshow(region_masks, cmap='gray')
plt.title("Regions used for Feature Aggregation")
plt.show()

# %%
