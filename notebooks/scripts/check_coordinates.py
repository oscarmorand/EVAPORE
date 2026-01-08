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
from PIL import Image
import warnings
import matplotlib.pyplot as plt

# %%
#img = np.array(Image.open("/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds/1_Aweight_best_iou.png"))
img = np.array(Image.open("/home/morand/afs/EVAPORE/data/FIVES/gt/1_A.png"))[:,:,0]

print(img.shape)

plt.imshow(img, cmap='gray')
plt.show()

# %%
from torch_geometric.utils.convert import from_networkx
from graph.graph_creation import img_to_graph

graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)
print(graph)

# %%
data = graph.nodes(data=True)
coordinates = [d['pos'] for k, d in data]
print(coordinates)


# %%
def plot_graph_coordinates_on_img(img, coordinates):
    plt.imshow(img, cmap='gray')
    y_coords = [c[0] for c in coordinates]
    x_coords = [c[1] for c in coordinates]
    plt.scatter(x_coords, y_coords, color='red', s=1)
    plt.show()


# %%
plot_graph_coordinates_on_img(img, coordinates)


# %%
def plot_graph_coordinates(coordinates):
    y_coords = [c[0] for c in coordinates]
    x_coords = [c[1] for c in coordinates]
    plt.scatter(x_coords, y_coords, color='red', s=1)
    # plot x and y axis lines
    plt.axhline(0, color='black',linewidth=0.5, ls='--')
    plt.axvline(0, color='black',linewidth=0.5, ls='--')
    plt.show()


# %%
plot_graph_coordinates(coordinates)

# %%
from graph.graph_utils import find_max_radius_node

def compute_normalized_centered_coords(nx_graph, node_idx, base_image_shape=None):  
    max_radius_node = find_max_radius_node(nx_graph)
    max_node_pos = nx_graph.nodes[max_radius_node]['pos']
    centered_x = nx_graph.nodes[node_idx]['pos'][0] - max_node_pos[0]
    centered_y = nx_graph.nodes[node_idx]['pos'][1] - max_node_pos[1]
    normalized_x = centered_x / (base_image_shape[0] if base_image_shape else 1)
    normalized_y = centered_y / (base_image_shape[1] if base_image_shape else 1)
    return normalized_x, normalized_y


# %%
nodes = graph.nodes(data=False)
print(nodes)

# %%
normalized_centered_coords = [compute_normalized_centered_coords(graph, n, base_image_shape=img.shape) for n in nodes]
print(normalized_centered_coords)

# %%
plot_graph_coordinates(normalized_centered_coords)

# %%
import os

dir = "/home/morand/afs/EVAPORE/data/FIVES/gt/"

gt_names = os.listdir(dir)
gt_paths = [os.path.join(dir, n) for n in gt_names if n.endswith('.png')]

# %%
gt_paths = gt_paths[:5]  # limit to first 5 for testing

for gt_path in gt_paths:
    print(f"Processing {gt_path}...")
    img = np.array(Image.open(gt_path))[:,:,0]
    graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)
    data = graph.nodes(data=True)
    coordinates = [d['pos'] for k, d in data]
    plot_graph_coordinates_on_img(img, coordinates)
    plot_graph_coordinates(coordinates)
    normalized_centered_coords = [compute_normalized_centered_coords(graph, n, base_image_shape=img.shape) for n in graph.nodes(data=False)]
    plot_graph_coordinates(normalized_centered_coords)

# %%
img = np.array(Image.open("/home/morand/afs/EVAPORE/data/FIVES/gt/1_A.png"))[:,:,0]
graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)
print(graph)

# %%
d = 200
N = np.ceil(img.shape[0] / d).astype(np.int32)
cum_radius_map = np.zeros((N, N), dtype=np.float32)

print(cum_radius_map.shape)

# %%
max_radius_node = np.zeros((N, N), dtype=np.int32) - 1
max_radius_map = np.zeros((N, N), dtype=np.float32)

for node, coord in zip(nodes, coordinates):
    radius = graph.nodes[node]['radius']
    y, x = coord
    i = int(y // d)
    j = int(x // d)
    if 0 <= i < N and 0 <= j < N:
        cum_radius_map[i, j] += radius

        if radius > max_radius_map[i, j]:
            max_radius_map[i, j] = radius
            max_radius_node[i, j] = node

# %%
plt.imshow(img)
plt.show()

plt.imshow(cum_radius_map, cmap='hot')
plt.colorbar()
plt.show()

plt.imshow(max_radius_map, cmap='hot')
plt.colorbar()
plt.show()

# %%
max_patch_i = np.argmax(cum_radius_map)
max_patch_coords = np.unravel_index(max_patch_i, cum_radius_map.shape)
print(f"Max patch at index {max_patch_i} with coordinates {max_patch_coords}")

# %%
max_patch_y, max_patch_x = max_patch_coords

max_node = max_radius_node[max_patch_y, max_patch_x]
print(f"Max node in max patch: {max_node}")

# %%
plt.imshow(img)
y,x = graph.nodes[max_node]['pos']
plt.scatter(x, y, color='red', s=50)
plt.show()

# %%
import os

dir = "/home/morand/afs/EVAPORE/data/FIVES/gt/"

gt_names = os.listdir(dir)
gt_paths = [os.path.join(dir, n) for n in gt_names if n.endswith('.png')]


# %%
def compute_reference_coords(coords, center_coords, base_image_shape=None):  
    centered_x = coords[0] - center_coords[0]
    centered_y = coords[1] - center_coords[1]
    normalized_x = centered_x / (base_image_shape[0] if base_image_shape else 1)
    normalized_y = centered_y / (base_image_shape[1] if base_image_shape else 1)
    if not base_image_shape is None:
        width = base_image_shape[0]
        if center_coords[0] > (width / 2):
            normalized_x = -normalized_x
    return normalized_x, normalized_y


# %%
gt_paths = gt_paths[:20]  # limit to first 5 for testing
d = 100

for gt_path in gt_paths:
    print(f"Processing {gt_path}...")
    img = np.array(Image.open(gt_path))[:,:,0]
    graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)
    data = graph.nodes(data=True)
    nodes = [k for k, d in data]
    coordinates = [d['pos'] for k, d in data]

    N = np.ceil(img.shape[0] / d).astype(np.int32)
    cum_radius_map = np.zeros((N, N), dtype=np.float32)

    print(cum_radius_map.shape)

    max_radius_node = np.zeros((N, N), dtype=np.int32) - 1
    max_radius_map = np.zeros((N, N), dtype=np.float32)

    for node, coord in zip(nodes, coordinates):
        radius = graph.nodes[node]['radius']
        y, x = coord
        i = int(y // d)
        j = int(x // d)
        if 0 <= i < N and 0 <= j < N:
            cum_radius_map[i, j] += radius

            if radius > max_radius_map[i, j]:
                max_radius_map[i, j] = radius
                max_radius_node[i, j] = node

    plt.imshow(cum_radius_map, cmap='hot')
    plt.colorbar()
    plt.show()

    max_patch_i = np.argmax(cum_radius_map)
    max_patch_coords = np.unravel_index(max_patch_i, cum_radius_map.shape)
    print(f"Max patch at index {max_patch_i} with coordinates {max_patch_coords}")

    max_patch_y, max_patch_x = max_patch_coords
    max_node = max_radius_node[max_patch_y, max_patch_x]
    print(f"Max node in max patch: {max_node}")

    plt.imshow(img)
    y,x = graph.nodes[max_node]['pos']
    plt.scatter(x, y, color='red', s=50)
    plt.show()
    center_coords = (x, y)

    plot_graph_coordinates(coordinates)
    xy_coords = [(coord[1], coord[0]) for coord in coordinates]

    normalized_centered_coords = [compute_reference_coords(coord, center_coords, base_image_shape=img.shape) for coord in xy_coords]

    yx_normalized_centered_coords = [(coord[1], coord[0]) for coord in normalized_centered_coords]  # swap back to (y, x) for plotting
    plot_graph_coordinates(yx_normalized_centered_coords)

# %%
from graph_neural_networks.data.dataset.dynamic.handcrafted import get_reference_y, get_reference_x

for gt_path in gt_paths:
    print(f"Processing {gt_path}...")
    img = np.array(Image.open(gt_path))[:,:,0]
    graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)

    reference_x = [get_reference_x(graph, n, base_image_shape=img.shape) for n in graph.nodes(data=False)]
    reference_y = [get_reference_y(graph, n, base_image_shape=img.shape) for n in graph.nodes(data=False)]

    coords = list(zip(reference_y, reference_x))

    plot_graph_coordinates(coords)


# %%
