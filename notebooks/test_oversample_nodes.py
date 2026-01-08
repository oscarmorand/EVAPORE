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
def oversample_nodes(graph: nx.Graph, max_dist: float) -> nx.Graph:

    # setup edge and node counters
    max_node_id = max(graph.nodes) + 1
    edge_counter = max([data['id'] for _, _, data in graph.edges(data=True)]) + 1

    G = graph.copy()
    for u, v in graph.edges(data=False):
        for _, data in graph.get_edge_data(u, v).items():
            centerline = np.array(data['centerline'])
            radius = np.array(data['radius'])
            length = data['length']
            if length < max_dist:
                continue

            n = int(length // max_dist) + 1
            l = length / n

            acc_length = 0.0
            switch = False
            split_points = []
            lengths = []
            indexes = []
            for i in range(len(centerline) - 1):
                p_0 = centerline[i]
                p_1 = centerline[i + 1]

                local_length= np.linalg.norm(p_1 - p_0)

                condition = (acc_length >= l)
                if switch:
                    condition = ((acc_length + local_length) >= l)

                if condition:
                    split_points.append(p_1)
                    lengths.append(acc_length)
                    indexes.append(i + 1)
                    acc_length = 0.0
                    switch = not switch

                acc_length += local_length

            node_ids = []
            for i, pos in enumerate(split_points):
                node_id = max_node_id
                max_node_id += 1
                G.add_node(
                    node_id, 
                    pos=pos,
                    radius=radius[indexes[i]]
                )
                node_ids.append(node_id)

            indexes = [0] + indexes + [len(centerline) - 1]
            node_ids = [u] + node_ids + [v]
            lengths.append(length - np.sum(lengths))

            for i in range(len(node_ids) - 1):
                i0, i1 = indexes[i], indexes[i + 1]
                n0, n1 = node_ids[i], node_ids[i + 1]
                local_length = lengths[i]
                local_centerline = centerline[i0:i1 + 1]
                local_radius = radius[i0:i1 + 1]
                G.add_edge(
                    n0,
                    n1,
                    id=edge_counter,
                    name=f"edge_{edge_counter}",
                    centerline=local_centerline.tolist(), 
                    radius=local_radius.tolist(),
                    length=float(local_length),
                    min_radius=np.min(local_radius),
                    max_radius=np.max(local_radius),
                    mean_radius=np.mean(local_radius),
                )
                edge_counter += 1

            G.remove_edge(u, v)

    return G


# %%
G = oversample_nodes(graph, max_dist=20.0)

# %%
from graph.graph_visualization import display_graph_overlay

display_graph_overlay(img, G, figsize=(20, 20))

# %%
print(G)

# %%
from graph.graph_visualization import display_topological_edge_classification, display_graph_edge_depth, display_edge_radius_decay, display_edges_hierarchy, display_graph_mean_radius

display_graph_mean_radius(graph, img)
display_graph_mean_radius(G, img)

# %%
display_topological_edge_classification(graph, img)
display_topological_edge_classification(G, img)

# %%
display_graph_edge_depth(graph, img)
display_graph_edge_depth(G, img)

# %%
display_edge_radius_decay(graph, img)
display_edge_radius_decay(G, img)

# %%
from graph.graph_utils import find_max_radius_node
from graph.graph_labeling import generate_graph_hierarchy

graph_max_node = find_max_radius_node(graph)
G_max_node = find_max_radius_node(G)

graph = generate_graph_hierarchy(graph, graph_max_node)
G = generate_graph_hierarchy(G, G_max_node)

display_edges_hierarchy(graph, img)
display_edges_hierarchy(G, img)


# %%
def get_virtual_centerline(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    # find absolute differences
    dx = abs(x0 - x1)
    dy = abs(y0 - y1)

    # find maximum difference
    steps = max(dx, dy)

    # calculate the increment in x and y
    xinc = dx/steps * (1 if x1 > x0 else -1)
    yinc = dy/steps * (1 if y1 > y0 else -1)

    # start with 1st point
    x = float(x0)
    y = float(y0)

    # make a list for coordinates
    x_coorinates = []
    y_coorinates = []

    for i in range(steps):
        # append the x,y coordinates in respective list
        x_coorinates.append(x)
        y_coorinates.append(y)

        # increment the values
        x = x + xinc
        y = y + yinc

    return np.array([x_coorinates, y_coorinates]).T


def get_graph_overlay_img(img: np.ndarray,
                          graph: nx.Graph,
                          show_edges: bool = True
                          ) -> np.ndarray:
    """
    Overlay the graph on the original image for visualization.

    Args:
        img (np.ndarray): Original binary image.
        graph (nx.Graph): Vascular graph to overlay.

    Returns:
        np.ndarray: Image in RGB format with graph overlay.
    """

    # Create an RGB version of the original image
    viz = np.zeros((*img.shape, 3), dtype=np.uint8)
    viz[img > 0] = [255, 255, 255]

    # Overlay edges in red
    if show_edges:
        for u, v, data in graph.edges(data=True):
            centerline = []
            if data.get('virtual_edge', False):
                x0, y0 = graph.nodes[u]['pos']
                x1, y1 = graph.nodes[v]['pos']
                centerline = get_virtual_centerline(x0, y0, x1, y1).astype(int)
                color = [0, 255, 255]  # Cyan for virtual edges
            else:
                centerline = np.array(data['centerline'])
                color = [255, 0, 0]  # Red for real edges
            viz[centerline[:,0], centerline[:,1]] = color

    # Overlay nodes in green
    for n, data in graph.nodes(data=True):
        pos = data['pos']
        viz[pos[0], pos[1]] = [0, 255, 0]

    return viz

def display_graph_overlay(img: np.ndarray,
                       graph: nx.Graph,
                       show_edges: bool = True
                       ) -> None:
    """
    Display the graph overlayed on the original image.

    Args:
        img (np.ndarray): Original binary image.
        graph (nx.Graph): Vascular graph to overlay.
    """
    viz = get_graph_overlay_img(img, graph, show_edges)

    plt.figure(figsize=(20, 20))
    plt.imshow(viz)
    plt.axis('off')
    plt.show()


# %%
from graph_neural_networks.data.dataset.dynamic.graph_transforms.add_edge_closest_cc import AddEdgeClosestCCTransform

transform = AddEdgeClosestCCTransform()

G_transformed = transform(G)

display_graph_overlay(img, G_transformed)

# %%
