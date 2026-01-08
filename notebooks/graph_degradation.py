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
#     display_name: test_env
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
#img = np.array(Image.open("/home/morand/afs/datasets/FIVES/train/Ground truth/1_A.png"))[:,:,0]
img = np.array(Image.open("/home/morand/afs/tests/New Piskel.png"))[:,:,0]

print(img.shape)

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# %%
from graph.graph_creation import img_to_graph

graph, pixel_graph = img_to_graph(img, clean=True, closing_radius=0, return_pixel_graph=True)

# %%
from graph.graph_visualization import display_graph_overlay

display_graph_overlay(img, graph)

from graph.graph_io import save_graph_to_dot
save_graph_to_dot(graph, "/home/morand/afs/tests/New Piskel.dot")

save_graph_to_dot(pixel_graph, "/home/morand/afs/tests/New Piskel Pixel.dot")


# %%
def get_parallel_edges(graph: nx.MultiGraph) -> set:
    """
    Get all edges connected to a node that have parallel edges.

    Args:
        graph (nx.MultiGraph): The input graph.
        node (int): The node to check for parallel edges.

    Returns:
        set: A set of edges that have parallel edges.
    """
    parallel_edges = set()
    for n in graph.nodes():
        for m in graph.neighbors(n):
            n_edges = graph.number_of_edges(n, m)
            if n_edges > 1:
                if (n, m) not in parallel_edges and (m, n) not in parallel_edges:
                    parallel_edges.add((n,m))
    return parallel_edges


# %%
print(get_parallel_edges(graph))


# %%
def plot_graph_difference(img: np.ndarray,
                          graph1: nx.Graph,
                          graph2: nx.Graph,
                          bounding_box_width: int = 5
                          ) -> None:
    """
    Display the differences between two graphs overlayed on the original image.

    Args:
        img (np.ndarray): Original binary image.
        graph1 (nx.Graph): First vascular graph.
        graph2 (nx.Graph): Second vascular graph.
    """
    viz = np.zeros((*img.shape, 3), dtype=np.uint8)
    viz[img > 0] = [255, 255, 255]

    # Edges in graph1 but not in graph2 in red
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    diff_edges = edges1.symmetric_difference(edges2)
    
    parallel_1 = get_parallel_edges(graph1)
    parallel_2 = get_parallel_edges(graph2)
    diff_parallel = parallel_1.symmetric_difference(parallel_2)

    diff_edges = diff_edges.union(diff_parallel)

    for u, v in diff_edges:
        data = {}
        if (u, v) in edges1 and (u, v) in edges2:
            keys_1, keys_2 = set(graph1.get_edge_data(u, v).keys()), set(graph2.get_edge_data(u, v).keys())
            for e in keys_1.union(keys_2):
                if e in keys_1:
                    if e in keys_2:
                        continue
                    data[e] = graph1.get_edge_data(u, v)[e]
                else:
                    data[e] = graph2.get_edge_data(u, v)[e]
        elif (u, v) in edges1:
            data = graph1.get_edge_data(u, v)
        else:
            data = graph2.get_edge_data(u, v)

        for e, val in data.items():
            centerline = np.array(val['centerline'])
            viz[centerline[:,0], centerline[:,1]] = [255, 0, 0]

            # Highlight bounding box of differences
            all_coords = np.array(val['centerline'])
            min_row, min_col = np.min(all_coords, axis=0) + (-bounding_box_width, -bounding_box_width) + (-1, -1)
            max_row, max_col = np.max(all_coords, axis=0) + (bounding_box_width, bounding_box_width) + (1, 1)
            viz[min_row:max_row+1, min_col - bounding_box_width:min_col + bounding_box_width] = [0, 255, 255]
            viz[min_row:max_row+1, max_col - bounding_box_width:max_col + bounding_box_width] = [0, 255, 255]
            viz[min_row - bounding_box_width:min_row + bounding_box_width, min_col:max_col+1] = [0, 255, 255]
            viz[max_row - bounding_box_width:max_row + bounding_box_width, min_col:max_col+1] = [0, 255, 255]

    plt.figure(figsize=(8, 8))
    plt.imshow(viz)
    plt.axis('off')
    plt.show()


# %%
def display_edges_set(graph: nx.MultiGraph, edges: set, img: np.ndarray) -> None:
    background = np.zeros((*img.shape, 3), dtype=np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(background, cmap='gray')

    for u, v in edges:
        if graph.has_edge(u, v):
            data = graph.get_edge_data(u, v)[0]
            centerline = np.array(data['centerline'])
            plt.plot(centerline[:,1], centerline[:,0], color='blue', linewidth=2)

    plt.axis('off')
    plt.show()


# %%
from graph.graph_labeling import generate_graph_topological_classification
from graph.graph_visualization import display_topological_edge_classification

# %%
display_topological_edge_classification(graph, img)

# %%
from graph.graph_utils import TopologicalClass
from graph.graph_labeling import generate_graph_depth
from graph.graph_random_edge_selection import uniform_edge_selection

def depth_edge_selection(graph: nx.Graph,
                         depth: int
                         ) -> tuple:
    if not any('depth' in d for _, _, d in graph.edges(data=True)):
        logging.warning("No edges have 'depth' attribute, computing graph depth...")
        graph = generate_graph_depth(graph)

    # Filter edges by depth level
    all_edges = [(u, v) for u, v, d in graph.edges(data=True)]
    edges_id_in_level = [i for i, (u, v, d) in enumerate(graph.edges(data=True)) if d.get('depth') == depth]

    if not edges_id_in_level:
        logging.warning(f"No edges found at depth level {depth}.")
        return graph

    # Randomly select an edge to remove
    edge_id = np.random.choice(edges_id_in_level)
    selected_edge = all_edges[edge_id]

    return selected_edge

def topological_edge_selection(graph: nx.MultiGraph, topological_class: TopologicalClass) -> tuple:
    if not any('topological_class' in d for _, _, d in graph.edges(data=True)):
        logging.warning("No edges have 'topological_class' attribute, computing classification...")
        graph = generate_graph_topological_classification(graph)
    
    # Filter edges by topological class
    all_edges = [(u, v) for u, v, d in graph.edges(data=True)]
    edges_id_in_class = [i for i, (u, v, d) in enumerate(graph.edges(data=True)) if d.get('topological_class') == topological_class]

    # Randomly select an edge to remove
    edge_id = np.random.choice(edges_id_in_class)
    selected_edge = all_edges[edge_id]

    return selected_edge


# %%
from graph.graph_degradation import EdgeDegradationMode, EdgeSelectionMode, edge_degradation 

# %%
graph = generate_graph_topological_classification(graph)
g_degraded = edge_degradation(graph, EdgeSelectionMode.UNIFORM, EdgeDegradationMode.DELETE)

display_topological_edge_classification(graph, img)
display_topological_edge_classification(g_degraded, img)
plot_graph_difference(img, graph, g_degraded, bounding_box_width=0)

# %%
n = graph.number_of_edges()

for i in range(n):
    print(f"Iteration {i+1}/{n}")

    display_topological_edge_classification(graph, img)

    if graph.number_of_edges() == 0 or graph.number_of_nodes() == 0:
        break

    g_degraded = edge_degradation(graph, EdgeSelectionMode.UNIFORM, EdgeDegradationMode.DELETE)
    
    graph = g_degraded

# %%
