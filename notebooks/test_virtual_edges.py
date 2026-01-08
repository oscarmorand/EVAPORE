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
from graph_neural_networks.data.utils.virtual_edges import add_virtual_edge, setup_non_virtual_edges, get_virtual_edges_index

# %%
img = np.array(Image.open("/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds/1_Aweight_best_iou.png"))
#img = np.array(Image.open("/home/morand/afs/tests/New Piskel.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/simple_loop_two.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/error_loop_branch.png"))[:,:,0]

print(img.shape)

plt.imshow(img, cmap='gray')
plt.show()

# %%
from torch_geometric.utils.convert import from_networkx
from graph.graph_creation import img_to_graph

G = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)
print(G)


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
display_graph_overlay(img, G, show_edges=True)

# %%
print(G.nodes())

# %%
print(G.edges)
for u, v, c in G.edges:
    if c != 0:
        print(u, v, c)

# %%
edges_data = G.edges(data=True)
edges_idx = [edge['id'] for (_, _, edge) in edges_data]

print("Edges ids:", edges_idx)

test = set(edges_idx)
print("Number of edges:", len(test))

# %%
import networkx as nx
import numpy as np

from graph_neural_networks.data.dataset.dynamic.graph_transforms.graph_transform import GraphTransform
from graph_neural_networks.data.utils.virtual_edges import add_virtual_edge, setup_non_virtual_edges

class AddDistanceCCEdgeTransform(GraphTransform):
    def __init__(self, 
                 distance_threshold: float = 20.0,
                 cc_condition: bool = True,
                 only_connect_extremities: bool = False,
                 only_keep_closest: bool = False,
                 distance_att: str = "euclidean"):
        """
        Graph transform that adds edges between nodes that are within a certain distance threshold
        and optionally belong to the same connected component (CC).

        Args:
            distance_threshold (float): The maximum distance between nodes to add an edge.
            cc_condition (bool): If True, only add edges between nodes in the same connected component.
        """
        self.distance_threshold = distance_threshold
        self.cc_condition = cc_condition
        self.only_connect_extremities = only_connect_extremities
        self.only_keep_closest = only_keep_closest
        self.distance_att = distance_att

    def get_distance_att(self, distance: float) -> float:
        if self.distance_att == "euclidean":
            return distance
        elif isinstance(self.distance_att, float):
            return self.distance_att
        else:
            return None

    def __call__(self, graph: nx.Graph) -> nx.Graph:
        """
        Apply the transform to the given graph.

        Args:
            graph: The input graph object.

        Returns:
            The transformed graph with additional edges.
        """

        # Classify the connected components if cc_condition is True
        if self.cc_condition:
            connected_components = list(nx.connected_components(graph))
            cc_map = {}
            for cc in connected_components:
                for node in cc:
                    cc_map[node] = cc

        graph = setup_non_virtual_edges(graph)

        G = graph.copy()

        # Add edges based on distance and connected component condition
        for (node1_id, node1_data) in graph.nodes(data=True):
            if self.only_connect_extremities and graph.degree(node1_id) > 1:
                continue
            closest_distance = float('inf')
            closest_node_id = None
            for (node2_id, node2_data) in graph.nodes(data=True):
                if node1_id == node2_id:
                    continue

                if self.only_connect_extremities and graph.degree(node2_id) > 1:
                    continue

                pos1 = np.array(node1_data.get('pos', None))
                pos2 = np.array(node2_data.get('pos', None))

                if pos1 is None or pos2 is None:
                    continue

                distance = np.linalg.norm(pos1 - pos2)

                # Add edge if within distance threshold
                if distance <= self.distance_threshold:
                    if not self.cc_condition or (self.cc_condition and cc_map[node1_id] != cc_map[node2_id]):
                        if self.only_keep_closest:
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_node_id = node2_id
                        else:
                            distance_att = self.get_distance_att(distance)
                            G = add_virtual_edge(G, node1_id, node2_id, length=distance_att)
            
            if self.only_keep_closest and closest_node_id is not None:
                distance_att = self.get_distance_att(closest_distance)
                G = add_virtual_edge(G, node1_id, closest_node_id, length=distance_att)

        return G

    def _build_config(self) -> dict:
        return {
            "_target_": self.__class__.__name__,
            "distance_threshold": self.distance_threshold,
            "cc_condition": self.cc_condition,
            "only_connect_extremities": self.only_connect_extremities,
            "distance_att": self.distance_att,
        }


# %%
transform = AddDistanceCCEdgeTransform(
    distance_threshold=100,
    cc_condition=False,
    only_connect_extremities=True,
    distance_att=None
)

G_transformed = transform(G)

display_graph_overlay(img, G_transformed, show_edges=True)


# %%
transform = AddDistanceCCEdgeTransform(
    distance_threshold=100,
    cc_condition=True,
    only_connect_extremities=True,
    distance_att=None
)

G_transformed = transform(G)

display_graph_overlay(img, G_transformed, show_edges=True)

# %%
transform = AddDistanceCCEdgeTransform(
    distance_threshold=100,
    cc_condition=True,
    only_connect_extremities=False,
    distance_att=None
)

G_transformed = transform(G)

display_graph_overlay(img, G_transformed, show_edges=True)

# %%
transform = AddDistanceCCEdgeTransform(
    distance_threshold=100,
    cc_condition=False,
    only_connect_extremities=False,
    distance_att=None
)

G_transformed = transform(G)

display_graph_overlay(img, G_transformed, show_edges=True)

# %%
transform = AddDistanceCCEdgeTransform(
    distance_threshold=50,
    cc_condition=True,
    only_connect_extremities=False,
    only_keep_closest=True,
    distance_att=None
)

G_transformed = transform(G)

display_graph_overlay(img, G_transformed, show_edges=True)


# %%
class AddEdgeClosestCCTransform(GraphTransform):
    def __init__(self, only_one_edge_per_cc: bool = True):
        self.only_one_edge_per_cc = only_one_edge_per_cc

    def __call__(self, graph: nx.Graph) -> nx.Graph:
        """
        Apply the transform to the given graph.

        Args:
            graph: The input graph object.

        Returns:
            The transformed graph with additional edges.
        """

        # Classify the connected components if cc_condition is True
        connected_components = list(nx.connected_components(graph))
        G = graph.copy()

        while len(connected_components) > 1:
            cc_map = {}
            for cc in connected_components:
                for node in cc:
                    cc_map[node] = cc

            cc_n_count = [len(cc) for cc in connected_components]
            small_ccs = [cc for cc in connected_components if len(cc) < max(cc_n_count)]

            for small_cc in small_ccs:
                global_closest_distance = float('inf')
                global_closest_node1_id = None
                global_closest_node2_id = None
                for node1_id in small_cc:
                    closest_distance = float('inf')
                    closest_node_id = None

                    other_nodes = []
                    for cc in connected_components:
                        if cc != cc_map[node1_id]:
                            other_nodes.extend(list(cc))
                    for node2_id in other_nodes:
                        if node1_id == node2_id:
                            continue

                        pos1 = np.array(graph.nodes[node1_id].get('pos', None))
                        pos2 = np.array(graph.nodes[node2_id].get('pos', None))

                        if pos1 is None or pos2 is None:
                            continue

                        distance = np.linalg.norm(pos1 - pos2)

                        # Add edge if within distance threshold
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_node_id = node2_id

                    distance_att = closest_distance
                    if self.only_one_edge_per_cc:
                        if closest_distance < global_closest_distance:
                            global_closest_distance = closest_distance
                            global_closest_node1_id = node1_id
                            global_closest_node2_id = closest_node_id
                    else:
                        G = add_virtual_edge(G, node1_id, closest_node_id, length=distance_att)

                if self.only_one_edge_per_cc:
                    G = add_virtual_edge(G, global_closest_node1_id, global_closest_node2_id, length=global_closest_distance)
        
            connected_components = list(nx.connected_components(G))

        return G

    def _build_config(self) -> dict:
        return {
            "_target_": self.__class__.__name__,
        }


# %%
transform = AddEdgeClosestCCTransform(only_one_edge_per_cc=False)

G_transformed = transform(G)

display_graph_overlay(img, G_transformed, show_edges=True)

# %%
transform = AddEdgeClosestCCTransform(only_one_edge_per_cc=True)

G_transformed = transform(G)

display_graph_overlay(img, G_transformed, show_edges=True)

# %%
