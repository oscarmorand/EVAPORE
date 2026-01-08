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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

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

# %%
from graph.graph_creation import img_to_graph

pred_graph = img_to_graph(pred)
gt_graph = img_to_graph(gt, clean=False)

# %%
from skimage.morphology import skeletonize

skel_g = skeletonize(gt)
plt.figure(figsize=(100,100))
plt.imshow(skel_g)
plt.title("Ground Truth Skeleton")
plt.show()

g_skel_in_p = np.logical_and(skel_g, pred)
plt.figure(figsize=(100,100))
plt.imshow(g_skel_in_p)
plt.title("Ground Truth Skeleton in Prediction")
plt.show()


g_skel_not_in_p = np.logical_and(skel_g, np.logical_not(pred))
plt.figure(figsize=(100,100))
plt.imshow(g_skel_not_in_p)
plt.title("Ground Truth Skeleton not in Prediction")
plt.show()

# %%
reconstructed_skel = g_skel_in_p + g_skel_not_in_p

print(np.allclose(reconstructed_skel, skel_g))

# %%
from scipy.ndimage import label

connectivity_structure = np.array([[1,1,1],
                                   [1,1,1],
                                   [1,1,1]])
labeled_array, num_features = label(g_skel_not_in_p, structure=connectivity_structure)

print(f"Number of disconnected components in GT skeleton not in Prediction: {num_features}")

# %%
plt.figure(figsize=(20,20))
plt.imshow(labeled_array, cmap='nipy_spectral')
plt.title("Labeled Connected Components of Missing Skeleton")
plt.colorbar()
plt.show()

# %%
features_to_keep = []

for i in range(1, num_features + 1):
    component = (labeled_array == i)
    size = np.sum(component)
    if size >= 5:
        features_to_keep.append(i)
new_num_features = len(features_to_keep)

print(f"Features to keep (size >= 5 pixels): {features_to_keep}")
print(f"Number of features to keep: {new_num_features}")

# %%
new_labeled_array = np.zeros_like(labeled_array)
for new_label, old_label in enumerate(features_to_keep, start=1):
    new_labeled_array[labeled_array == old_label] = new_label

plt.figure(figsize=(20,20))
plt.imshow(new_labeled_array, cmap='nipy_spectral')
plt.title("Filtered Labeled Connected Components of Missing Skeleton")
plt.colorbar()
plt.show()

# %%
for i in range(1, new_num_features + 1):
    component = (new_labeled_array == i).astype(np.uint8)
    intersecting_edges = []
    for u, v, data in gt_graph.edges(data=True):
        centerline = np.array(data['centerline'])
        centerline_mask = np.zeros_like(skel_g, dtype=np.uint8)
        centerline_mask[centerline[:,0], centerline[:,1]] = 1
        if (component * centerline_mask).any():
            intersecting_edges.append((u, v))
    #print(f"Component {i} intersects with edges: {intersecting_edges}")

# %%
for u, v, data in gt_graph.edges(data=True):
    centerline = np.array(data['centerline'])
    centerline_mask = np.zeros_like(skel_g, dtype=np.uint8)
    centerline_mask[centerline[:,0], centerline[:,1]] = 1

    intersecting_components = []
    for i in range(1, new_num_features + 1):
        component = (new_labeled_array == i).astype(np.uint8)
        if (component * centerline_mask).any():
            intersecting_components.append(i)

    #print(f"Edge ({u}, {v}) intersects with components: {intersecting_components}")

# %%
component_mask = np.zeros((new_num_features, gt.shape[0], gt.shape[1]), dtype=np.uint8)
for i in range(1, new_num_features + 1):
    component_mask[i-1] = (new_labeled_array == i).astype(np.uint8)

for u, v, data in gt_graph.edges(data=True):
    centerline = np.array(data['centerline'])
    centerline_mask = np.zeros_like(skel_g, dtype=np.uint8)
    centerline_mask[centerline[:,0], centerline[:,1]] = 1

    intersecting_components = []
    for i in range(new_num_features):
        if (component_mask[i] * centerline_mask).any():
            intersecting_components.append(i+1)

    if len(intersecting_components) > 0:
        print(f"Edge ({u}, {v}) intersects with components: {intersecting_components}")

# %%
plt.figure(figsize=(100,100))
plt.imshow(g_skel_in_p)
plt.title("Ground Truth Skeleton in Prediction")
plt.show()

# %%
from skimage.morphology import binary_dilation

footprint = np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]])
g_skel_not_in_p_dilated = np.logical_and(binary_dilation(g_skel_not_in_p, footprint), skel_g)

plt.figure(figsize=(100,100))
plt.imshow(g_skel_not_in_p)
plt.title("Ground Truth Skeleton not in Prediction")
plt.show()

plt.figure(figsize=(100,100))
plt.imshow(g_skel_not_in_p_dilated)
plt.title("Dilated Ground Truth Skeleton not in Prediction")
plt.show()

# %%
g_in_p_graph = img_to_graph(g_skel_in_p, clean=False)
g_not_in_p_graph = img_to_graph(g_skel_not_in_p_dilated, clean=False)

# %%
from graph.graph_visualization import display_graph_overlay

display_graph_overlay(skel_g, g_in_p_graph, figsize=(100,100))
display_graph_overlay(skel_g, g_not_in_p_graph, figsize=(100,100))

# %%
import networkx as nx
from enum import Enum

class EdgePredState(Enum):
    NOT_IN_PREDICTION = 0
    IN_PREDICTION = 1

def combine_graphs(in_p_graph, not_in_p_graph):
    combined_graph = nx.MultiGraph()

    in_p_node_coords = set([tuple(data['pos']) for _, data in in_p_graph.nodes(data=True)])
    not_in_p_node_coords = set([tuple(data['pos']) for _, data in not_in_p_graph.nodes(data=True)])

    combined_node_coords = in_p_node_coords.union(not_in_p_node_coords)

    for combined_node_i, coord in enumerate(combined_node_coords):
        combined_graph.add_node(combined_node_i, pos=coord)

    combined_node_coords_to_node = {tuple(data['pos']): node for node, data in combined_graph.nodes(data=True)}

    source_name_to_attr = {
        "not_in_p_graph": EdgePredState.NOT_IN_PREDICTION,
        "in_p_graph": EdgePredState.IN_PREDICTION
    }

    def add_edges_from_graph(source_graph, source_name): 
        for u, v, data in source_graph.edges(data=True):
            coord_u = tuple(source_graph.nodes[u]['pos'])
            coord_v = tuple(source_graph.nodes[v]['pos'])
            if coord_u not in combined_node_coords_to_node or coord_v not in combined_node_coords_to_node:
                print(f"Skipping edge from {source_name} with unknown nodes:", coord_u, coord_v)
                continue
            combined_u = combined_node_coords_to_node[coord_u]
            combined_v = combined_node_coords_to_node[coord_v]
            combined_graph.add_edge(
                combined_u, 
                combined_v, 
                **data, 
                edge_pred_state=source_name_to_attr[source_name]
            )   

    add_edges_from_graph(in_p_graph, "in_p_graph")
    add_edges_from_graph(not_in_p_graph, "not_in_p_graph")

    return combined_graph


# %%
combined_graph = combine_graphs(g_in_p_graph, g_not_in_p_graph)

# %%
display_graph_overlay(skel_g, combined_graph, figsize=(100,100))


# %%
def display_edge_pred_state(img, graph, figsize=(20,20)):

    viz = np.zeros((*img.shape, 3), dtype=np.uint8)
    viz[img > 0] = [255, 255, 255]

    # Overlay edges in red
    for u, v, data in graph.edges(data=True):
        centerline = np.array(data['centerline'])
        edge_pred_state = data.get('edge_pred_state', None)
        if edge_pred_state == EdgePredState.IN_PREDICTION:
            color = [0, 0, 255]  # Blue for in prediction
        elif edge_pred_state == EdgePredState.NOT_IN_PREDICTION:
            color = [255, 0, 0]  # Red for not in prediction
        else:
            color = [255, 255, 0]  # Yellow for unknown state
        viz[centerline[:,0], centerline[:,1]] = color

    # Overlay nodes in green
    for n, data in graph.nodes(data=True):
        pos = data['pos']
        viz[pos[0], pos[1]] = [0, 255, 0]

    plt.figure(figsize=figsize)
    plt.imshow(viz)
    plt.title("Edge Prediction State Visualization")
    plt.show()


# %%
display_edge_pred_state(skel_g, combined_graph, figsize=(100,100))

# %%
for pred_path, gt_path in zip(pred_paths, gt_paths):
    print(gt_path)
    pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))
    gt = np.array(Image.open(gt_path))[:,:,0]

    gt_graph = img_to_graph(gt, clean=True)

    skel_g = np.zeros_like(gt, dtype=bool)
    for u, v, data in gt_graph.edges(data=True):
        centerline = np.array(data['centerline'])
        skel_g[centerline[:,0], centerline[:,1]] = True
    g_skel_in_p = np.logical_and(skel_g, pred)
    g_skel_not_in_p = np.logical_and(skel_g, np.logical_not(pred))
    g_skel_not_in_p_dilated = np.logical_and(binary_dilation(g_skel_not_in_p, footprint), skel_g)

    g_in_p_graph = img_to_graph(g_skel_in_p, clean=False)
    g_not_in_p_graph = img_to_graph(g_skel_not_in_p_dilated, clean=False)

    combined_graph = combine_graphs(g_in_p_graph, g_not_in_p_graph)

    #display_edge_pred_state(skel_g, combined_graph, figsize=(100,100))

# %%
import networkx as nx
import numpy as np
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper

class OversampleNodesTransform:
    def __init__(self,
                 max_dist: float = 10.0,
                 remove_original_edges: bool = True):
        self.max_dist = max_dist
        self.remove_original_edges = remove_original_edges

    def oversample_graph(self, graph: nx.Graph) -> nx.Graph:
        # setup edge and node counters
        max_node_id = max(graph.nodes) + 1
        edge_counter = max([data['id'] for _, _, data in graph.edges(data=True)]) + 1

        old_nodes = list(graph.nodes(data=False))
        new_nodes = []
        new_nodes_parent_distances = {}

        G = graph.copy()
        for u, v in graph.edges(data=False):
            new_edge_sampled = False
            for _, data in graph.get_edge_data(u, v).items():
                edge_pred_state = data.get('edge_pred_state', None)
                if edge_pred_state == EdgePredState.NOT_IN_PREDICTION:
                    continue
                centerline = np.array(data['centerline'])
                radius = np.array(data['radius'])
                length = data['length']
                if length < self.max_dist:
                    continue

                n = int(length // self.max_dist) + 1
                if n < 2:
                    continue
                l = length / n

                acc_length = 0.0
                switch = False
                split_points = []
                lengths = []
                indexes = []
                for i in range(len(centerline) - 1):
                    if len(split_points) >= n - 1:
                        break
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
                    new_nodes.append(node_id)
                    first_part_length = np.sum(lengths[:i+1])
                    second_part_length = length - first_part_length
                    new_nodes_parent_distances[node_id] = [(u, first_part_length), (v, second_part_length)]

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
                        edge_pred_state=edge_pred_state
                    )
                    edge_counter += 1
                    new_edge_sampled = True

            if self.remove_original_edges and new_edge_sampled:
                G.remove_edge(u, v)
                
        return G, old_nodes, new_nodes, new_nodes_parent_distances
    
    def oversample_distance_matrix(self,
                                   distance_matrix: np.ndarray,
                                   oversampled_graph: nx.Graph,
                                   old_nodes: list[int],
                                   new_nodes: list[int],
                                   new_nodes_parent_distances: dict):

        N = distance_matrix.shape[0]
        new_N = oversampled_graph.number_of_nodes()
        new_distance_matrix = np.ones((new_N, new_N)) * -1
        new_distance_matrix[0:N, 0:N] = distance_matrix
        oti = {n: i for i, n in enumerate(old_nodes)}

        def set_distance_between_new_and_old(mat, N, new_node, old_node, val):
            mat[N + new_node, old_node] = val
            mat[old_node, N + new_node] = val
        def set_distance_between_new_nodes(mat, N, new_node1, new_node2, val):
            mat[N + new_node1, N + new_node2] = val
            mat[N + new_node2, N + new_node1] = val
            
        for i, new_node in enumerate(new_nodes):
            (p1, d1), (p2, d2) = new_nodes_parent_distances[new_node]
            for j, old_node in enumerate(old_nodes):
                if old_node == p1:
                    set_distance_between_new_and_old(new_distance_matrix, N, i, j, d1)
                elif old_node == p2:
                    set_distance_between_new_and_old(new_distance_matrix, N, i, j, d2)
                else:
                    closest_parent = p1
                    dist_to_closest_parent = d1
                    p1_to_old = distance_matrix[oti[old_node], oti[p1]]
                    p2_to_old = distance_matrix[oti[old_node], oti[p2]]
                    if p1_to_old == -1 and p2_to_old == -1:
                        set_distance_between_new_and_old(new_distance_matrix, N, i, j, -1)
                        continue
                    if (p2_to_old + d2 < p1_to_old + d1) or p1_to_old == -1:
                        closest_parent = p2
                        dist_to_closest_parent = d2
                    set_distance_between_new_and_old(new_distance_matrix, N, i, j, dist_to_closest_parent + distance_matrix[oti[old_node], oti[closest_parent]])
                    new_distance_matrix[j, N + i] = new_distance_matrix[N + i, j]

        for i, new_node in enumerate(new_nodes):
            (p1, d1), (p2, d2) = new_nodes_parent_distances[new_node]
            for j in range(i, len(new_nodes)):
                other_new_node = new_nodes[j]
                if i == j:
                    set_distance_between_new_nodes(new_distance_matrix, N, i, j, 0.0)
                    continue
                (other_p1, other_d1), (other_p2, other_d2) = new_nodes_parent_distances[other_new_node]
                if p1 == other_p1 and p2 == other_p2:
                    inter_dist = abs(other_d1 - d1)
                    set_distance_between_new_nodes(new_distance_matrix, N, i, j, inter_dist)
                    continue
                other_new_to_p1 = new_distance_matrix[N + j, oti[p1]]
                other_new_to_p2 = new_distance_matrix[N + j, oti[p2]]
                if other_new_to_p1 == -1 and other_new_to_p2 == -1:
                    set_distance_between_new_nodes(new_distance_matrix, N, i, j, -1)
                    continue
                closest_parent = p1
                dist_to_closest_parent = d1
                if (other_new_to_p2 + d2 < other_new_to_p1 + d1) or other_new_to_p1 == -1:
                    closest_parent = p2
                    dist_to_closest_parent = d2
                set_distance_between_new_nodes(new_distance_matrix, N, i, j, dist_to_closest_parent + new_distance_matrix[N + j, oti[closest_parent]])
        
        return new_distance_matrix


    def __call__(self, graph_wrapper: GraphWrapper) -> GraphWrapper:
        graph = graph_wrapper.get_graph()

        oversampled_graph, old_nodes, new_nodes, new_nodes_parent_distances = self.oversample_graph(graph)
        graph_wrapper.old_nodes = old_nodes
        graph_wrapper.new_nodes = new_nodes
        graph_wrapper.new_nodes_parent_distances = new_nodes_parent_distances

        graph_wrapper.is_oversampled = True
        if graph_wrapper.distance_matrix is not None:
            oversampled_distance_matrix = self.oversample_distance_matrix(
                graph_wrapper.distance_matrix,
                oversampled_graph,
                old_nodes,
                new_nodes,
                new_nodes_parent_distances
            )
            graph_wrapper.oversampled_distance_matrix = oversampled_distance_matrix

        graph_wrapper.set_graph(oversampled_graph)
        return graph_wrapper


    def _build_config(self) -> dict:
        return {
            "_target_": self.__class__.__name__,
            "max_dist": self.max_dist,
            "remove_original_edges": self.remove_original_edges,
        }

# %%

from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper

# %%
pred_path = pred_paths[0]
gt_path = gt_paths[0]

# %%
pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))
gt = np.array(Image.open(gt_path))[:,:,0]

gt_graph = img_to_graph(gt, clean=True)

skel_g = np.zeros_like(gt, dtype=bool)
for u, v, data in gt_graph.edges(data=True):
    centerline = np.array(data['centerline'])
    skel_g[centerline[:,0], centerline[:,1]] = True
g_skel_in_p = np.logical_and(skel_g, pred)
g_skel_not_in_p = np.logical_and(skel_g, np.logical_not(pred))
g_skel_not_in_p_dilated = np.logical_and(binary_dilation(g_skel_not_in_p, footprint), skel_g)

g_in_p_graph = img_to_graph(g_skel_in_p, clean=False)
display_graph_overlay(skel_g, g_in_p_graph, figsize=(100,100))
g_not_in_p_graph = img_to_graph(g_skel_not_in_p_dilated, clean=False)
display_graph_overlay(skel_g, g_not_in_p_graph, figsize=(100,100))

combined_graph = combine_graphs(g_in_p_graph, g_not_in_p_graph)

display_edge_pred_state(skel_g, combined_graph, figsize=(100,100))

# %%
graph_wrapper = GraphWrapper(combined_graph)

transform = OversampleNodesTransform(max_dist=50, remove_original_edges=True)

transformed_graph_wrapper = transform(graph_wrapper)

transformed_graph = transformed_graph_wrapper.graph

display_edge_pred_state(skel_g, transformed_graph, figsize=(100,100))

# %%
