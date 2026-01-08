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
from graph.graph_creation import img_to_graph

graph = img_to_graph(img)

print(graph)

# %%
from graph_neural_networks.data.dataset.dynamic.graph_transforms.compute_distance_matrix import ComputeDistanceMatrixTransform
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
from graph_neural_networks.data.dataset.dynamic.graph_transforms.oversample_nodes import OversampleNodesTransform


graph_wrapper = GraphWrapper(graph)

transform = ComputeDistanceMatrixTransform()
oversample_transform = OversampleNodesTransform(max_dist=20)

transformed_graph_wrapper = transform(graph_wrapper)
transformed_graph_wrapper = oversample_transform(transformed_graph_wrapper)

geodesic_distance_matrix = transformed_graph_wrapper.oversampled_distance_matrix
#geodesic_distance_matrix = transformed_graph_wrapper.distance_matrix

# %%
plt.imshow(geodesic_distance_matrix, interpolation='nearest')
plt.colorbar()
plt.show()

# %%
graph = transformed_graph_wrapper.graph

# %%
euclidean_distance_matrix = np.zeros_like(geodesic_distance_matrix)

idx_to_node = {idx: n for idx, n in enumerate(graph.nodes(data=True))}

for idx_0, n_0 in enumerate(graph.nodes(data=True)):
    for idx_1, n_1 in enumerate(graph.nodes(data=True)):
        coord_0 = n_0[1]['pos']
        coord_1 = n_1[1]['pos']
        euclidean_distance_matrix[idx_0, idx_1] = np.linalg.norm(np.array(coord_0) - np.array(coord_1))

# %%
plt.imshow(euclidean_distance_matrix, interpolation='nearest')
plt.colorbar()
plt.show()

# %%
pos_euclid = euclidean_distance_matrix > 0

test = geodesic_distance_matrix.copy()
test[pos_euclid] = test[pos_euclid] / euclidean_distance_matrix[pos_euclid]

plt.imshow(test, interpolation='nearest')
plt.colorbar()
plt.show()

# %%
max_id = np.unravel_index(np.argmax(test, axis=None), test.shape)
x, y = max_id
print("Max ratio at indices:", max_id)
print("Max ratio value:", test[y, x])
print("Geodesic distance at max ratio:", geodesic_distance_matrix[y, x])
print("euclidean distance at max ratio:", euclidean_distance_matrix[y, x])
print(f"node_0: {idx_to_node[x][0]}")
print(f"node_1: {idx_to_node[y][0]}")

# %%
plt.figure(figsize=(40,40))
plt.imshow(img, cmap='gray')
node_0 = idx_to_node[max_id[0]]
node_1 = idx_to_node[max_id[1]]
node_0_y, node_0_x = node_0[1]['pos']
node_1_y, node_1_x = node_1[1]['pos']

plt.plot([node_0_x, node_1_x], [node_0_y, node_1_y], color='red')

plt.show()


# %%
plt.figure(figsize=(40,40))
plt.imshow(img, cmap='gray')

for i in range(test.shape[0]):
    for j in range(i+1, test.shape[1]):
        if test[i, j] > 10:
            node_a = idx_to_node[i]
            node_b = idx_to_node[j]
            print(f"High ratio between node {i} at {node_a[1]['pos']} and node {j} at {node_b[1]['pos']}: ratio={test[i,j]}, geodesic={geodesic_distance_matrix[i,j]}, euclidean={euclidean_distance_matrix[i,j]}")

            y_a, x_a = node_a[1]['pos']
            y_b, x_b = node_b[1]['pos']

            color_ratio = (1.0, 0.0, 0.0, min(1.0, test[i,j]/50.0))

            plt.plot([x_a, x_b], [y_a, y_b], color=color_ratio)

plt.show()

# %%
test_masked = test * np.tril(np.ones_like(test), k=-1)
plt.imshow(test_masked, interpolation='nearest')
plt.colorbar()
plt.show()
sorted_x, sorted_y = np.unravel_index(np.argsort(test_masked.flatten())[::-1], test_masked.shape)

sorted_nodes_couples = [(idx_to_node[i], idx_to_node[j]) for i, j in zip(sorted_y, sorted_x)]

print(sorted_y)
print(len(sorted_nodes_couples))
print(test_masked.shape[0] * test_masked.shape[1] )

# %%
max_indices_top_20 = sorted_nodes_couples[:20]
print(max_indices_top_20)

# %%
plt.figure(figsize=(40,40))
plt.imshow(img, cmap='gray')
for top_i, (node_0, node_1) in enumerate(max_indices_top_20):
    node_0_y, node_0_x = node_0[1]['pos']
    node_1_y, node_1_x = node_1[1]['pos']

    colormap = plt.get_cmap("hot")
    ratio_value = colormap(top_i / len(max_indices_top_20))

    plt.plot([node_0_x, node_1_x], [node_0_y, node_1_y], color=ratio_value)
    
plt.show()


# %%
max_ratio_id_for_each_node = np.argmax(test, axis=0)
print(max_ratio_id_for_each_node.shape)

# %%
max_ratio_node_for_each_node = [(idx_to_node[i], idx_to_node[j]) for j, i in enumerate(max_ratio_id_for_each_node)]


# %%
plt.figure(figsize=(40,40))
plt.imshow(img, cmap='gray')
for node_0, node_1 in max_ratio_node_for_each_node:
    node_0_y, node_0_x = node_0[1]['pos']
    node_1_y, node_1_x = node_1[1]['pos']

    plt.plot([node_0_x, node_1_x], [node_0_y, node_1_y], color='red')

plt.show()

# %%
reciprocal_max_ratio_idx = []
for j, i in enumerate(max_ratio_id_for_each_node):
    if max_ratio_id_for_each_node[i] == j:
        reciprocal_max_ratio_idx.append((i, j))

print(reciprocal_max_ratio_idx)

reciprocal_max_ratio_nodes = [(idx_to_node[i], idx_to_node[j]) for i, j in reciprocal_max_ratio_idx]

# %%
plt.figure(figsize=(40,40))
plt.imshow(img, cmap='gray')

for node_0, node_1 in reciprocal_max_ratio_nodes:
    node_0_y, node_0_x = node_0[1]['pos']
    node_1_y, node_1_x = node_1[1]['pos']

    plt.plot([node_0_x, node_1_x], [node_0_y, node_1_y], color='red')

plt.show()

# %%
old_nodes_id = [i for i, (n, d) in enumerate(graph.nodes(data=True)) if not d.get('oversampled', False)]
print(old_nodes_id)

# %%
aaa = np.array(max_ratio_node_for_each_node)[old_nodes_id]
print(aaa)

# %%
plt.figure(figsize=(40,40))
plt.imshow(img, cmap='gray')

for n in old_nodes_id:
    max_r = aaa[n]
    node_0 = aaa[n][0]
    node_1 = aaa[n][1]

    node_0_y, node_0_x = node_0[1]['pos']
    node_1_y, node_1_x = node_1[1]['pos']

    plt.plot([node_0_x, node_1_x], [node_0_y, node_1_y], color='red')

plt.show()

# %%
import networkx as nx
from graph_neural_networks.data.utils.pred_state import EdgePredState
import torch
import matplotlib.pyplot as plt

def happend_edge_index_undirected(index_list: list[list[int]], u: int, v: int) -> None:
    index_list.append([u, v])
    index_list.append([v, u])  # undirected graph

def get_edges_index(nx_graph: nx.Graph):
    message_passing_edges_index = []
    gt_edges_index = []
    virtual_edges_index = []
    in_pred_edges_index = []
    not_in_pred_edges_index = []
    visited_edges = set()
    for u, v in nx_graph.edges(data=False):
        if (u, v) in visited_edges:
                continue
        visited_edges.add((u, v))
        for e, data in nx_graph.get_edge_data(u, v).items():
            pred_state = data.get('edge_pred_state', None)
            virtual_edge = data.get('virtual_edge', False)
            if virtual_edge:
                happend_edge_index_undirected(message_passing_edges_index, u, v)
                happend_edge_index_undirected(virtual_edges_index, u, v)
            else:
                if pred_state is None or pred_state in [EdgePredState.IN_PREDICTION, EdgePredState.IN_PREDICTION.value]:
                    happend_edge_index_undirected(message_passing_edges_index, u, v)
                    happend_edge_index_undirected(in_pred_edges_index, u, v)
                elif pred_state in [EdgePredState.NOT_IN_PREDICTION, EdgePredState.NOT_IN_PREDICTION.value]:
                    happend_edge_index_undirected(not_in_pred_edges_index, u, v)
                happend_edge_index_undirected(gt_edges_index, u, v)

    message_passing_edges_index_tensor = torch.tensor(message_passing_edges_index, dtype=torch.long).t().contiguous()
    gt_edges_index_tensor = torch.tensor(gt_edges_index, dtype=torch.long).t().contiguous()
    virtual_edges_index_tensor = torch.tensor(virtual_edges_index, dtype=torch.long).t().contiguous()
    in_pred_edges_index_tensor = torch.tensor(in_pred_edges_index, dtype=torch.long).t().contiguous()
    not_in_pred_edges_index_tensor = torch.tensor(not_in_pred_edges_index, dtype=torch.long).t().contiguous()

    return message_passing_edges_index_tensor, gt_edges_index_tensor, virtual_edges_index_tensor, in_pred_edges_index_tensor, not_in_pred_edges_index_tensor


# %%
from enum import Enum
import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation

from graph.graph_creation import img_to_graph

source_name_to_attr = {
    "not_in_p_graph": EdgePredState.NOT_IN_PREDICTION,
    "in_p_graph": EdgePredState.IN_PREDICTION
}

def add_edges_from_graph(combined_graph, source_graph, source_name, combined_node_coords_to_node): 
    G = combined_graph.copy()
    for u, v, data in source_graph.edges(data=True):
        coord_u = tuple(source_graph.nodes[u]['pos'])
        coord_v = tuple(source_graph.nodes[v]['pos'])
        if coord_u not in combined_node_coords_to_node or coord_v not in combined_node_coords_to_node:
            print(f"Skipping edge from {source_name} with unknown nodes:", coord_u, coord_v)
            continue
        combined_u = combined_node_coords_to_node[coord_u]
        combined_v = combined_node_coords_to_node[coord_v]
        G.add_edge(
            combined_u, 
            combined_v, 
            **data, 
            edge_pred_state=source_name_to_attr[source_name]
        )  
    return G

def combine_graphs(in_p_graph, not_in_p_graph, radius_map):
    combined_graph = nx.MultiGraph()

    in_p_node_coords = set([tuple(data['pos']) for _, data in in_p_graph.nodes(data=True)])
    not_in_p_node_coords = set([tuple(data['pos']) for _, data in not_in_p_graph.nodes(data=True)])

    combined_node_coords = in_p_node_coords.union(not_in_p_node_coords)

    for combined_node_i, coord in enumerate(combined_node_coords):
        radius = radius_map[coord[0], coord[1]]
        combined_graph.add_node(combined_node_i, pos=coord, radius=radius)

    combined_node_coords_to_node = {tuple(data['pos']): node for node, data in combined_graph.nodes(data=True)}

    combined_graph = add_edges_from_graph(combined_graph, in_p_graph, "in_p_graph", combined_node_coords_to_node)
    combined_graph = add_edges_from_graph(combined_graph, not_in_p_graph, "not_in_p_graph", combined_node_coords_to_node)

    return combined_graph


def get_combined_graph(gt, pred):
    gt_graph = img_to_graph(gt, clean=True, closing_radius=1)
    radius_map = np.zeros_like(gt, dtype=np.float32)
    skel_g = np.zeros_like(gt, dtype=bool)

    for u, v, data in gt_graph.edges(data=True):
        centerline = np.array(data['centerline'])
        radius = np.array(data['radius'])
        skel_g[centerline[:,0], centerline[:,1]] = True
        radius_map[centerline[:,0], centerline[:,1]] = radius

    g_skel_in_p = np.logical_and(skel_g, pred)
    g_skel_not_in_p = np.logical_and(skel_g, np.logical_not(pred))

    footprint = np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]])
    g_skel_not_in_p_dilated = np.logical_and(binary_dilation(g_skel_not_in_p, footprint), skel_g)

    g_in_p_graph = img_to_graph(g_skel_in_p, clean=False)
    g_not_in_p_graph = img_to_graph(g_skel_not_in_p_dilated, clean=False)

    combined_graph = combine_graphs(g_in_p_graph, g_not_in_p_graph, radius_map)
    return combined_graph



# %%
#from graph_neural_networks.data.utils.pred_state import get_combined_graph
import os
from PIL import Image

pred_dir = "/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds"
gt_dir = "/home/morand/afs/EVAPORE/data/FIVES/gt"

pred_filenames = sorted(os.listdir(pred_dir))
gt_filenames = sorted(os.listdir(gt_dir))

pred_filenames = [f for f in pred_filenames if f.endswith('.png')]

pred_paths = [os.path.join(pred_dir, f) for f in pred_filenames]
gt_paths = [os.path.join(gt_dir, f) for f in gt_filenames]

pred_path, gt_path = pred_paths[0], gt_paths[0]

gt = np.array(Image.open(gt_path))[:,:,0]
print(gt.shape[1::-1])
pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1], resample=Image.NEAREST))

graph = get_combined_graph(gt, pred)

print(graph)

# %%
from graph.graph_visualization import display_graph_overlay
from skimage.morphology import skeletonize

black = np.zeros_like(gt, dtype=bool)

display_graph_overlay(black, graph, figsize=(100,100))

# %%
from graph_neural_networks.data.dataset.dynamic.graph_transforms.compute_distance_matrix import ComputeDistanceMatrixTransform
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
from graph_neural_networks.data.dataset.dynamic.graph_transforms.oversample_nodes import OversampleNodesTransform
from graph_neural_networks.data.dataset.dynamic.graph_transforms.add_edge_closest_cc import AddEdgeClosestCCTransform

graph_wrapper = GraphWrapper(graph)

graph_wrapper = ComputeDistanceMatrixTransform()(graph_wrapper)
graph_wrapper = OversampleNodesTransform(max_dist=100)(graph_wrapper)
final_wrapper = AddEdgeClosestCCTransform(only_one_edge_per_cc=True)(graph_wrapper)


print(graph_wrapper.in_pred_graph)
graph_final = final_wrapper.graph

# %%
display_graph_overlay(black, graph_final, figsize=(100,100))

# %%
message_passing_edges_index_tensor, gt_edges_index_tensor, virtual_edges_index_tensor, in_pred_edges_index_tensor, not_in_pred_edges_index_tensor = get_edges_index(graph_final)

# %%
print(graph_final)
N = nx.number_of_nodes(graph_final)
print(N)
E = graph_final.number_of_edges()
print(E)

print(E * 2)

# %%
print(message_passing_edges_index_tensor.shape)
print(message_passing_edges_index_tensor)

# %%
print(gt_edges_index_tensor.shape)
print(gt_edges_index_tensor)

# %%
print(virtual_edges_index_tensor.shape)
print(virtual_edges_index_tensor)
print(virtual_edges_index_tensor.t())

# %%
print(in_pred_edges_index_tensor.shape)
print(in_pred_edges_index_tensor)

# %%
print(not_in_pred_edges_index_tensor.shape)
print(not_in_pred_edges_index_tensor)

# %%
n_gt_edges = gt_edges_index_tensor.shape[1]
n_in_pred = in_pred_edges_index_tensor.shape[1]
n_not_in_pred = not_in_pred_edges_index_tensor.shape[1]

print(n_gt_edges, "=", n_in_pred + n_not_in_pred)

# %%
n_virtual_edges = virtual_edges_index_tensor.shape[1]
n_message_passing_edges = message_passing_edges_index_tensor.shape[1]

print(n_message_passing_edges, "=", n_virtual_edges + n_in_pred)

# %%
print(n_virtual_edges + n_in_pred + n_not_in_pred)
print(graph_final.number_of_edges() * 2)

# %%
virtual_edges_set = set([tuple(edge) for edge in virtual_edges_index_tensor.t().tolist()])
print(len(virtual_edges_set))


# %%
def compare_list_and_set(edge_list: list):
    eadge_set = set([tuple(edge) for edge in edge_list])
    print(len(edge_list), len(eadge_set))
    if len(edge_list) != len(eadge_set):
        print("Lists and set are NOT equal!: ", len(edge_list), len(eadge_set))
        visited = set()
        duplicates = set()
        for edge in edge_list:
            edge_tuple = tuple(edge)
            if edge_tuple not in visited:
                visited.add(edge_tuple)
            else:
                duplicates.add(edge_tuple)
        print("Duplicate edges found:", duplicates)
        plt.figure(figsize=(50,50))
        plt.imshow(black, cmap='gray')
        for edge in duplicates:
            u, v = edge
            pos_u = graph_final.nodes[u]['pos']
            pos_v = graph_final.nodes[v]['pos']
            if u == v:
                plt.scatter(pos_u[1], pos_u[0], color='red', s=500)
            plt.plot([pos_u[1], pos_v[1]], [pos_u[0], pos_v[0]], color='red', linewidth=5)
        plt.show()
    else:
        print("Lists and set are equal:", len(edge_list) == len(eadge_set), len(edge_list), len(eadge_set))

compare_list_and_set(virtual_edges_index_tensor.t().tolist())
compare_list_and_set(not_in_pred_edges_index_tensor.t().tolist())
compare_list_and_set(in_pred_edges_index_tensor.t().tolist())
compare_list_and_set(gt_edges_index_tensor.t().tolist())
compare_list_and_set(message_passing_edges_index_tensor.t().tolist())

# %%
distance_matrix = final_wrapper.oversampled_distance_matrix

plt.figure(figsize=(30,30))
plt.imshow(distance_matrix, interpolation='nearest')

for u,v in not_in_pred_edges_index_tensor.t():
    plt.scatter(u, v, c='red')

plt.colorbar()
plt.show()

# %%
y_positives = distance_matrix[not_in_pred_edges_index_tensor[0], not_in_pred_edges_index_tensor[1]]

print(y_positives)

# %%
distance_ratio_sampling_ratio: float = 1.0
hard_negative_sampling_ratio: float = 1.0

# %%
n_positive_edges = not_in_pred_edges_index_tensor.shape[1]
n_distance_ratio_neg_edges = int(n_positive_edges * distance_ratio_sampling_ratio)
n_hard_neg_edges = int(n_positive_edges * hard_negative_sampling_ratio)

# %%
print(n_positive_edges, n_distance_ratio_neg_edges, n_hard_neg_edges)

# %%
from torch_geometric.data import Data

matrix = final_wrapper.oversampled_distance_matrix if final_wrapper.oversampled_distance_matrix is not None else final_wrapper.distance_matrix
matrix = torch.tensor(matrix, dtype=torch.float32)

pos = torch.tensor([list(data['pos']) for _, data in graph_final.nodes(data=True)], dtype=torch.float32)

pyg_graph = Data(x = torch.zeros((N, 1)), 
                 edge_index=message_passing_edges_index_tensor,
                 gt_edge_index = gt_edges_index_tensor,
                 virtual_edge_index = virtual_edges_index_tensor,
                 in_pred_edge_index = in_pred_edges_index_tensor,
                 not_in_pred_edge_index = not_in_pred_edges_index_tensor,
                 distance_matrix = matrix,
                 pos=pos)

# %%
print(pyg_graph)
print(pyg_graph.pos)

# %%
negative_neighbor_sampling = True


# %%
def display_node(img: np.ndarray, graph: nx.Graph, node_id: int):
    plt.figure(figsize=(40,40))
    plt.imshow(img)
    node_pos = graph.nodes(data=True)[node_id]['pos']
    plt.scatter(node_pos[1], node_pos[0], s=100, c='r')
    for n in graph.neighbors(node_id):
        neighbor_pos = graph.nodes(data=True)[n]['pos']
        plt.scatter(neighbor_pos[1], neighbor_pos[0], s=100, c='b')
    plt.show()



# %%
data = pyg_graph

not_in_pred_edge_index = data.not_in_pred_edge_index[:, ::2]
print(not_in_pred_edge_index.shape)
print(not_in_pred_edge_index)
n_positive_edges = not_in_pred_edge_index.shape[1]

distance_matrix = data.distance_matrix
not_in_pred_edge_label = distance_matrix[not_in_pred_edge_index[0], not_in_pred_edge_index[1]]

# %%
in_pred_node_neighbors = {}
for u, v in data.in_pred_edge_index.t().tolist():
    if u not in in_pred_node_neighbors:
        in_pred_node_neighbors[u] = set()
    if v not in in_pred_node_neighbors:
        in_pred_node_neighbors[v] = set()
    in_pred_node_neighbors[u].add(v)
    in_pred_node_neighbors[v].add(u)

y_neighbors_samples = []
neighbors_samples_index = []
if negative_neighbor_sampling:
    for u, v in zip(not_in_pred_edge_index[0], not_in_pred_edge_index[1]):
        u_neighbors = torch.tensor(list(in_pred_node_neighbors.get(u.item(), set())))
        if len(u_neighbors) > 0:
            picked_u_neighbors = u_neighbors[torch.randint(0, len(u_neighbors), (1,))]
            for picked_u in picked_u_neighbors:
                neighbors_samples_index.append((picked_u.item(), v))
                y_neighbors_samples.append(distance_matrix[picked_u, v])
        v_neighbors = torch.tensor(list(in_pred_node_neighbors.get(v.item(), set())))
        if len(v_neighbors) > 0:
            picked_v_neighbors = v_neighbors[torch.randint(0, len(v_neighbors), (1,))]
            for picked_v in picked_v_neighbors:
                neighbors_samples_index.append((u, picked_v.item()))
                y_neighbors_samples.append(distance_matrix[u, picked_v])

neighbors_samples_index = torch.tensor(neighbors_samples_index).t()
neighbors_samples_label = distance_matrix[neighbors_samples_index[0], neighbors_samples_index[1]]

print(neighbors_samples_index.shape)
print(neighbors_samples_label.shape)

# %%
geodesic_euclidean_ratio_sampling_ratio = 1.0

yx_pos = data.pos
euclidean_distance_matrix = torch.cdist(yx_pos, yx_pos, p=2)

distance_ratio_matrix = distance_matrix.detach().clone()
pos_euclid = euclidean_distance_matrix > 0
distance_ratio_matrix[pos_euclid] = distance_ratio_matrix[pos_euclid] / euclidean_distance_matrix[pos_euclid]

plt.imshow(distance_ratio_matrix)
plt.colorbar()
plt.show()

n_geodesic_euclidean_ratio_samples = int(n_positive_edges * geodesic_euclidean_ratio_sampling_ratio)

distance_ratio_matrix_masked = distance_ratio_matrix * torch.tril(torch.ones_like(distance_ratio_matrix))
sorted_y, sorted_x = torch.unravel_index(torch.argsort(distance_ratio_matrix_masked.flatten(), descending=True), distance_ratio_matrix_masked.shape)
top_sorted_x, top_sorted_y = sorted_x[:n_geodesic_euclidean_ratio_samples], sorted_y[:n_geodesic_euclidean_ratio_samples]

geodesic_euclidean_ratio_index = torch.stack([top_sorted_y, top_sorted_x], dim=0)
print(geodesic_euclidean_ratio_index.shape)
geodesic_euclidean_ratio_label = distance_matrix[top_sorted_y, top_sorted_x]
print(geodesic_euclidean_ratio_label.shape)

for i, (u,v) in enumerate(geodesic_euclidean_ratio_index.t()):
    label = geodesic_euclidean_ratio_label[i]
    real_dist = distance_matrix[u.item(), v.item()]
    print(f"Edge {i}: ({u.item()}, {v.item()}) -> label: {label.item()}, real dist: {real_dist.item()}")
    if real_dist.item() != label.item():
        print("Mismatch found!")

# %%
edge_label = torch.cat([not_in_pred_edge_label, neighbors_samples_label, geodesic_euclidean_ratio_label], axis=0)

print(not_in_pred_edge_label.shape)
print(neighbors_samples_label.shape)
print(geodesic_euclidean_ratio_label.shape)

print(edge_label.shape)

# %%
edge_label_index = torch.cat([not_in_pred_edge_index, neighbors_samples_index, geodesic_euclidean_ratio_index], axis=1)

print(not_in_pred_edge_index.shape)
print(neighbors_samples_index.shape)
print(geodesic_euclidean_ratio_index.shape)

print(edge_label_index.shape)

# %%
print(edge_label)

# %%
data.edge_label = edge_label
data.edge_label_index = edge_label_index

# %%
for i, (u,v) in enumerate(edge_label_index.t()):
    label = edge_label[i]
    real_dist = distance_matrix[u.item(), v.item()]
    print(f"Edge {i}: ({u.item()}, {v.item()}) -> label: {label.item()}, real dist: {real_dist.item()}")
    if real_dist.item() != label.item():
        print("Mismatch found!")

# %%
for i, (u,v) in enumerate(not_in_pred_edge_index.t()):
    label = not_in_pred_edge_label[i]
    real_dist = distance_matrix[u.item(), v.item()]
    print(f"Edge {i}: ({u.item()}, {v.item()}) -> label: {label.item()}, real dist: {real_dist.item()}")
    if real_dist.item() != label.item():
        print("Mismatch found!")

# %%
for i, (u,v) in enumerate(neighbors_samples_index.t()):
    label = neighbors_samples_label[i]
    real_dist = distance_matrix[u.item(), v.item()]
    print(f"Edge {i}: ({u.item()}, {v.item()}) -> label: {label.item()}, real dist: {real_dist.item()}")
    if real_dist.item() != label.item():
        print("Mismatch found!")

# %%
