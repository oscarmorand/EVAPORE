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
import torch
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.measure import label
import networkx as nx
from graph_neural_networks.data.dataset.dynamic.graph_transforms.oversample_nodes import OversampleNodesTransform
from graph_neural_networks.data.dataset.dynamic.graph_transforms.compute_distance_matrix import ComputeDistanceMatrixTransform
from graph_neural_networks.data.dataset.dynamic.apply_graph_transforms import apply_graph_transforms
from graph_neural_networks.data.utils.pred_state import get_combined_graph
from graph_neural_networks.data.utils.pred_state import EdgePredState
import networkx as nx
import torch
from torch_geometric.data import Data
from graph_neural_networks.reconstruction.path_reconstruction.min_energy_path_reconstruction import SquaredMinEnergyPathReconstructionMethod
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
from skimage.morphology import dilation, disk


# %%
def happend_edge_index_undirected(index_list: list[list[int]],
                                u: int,
                                v: int) -> None:
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
def cut_mask_from_negative_edges(centerline, mask):
    centerline_mask = np.zeros_like(mask, dtype=bool)
    for (x, y) in centerline:
        centerline_mask[x, y] = True

    combined_mask = np.logical_and(centerline_mask, np.logical_not(mask))
    labeled_mask, num_labels = label(combined_mask, return_num=True, connectivity=2)
    centerlines = [[] for _ in range(num_labels)]
    for (x, y) in centerline:
        label_id = labeled_mask[x, y]
        if label_id > 0:
            centerlines[label_id - 1].append([int(x), int(y)])
    
    return centerlines

def cut_mask_from_negative_edges_for_all(centerlines, mask, classes=None, edges=None, return_old_centerlines=False):
    new_centerlines = []
    new_classes = []
    new_edges = []
    old_centerlines = []
    for i, centerline in enumerate(centerlines):
        new_centerline = cut_mask_from_negative_edges(centerline, mask)
        n_new_centerline = len(new_centerline)
        if n_new_centerline > 0:
            new_centerlines.extend(new_centerline)
            if return_old_centerlines:
                old_centerlines.extend([centerline for _ in range(n_new_centerline)])
            if classes is not None:
                cls = classes[i]
                new_classes.extend([cls for _ in range(len(new_centerline))])
            if edges is not None:
                edge = edges[i]
                new_edges.extend([edge for _ in range(len(new_centerline))])

    res = {"new_centerlines": new_centerlines}
    if return_old_centerlines:
        res["old_centerlines"] = old_centerlines
    if classes is not None:
        res["classes"] = new_classes
    if edges is not None:
        res["edges"] = new_edges
    return res


# %%
def get_query_edges(graph: nx.Graph, n_closest: int = 1) -> torch.Tensor:
    G = graph.copy()

    connected_components = list(nx.connected_components(graph))

    cc_extrimities = {}
    for cc_i, cc in enumerate(connected_components):
        extrimities = []
        for n in cc:
            if G.degree(n) == 1:
                extrimities.append(n)
        cc_extrimities[cc_i] = extrimities

    cc_n_count = [len(cc) for cc in connected_components]
    main_cc = np.argmax(cc_n_count)
    small_ccs = {i: cc for i, cc in enumerate(connected_components) if i != main_cc}

    virtual_edges_to_add = []
    for i, cc in small_ccs.items():
        extremities = cc_extrimities[i]
        other_ccs = {j: other_cc for j, other_cc in enumerate(connected_components) if j != i}
        other_ccs_all_nodes = []
        for other_cc in other_ccs.values():
            other_ccs_all_nodes.extend(other_cc)

        for extrimity in extremities:
            other_nodes_distances = []
            extrimity_pos = np.array(G.nodes[extrimity]['pos'])
            for other_cc_node in other_ccs_all_nodes:
                other_cc_node_pos = np.array(G.nodes[other_cc_node]['pos'])
                dist = np.linalg.norm(extrimity_pos - other_cc_node_pos)
                other_nodes_distances.append(dist)
            other_nodes_distances = np.array(other_nodes_distances)
            closest_indices = np.argsort(other_nodes_distances)[:n_closest]
            closest_nodes = [other_ccs_all_nodes[idx] for idx in closest_indices]

            for extremity_closest_other_cc_node in closest_nodes:
                if (extrimity, extremity_closest_other_cc_node) not in virtual_edges_to_add and (extremity_closest_other_cc_node, extrimity) not in virtual_edges_to_add:
                    virtual_edges_to_add.append([extrimity, extremity_closest_other_cc_node])
    
    virtual_edges_index_tensor = torch.tensor(virtual_edges_to_add, dtype=torch.long).t().contiguous()
    return virtual_edges_index_tensor


# %%
def clean_paths_on_surface_of_mask(centerlines, mask, kernel_size=1, threshold=0.5, classes=None):
    new_centerlines = []
    if classes is not None:
        new_classes = []
        
    labeled_mask = label(mask)
    dilated_mask = dilation(labeled_mask, disk(kernel_size))
    
    for i, centerline in enumerate(centerlines):
        centerline_mask = np.zeros_like(mask, dtype=np.uint8)
        for (x, y) in centerline:
            centerline_mask[x, y] = 1

        combined_mask = centerline_mask * dilated_mask
        values_on_mask = np.count_nonzero(combined_mask)
        n_diff_cc = len(np.unique(combined_mask)) - (1 if 0 in combined_mask else 0)
        ratio_on_mask = values_on_mask / len(centerline)

        if ratio_on_mask <= threshold or n_diff_cc > 1:
            new_centerlines.append(centerline)
            if classes is not None:
                new_classes.append(classes[i])

    if classes is not None:
        return new_centerlines, new_classes
    return new_centerlines


# %%
import math

def is_reconstructed_path_not_too_far(true_path_existing_centerline, true_path_reconstructed_centerline, distance_ratio_threshold):
    sum_min_distances = 0
    for (xr, yr) in true_path_reconstructed_centerline:
        r = np.array([xr, yr])
        min_dist = float('inf')
        for (xe, ye) in true_path_existing_centerline:
            e = np.array([xe, ye])
            dist = np.linalg.norm(r - e)
            if dist < min_dist:
                min_dist = dist
        sum_min_distances += min_dist
    sum_min_distances /= len(true_path_reconstructed_centerline)
    return sum_min_distances <= distance_ratio_threshold

def remove_too_far_reconstructed_paths_for_all(true_path_existing_centerlines, true_path_reconstructed_centerlines, distance_ratio_threshold):
    new_reconstructed_classes = []
    for i, true_path_reconstructed_centerline in enumerate(true_path_reconstructed_centerlines):
        true_path_existing_centerline = true_path_existing_centerlines[i]
        condition = is_reconstructed_path_not_too_far(true_path_existing_centerline, true_path_reconstructed_centerline, distance_ratio_threshold)
        new_reconstructed_classes.append(int(condition))
    return new_reconstructed_classes


# %%
import os
import json
from graph.graph_visualization import display_graph_overlay
import matplotlib.pyplot as plt
from graph.graph_creation import img_to_graph

data_folder = "/home/morand/afs/EVAPORE/data/FIVES/"
gt_folder = os.path.join(data_folder, "gt")
pred_folder = os.path.join(data_folder, "pred")
img_folder = os.path.join(data_folder, "img")
prob_map_folder = os.path.join(data_folder, "probability_maps")
old_centerlines_folder = os.path.join(data_folder, "old_centerlines")
new_centerlines_folder = os.path.join(data_folder, "centerlines")

gt_path_list = os.listdir(gt_folder)
gt_path_list.sort()
print(len(gt_path_list))

path_reconstruction_method = SquaredMinEnergyPathReconstructionMethod()
train_transforms = {
        "compute_distance_matrix_transform": ComputeDistanceMatrixTransform(skip_first_neighbor=True),
        "oversample_nodes_transform": OversampleNodesTransform(max_dist=50, remove_original_edges=True),
}
eval_transforms = {
    "oversample_nodes_transform": OversampleNodesTransform(max_dist=50, remove_original_edges=True),
}

debug_n = 1
plot = False


# %%
def process_case(i):
    # ========= Load data ========
    filename = gt_path_list[i]
    print("Processing file:", filename)
    gt_path = os.path.join(gt_folder, filename)
    pred_path = os.path.join(pred_folder, filename)
    prob_map_path = os.path.join(prob_map_folder, filename.replace(".png", ".pt"))

    centerlines_path = os.path.join(old_centerlines_folder, filename.replace(".png", ".json"))
    with open(centerlines_path, 'r') as f:
        centerlines_data = json.load(f)

    prob_map = torch.load(prob_map_path)[1]
    gt_np = np.array(Image.open(gt_path).convert("L")) > 0
    segmentation_mask_np = np.array(Image.open(pred_path).convert("L")) > 0

    # ======== Train data creation ========
    combined_graph: nx.Graph = get_combined_graph(gt_np, segmentation_mask_np)
    graph_wrapper = GraphWrapper(combined_graph)
    new_nx_graph: nx.Graph = apply_graph_transforms(graph_wrapper, train_transforms).get_graph()
    in_pred_graph: nx.Graph = graph_wrapper.in_pred_graph

    # Positive sampling
    true_path_edges = []
    true_path_existing_centerlines = []
    for u, v, d in new_nx_graph.edges(data=True):
        edge_pred_state = d.get("edge_pred_state", None)
        if edge_pred_state in [EdgePredState.NOT_IN_PREDICTION, EdgePredState.NOT_IN_PREDICTION.value]:
            if new_nx_graph.degree(u) > 1 and new_nx_graph.degree(v) > 1:
                true_path_edges.append([u, v])
                path = list(d["centerline"])
                true_path_existing_centerlines.append(path)

    true_path_edges = torch.tensor(true_path_edges, dtype=torch.long).t().contiguous()
    true_path_reconstructed_centerlines = path_reconstruction_method.reconstruct(map=prob_map, graph=new_nx_graph, new_edges=true_path_edges)
    true_path_reconstructed_centerlines = [[list([int(x), int(y)]) for (x, y) in path] for path in true_path_reconstructed_centerlines]
    true_path_reconstruced_classes = remove_too_far_reconstructed_paths_for_all(true_path_existing_centerlines, true_path_reconstructed_centerlines, distance_ratio_threshold=5)
    res = cut_mask_from_negative_edges_for_all(true_path_reconstructed_centerlines, segmentation_mask_np, classes=true_path_reconstruced_classes)
    true_path_reconstructed_centerlines = res["new_centerlines"]
    true_path_reconstruced_classes = res["classes"]
    true_path_reconstructed_centerlines, true_path_reconstruced_classes = clean_paths_on_surface_of_mask(true_path_reconstructed_centerlines, segmentation_mask_np, kernel_size=2, threshold=0.5, classes=true_path_reconstruced_classes)

    res = cut_mask_from_negative_edges_for_all(true_path_existing_centerlines, segmentation_mask_np)
    true_path_existing_centerlines = res["new_centerlines"]
    true_path_existing_centerlines = clean_paths_on_surface_of_mask(true_path_existing_centerlines, segmentation_mask_np, kernel_size=2,threshold=0.5)
    true_path_existing_classes = [1] * len(true_path_existing_centerlines)

    true_path_centerlines = true_path_existing_centerlines + true_path_reconstructed_centerlines
    true_edges_classes = true_path_existing_classes + true_path_reconstruced_classes

    # Negative sampling
    false_query_edges = get_query_edges(in_pred_graph, n_closest=5).t().contiguous().tolist()
    train_negative_edges = []
    for (u, v) in false_query_edges:
        if not new_nx_graph.has_edge(u, v) and not new_nx_graph.has_edge(v, u):
            train_negative_edges.append([u, v])
    train_negative_edges = torch.tensor(train_negative_edges, dtype=torch.long).t().contiguous()

    train_negative_centerlines = path_reconstruction_method.reconstruct(map=prob_map, graph=new_nx_graph, new_edges=train_negative_edges)
    train_negative_centerlines = [[list([int(x), int(y)]) for (x, y) in path] for path in train_negative_centerlines]
    res = cut_mask_from_negative_edges_for_all(train_negative_centerlines, segmentation_mask_np)
    train_negative_centerlines = res["new_centerlines"]
    train_negative_centerlines = clean_paths_on_surface_of_mask(train_negative_centerlines, segmentation_mask_np, kernel_size=2, threshold=0.5)
    train_negative_classes = [0] * len(train_negative_centerlines)

    # Combine positive and negative samples
    train_path_centerlines = train_negative_centerlines + true_path_centerlines
    train_edges_classes = train_negative_classes + true_edges_classes

    new_train_data =  {
        "path_centerlines": train_path_centerlines,
        "edges_classes": train_edges_classes
    }

    # ======== Eval data ========
    pred_graph: nx.Graph = img_to_graph(segmentation_mask_np, clean = True, closing_radius=1, return_pixel_graph=False)
    graph_wrapper = GraphWrapper(pred_graph)
    new_nx_graph: nx.Graph = apply_graph_transforms(graph_wrapper, eval_transforms).get_graph()

    nodes_radius_data = {}
    for id, n_data in new_nx_graph.nodes(data=True):
        nodes_radius_data[id] = float(n_data["radius"])

    virtual_edges_index_tensor = get_query_edges(new_nx_graph, n_closest=5)
    eval_edges = virtual_edges_index_tensor.t().tolist()
    eval_path_centerlines = path_reconstruction_method.reconstruct(map=prob_map, graph=new_nx_graph, new_edges=virtual_edges_index_tensor)
    eval_path_centerlines = [[list([int(x), int(y)]) for (x, y) in path] for path in eval_path_centerlines]
    res = cut_mask_from_negative_edges_for_all(eval_path_centerlines, segmentation_mask_np, edges=eval_edges, return_old_centerlines=True)
    eval_path_centerlines_cut = res["new_centerlines"]
    eval_path_centerlines = res["old_centerlines"]
    eval_edges = res["edges"]

    print("Number of eval edges:", len(eval_edges))
    print("Number of nodes radius data:", len(nodes_radius_data))
    print("Number of eval paths before cut:", len(eval_path_centerlines))
    print("Number of eval paths after cut:", len(eval_path_centerlines_cut))

    plt.figure(figsize=(80,80))
    img_background = np.zeros((segmentation_mask_np.shape[0], segmentation_mask_np.shape[1], 3), dtype=np.uint8)
    img_background[segmentation_mask_np] = [255,255,255]
    for path in eval_path_centerlines:
        for (x, y) in path:
            img_background[x, y] += np.array([0, 50, 0], dtype=np.uint8)
    plt.imshow(img_background)
    plt.show()

    plt.figure(figsize=(80,80))
    img_background = np.zeros((segmentation_mask_np.shape[0], segmentation_mask_np.shape[1], 3), dtype=np.uint8)
    img_background[segmentation_mask_np] = [255,255,255]
    for path in eval_path_centerlines:
        for (x, y) in path:
            img_background[x, y] = [0, 255, 0]
    for path in eval_path_centerlines_cut:
        for (x, y) in path:
            img_background[x, y] = [255, 0, 0]
    plt.imshow(img_background)
    plt.show()

    new_eval_data = {
        "edges": eval_edges,
        "nodes_radius": nodes_radius_data,
        "path_centerlines": eval_path_centerlines_cut,
        "full_path_centerlines": eval_path_centerlines
    }

    # ======== Visualization ========

    plt.figure(figsize=(80,80))
    img_background = np.zeros((segmentation_mask_np.shape[0], segmentation_mask_np.shape[1], 3), dtype=np.uint8)
    img_background[segmentation_mask_np] = [255,255,255]
    for path, edge_class in zip(train_path_centerlines, train_edges_classes):
        for (x, y) in path:
            img_background[x, y] = [0, 255, 0] if edge_class == 1 else [255, 0, 0]
    for path in eval_path_centerlines_cut:
        for (x, y) in path:
            img_background[x, y] = np.clip(img_background[x, y] + np.array([0, 0, 255]), 0, 255)
    plt.imshow(img_background)
    plt.show()

    # ======== Save data ========

    centerlines_data = {
        "train": new_train_data,
        "eval": new_eval_data
    }
    #with open(centerlines_path, 'w') as f:
    #    json.dump(centerlines_data, f, indent=4)

process_case(50)

# %%
