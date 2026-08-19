import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import networkx as nx
import torch
import os

from graph.graph_pred_state import get_combined_graph
from graph.graph_pred_state import EdgePredState
from graph.graph_oversampling import OversampleNodesTransform
from graph.graph_wrapper import GraphWrapper
from path_neural_networks.utils.paths_creation import cut_mask_from_negative_edges_for_all, clean_paths_on_surface_of_mask, get_query_edges

from utils.reconstruction.path_reconstruction.euclidean_path_reconstruction import EuclideanPathReconstructionMethod

path_reconstruction_method = EuclideanPathReconstructionMethod()

def get_positive_samples(new_nx_graph: nx.Graph, 
                         pred_np: np.ndarray,
                         max_dist: float
) -> tuple[list, list]:
    '''
    Computes positive samples for training the path reconstruction model.
    It identifies edges that are in the ground truth but not in the prediction and get the paths for these edges.
    The reconstructed paths are then filtered to get clean positive samples for training the model.

    Args:
        new_nx_graph (nx.Graph): The graph containing all the edges (both in prediction and not in prediction).
        pred_np (np.ndarray): The numpy array representing the prediction mask.
        max_dist (float): The maximum distance threshold to consider for true path edges.

    Returns:
        tuple[list, list]: A tuple containing the list of positive centerlines and their corresponding classes (all 1)
    '''
    true_path_edges = []
    for u, v, d in new_nx_graph.edges(data=True):
        edge_pred_state = d.get("edge_pred_state", None)
        u_pos, v_pos = new_nx_graph.nodes[u]['pos'], new_nx_graph.nodes[v]['pos']
        if edge_pred_state in [EdgePredState.NOT_IN_PREDICTION, EdgePredState.NOT_IN_PREDICTION.value]:
            euclidean_dist = np.linalg.norm(np.array(u_pos) - np.array(v_pos))
            if u_pos != v_pos and euclidean_dist <= max_dist:
                true_path_edges.append([u, v])

    true_path_edges = torch.tensor(true_path_edges, dtype=torch.long).t().contiguous()
    train_positive_centerlines = path_reconstruction_method.reconstruct(map=None, graph=new_nx_graph, new_edges=true_path_edges)
    train_positive_centerlines = [[list([int(x), int(y)]) for (x, y) in path] for path in train_positive_centerlines]
    train_positive_centerlines = cut_mask_from_negative_edges_for_all(train_positive_centerlines, pred_np)["new_centerlines"]
    train_positive_centerlines = clean_paths_on_surface_of_mask(train_positive_centerlines, pred_np, kernel_size=2, threshold=0.5)
    train_positive_classes = [1] * len(train_positive_centerlines)

    return train_positive_centerlines, train_positive_classes

def get_negative_samples(new_nx_graph: nx.Graph, 
                         in_pred_graph: nx.Graph,
                         pred_np: np.ndarray,
                         gt_np: np.ndarray,
                         max_dist: float,
                         n_closest: int,
) -> tuple[list, list]:
    '''
    Computes negative samples for training the path reconstruction model. 
    It identifies edges that are not present in the graph and get the paths for these edges. 
    The reconstructed paths are then filtered to get clean negative samples for training the model.

    Args:
        new_nx_graph (nx.Graph): The graph containing all the edges (both in prediction and not in prediction).
        in_pred_graph (nx.Graph): The graph containing only the edges that are in the prediction.
        pred_np (np.ndarray): The numpy array representing the prediction mask.
        gt_np (np.ndarray): The numpy array representing the ground truth mask.
        max_dist (float): The maximum distance threshold to consider for negative path edges.
        n_closest (int): The number of closest nodes to consider when generating negative samples.

    Returns:
        tuple[list, list]: A tuple containing the list of negative centerlines and their corresponding classes (all 0)
    '''
    false_query_edges = get_query_edges(in_pred_graph, n_closest=n_closest, max_dist=max_dist).t().contiguous().tolist()
    train_negative_edges = []
    for (u, v) in false_query_edges:
        if not new_nx_graph.has_edge(u, v) and not new_nx_graph.has_edge(v, u):
            train_negative_edges.append([u, v])
    train_negative_edges = torch.tensor(train_negative_edges, dtype=torch.long).t().contiguous()

    train_negative_centerlines = path_reconstruction_method.reconstruct(map=None, graph=new_nx_graph, new_edges=train_negative_edges)
    train_negative_centerlines = [[list([int(x), int(y)]) for (x, y) in path] for path in train_negative_centerlines]
    train_negative_centerlines = cut_mask_from_negative_edges_for_all(train_negative_centerlines, pred_np)["new_centerlines"]
    train_negative_centerlines = clean_paths_on_surface_of_mask(train_negative_centerlines, pred_np, kernel_size=2, threshold=0.5)
    train_negative_centerlines = clean_paths_on_surface_of_mask(train_negative_centerlines, gt_np, kernel_size=0, threshold=0.5)
    train_negative_classes = [0] * len(train_negative_centerlines)

    return train_negative_centerlines, train_negative_classes

def process_case(i: int, 
                 data_dir: str,
                 centerlines_folder: str,
                 centerline_max_dist: float,
                 oversampling_max_dist: float,
                 n_closest: int,
                 display: bool = False) -> None:
    '''
    Process a single case to generate training data for path reconstruction, and save the generated centerlines and their classes to a JSON file.

    Args:
        i (int): Index of the case to process.
        centerline_max_dist (float): The maximum distance threshold to consider for path when generating samples.
        oversampling_max_dist (float): The maximum distance threshold to consider when oversampling nodes in the graph.
        n_closest (int): The number of closest nodes to consider when generating negative samples.
        display (bool): Whether to display the case being processed (default: False).
    '''

    gt_folder = os.path.join(data_dir, "gt")
    pred_folder = os.path.join(data_dir, "pred")
    img_folder = os.path.join(data_dir, "img")

    gt_path_list = os.listdir(gt_folder)
    gt_path_list.sort()

    filename = gt_path_list[i]
    gt_path = os.path.join(gt_folder, filename)
    pred_path = os.path.join(pred_folder, filename)
    centerline_path = os.path.join(centerlines_folder, filename.replace(".png", ".json"))

    if os.path.exists(centerline_path):
        print(f"Centerline file already exists for {filename}, skipping...")
        return

    gt_np = np.array(Image.open(gt_path).convert("L")) > 0
    pred_np = np.array(Image.open(pred_path).convert("L")) > 0

    if gt_np.sum() == 0 or pred_np.sum() == 0:
        train_data = {"path_centerlines": [], "edges_classes": []}
        with open(centerline_path, 'w') as f:
            json.dump(train_data, f)
        print(f"No vessels in GT or prediction for {filename}, skipping...")
        return

    # Train data creation
    combined_graph: nx.Graph = get_combined_graph(gt_np, pred_np)
    oversample_nodes_transform = OversampleNodesTransform(oversampling_max_dist, remove_original_edges=True)
    graph_wrapper: GraphWrapper = oversample_nodes_transform(GraphWrapper(combined_graph))
    new_nx_graph: nx.Graph = graph_wrapper.get_graph()
    in_pred_graph: nx.Graph = graph_wrapper.in_pred_graph

    # Get positive and negative samples
    true_path_centerlines, true_edges_classes = get_positive_samples(new_nx_graph, pred_np, centerline_max_dist)
    train_negative_centerlines, train_negative_classes = get_negative_samples(new_nx_graph, in_pred_graph, pred_np, gt_np, centerline_max_dist, n_closest)

    if display:
        img = np.array(Image.open(os.path.join(img_folder, filename)).convert("RGB"))

        mask_img = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
        mask_img[gt_np] = np.array([255, 0, 0], dtype=np.uint8)  # Red for false negatives
        mask_img[pred_np] += np.array([0, 255, 255], dtype=np.uint8)  # Cyan for false positives

        centerlines_img = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
        centerlines_img[pred_np] = [255, 255, 255]
        for centerline in true_path_centerlines:
            for (x, y) in centerline:
                centerlines_img[x, y] = [0, 255, 0]  # Green for true path centerlines
        for centerline in train_negative_centerlines:
            for (x, y) in centerline:
                centerlines_img[x, y] = [255, 0, 0]  # Red for negative samples
        
        fig, axs = plt.subplots(1, 3, figsize=(40, 20))
        axs[0].imshow(img)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        axs[1].imshow(mask_img)
        axs[1].set_title("Prediction and GT: TP (white), FN (red), FP (cyan)")
        axs[1].axis('off')
        axs[2].imshow(centerlines_img)
        axs[2].set_title("Prediction mask and centerlines: positives samples (green), negative samples (red)")
        axs[2].axis('off')
        plt.show()

    # Combine positive and negative samples to create the training data
    train_data =  {
        "path_centerlines": train_negative_centerlines + true_path_centerlines,
        "edges_classes": train_negative_classes + true_edges_classes
    }

    # Save the training data to a JSON file
    with open(centerline_path, 'w') as f:
        json.dump(train_data, f, indent=4)
