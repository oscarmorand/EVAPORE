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
#     display_name: graph-neural-networks (3.12.11)
#     language: python
#     name: python3
# ---

# %%
import os
from PIL import Image
import numpy as np

from graph_neural_networks.data.utils.pred_state import get_combined_graph
from graph.graph_visualization import display_graph_overlay

data_folder = "/home/morand/afs/EVAPORE/data/FIVES_clean/"
gt_folder = os.path.join(data_folder, "gt")
pred_folder = os.path.join(data_folder, "pred")

filename = "FIVES_001.png"

gt_path = os.path.join(gt_folder, filename)
pred_path = os.path.join(pred_folder, filename)
gt_np = np.array(Image.open(gt_path).convert("L")) > 0
pred_np = np.array(Image.open(pred_path).convert("L")) > 0

combined_graph = get_combined_graph(gt=pred_np, pred=gt_np)
display_graph_overlay(gt_np, combined_graph, figsize=(40, 40))

# %%
import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from graph.graph_creation import img_to_graph
import networkx as nx
import torch
import matplotlib.pyplot as plt
from graph.graph_visualization import display_graph_overlay
from graph_neural_networks.data.utils.pred_state import EdgePredState

data_folder = "/home/morand/afs/EVAPORE/data/FIVES_clean/"
gt_folder = os.path.join(data_folder, "gt")
pred_folder = os.path.join(data_folder, "pred")
img_folder = os.path.join(data_folder, "img")

centerlines_folder = os.path.join(data_folder, f"filtering_centerlines")

gt_path_list = os.listdir(gt_folder)
gt_path_list.sort()
print(len(gt_path_list))

def process_case(i):
    # ========= Load data ========
    filename = gt_path_list[i]
    gt_path = os.path.join(gt_folder, filename)
    pred_path = os.path.join(pred_folder, filename)
    new_centerline_path = os.path.join(centerlines_folder, filename.replace(".png", ".json"))

    gt_np = np.array(Image.open(gt_path).convert("L")) > 0
    pred_np = np.array(Image.open(pred_path).convert("L")) > 0

    # ======== Train data creation ========
    # We reverse the gt and pred in the function because we want to get the parts of the pred skeleton that are not in the gt
    pred_combined_graph: nx.Graph = get_combined_graph(gt=pred_np, pred=gt_np)

    train_positive_centerlines = []
    train_negative_centerlines = []

    ccs = nx.connected_components(pred_combined_graph)
    for cc in ccs:
        subgraph = pred_combined_graph.subgraph(cc)
        centerlines = []
        edge_pred_state = None
        edges_pred_states = None
        for n1, n2, e_data in subgraph.edges(data=True):
            edge_pred_state = e_data["edge_pred_state"]
            if edges_pred_states is None:
                edges_pred_states = edge_pred_state
            if edge_pred_state != edges_pred_states:
                break
            centerline = e_data["centerline"]
            centerlines.append(centerline)
        if edge_pred_state == edges_pred_states:
            if edge_pred_state == EdgePredState.IN_PREDICTION:
                train_positive_centerlines.extend(centerlines)
            else:
                train_negative_centerlines.extend(centerlines)

    train_positive_classes = [1] * len(train_positive_centerlines)
    train_negative_classes = [0] * len(train_negative_centerlines)

    # Combine positive and negative samples
    train_path_centerlines = train_negative_centerlines + train_positive_centerlines
    train_edges_classes = train_negative_classes + train_positive_classes

    new_train_data =  {
        "path_centerlines": train_path_centerlines,
        "edges_classes": train_edges_classes
    }

    # ======== Eval data ========
    pred_graph: nx.Graph = img_to_graph(pred_np, clean = True, closing_radius=1, return_pixel_graph=False)

    nodes_radius_data = {}
    for id, n_data in pred_graph.nodes(data=True):
        nodes_radius_data[id] = float(n_data["radius"])

    eval_path_centerlines_cut = []
    eval_path_centerlines = []
    eval_edges = []

    ccs = nx.connected_components(pred_graph)
    for cc in ccs:
        subgraph = pred_graph.subgraph(cc)
        for n1, n2, e_data in subgraph.edges(data=True):
            centerline = e_data["centerline"]
            eval_edges.append((n1, n2))
            eval_path_centerlines_cut.append(centerline)
            eval_path_centerlines.append(centerline)

    new_eval_data = {
        "edges": eval_edges,
        "nodes_radius": nodes_radius_data,
        "path_centerlines": eval_path_centerlines_cut,
        "full_path_centerlines": eval_path_centerlines
    }

    # ======== Save data ========

    centerlines_data = {
        "train": new_train_data,
        "eval": new_eval_data
    }
    with open(new_centerline_path, 'w') as f:
        json.dump(centerlines_data, f, indent=4)


# %%
process_case(0)

# %%
for i in tqdm(range(len(gt_path_list))):
    process_case(i)

# %%
for i in tqdm(range(len(gt_path_list))):
    filename = gt_path_list[i]
    new_centerline_path = os.path.join(centerlines_folder, filename.replace(".png", ".json"))

    with open(new_centerline_path, 'r') as f:
        centerlines_data = json.load(f)

    train_data = centerlines_data["train"]
    eval_data = centerlines_data["eval"]

    train_path_centerlines = train_data["path_centerlines"]
    train_edges_classes = train_data["edges_classes"]

    if len(train_edges_classes) == 0:
        print(f"Case {filename} has no training data.")

# %%
import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from graph.graph_creation import img_to_graph
import networkx as nx
import torch
import matplotlib.pyplot as plt
from graph.graph_visualization import display_graph_overlay
from graph_neural_networks.data.utils.pred_state import EdgePredState

data_folder = "/home/morand/afs/EVAPORE/data/FIVES_clean/"
gt_folder = os.path.join(data_folder, "gt")
pred_folder = os.path.join(data_folder, "pred")
img_folder = os.path.join(data_folder, "img")

centerlines_folder = os.path.join(data_folder, f"all_filtering_centerlines")

gt_path_list = os.listdir(gt_folder)
gt_path_list.sort()
print(len(gt_path_list))

def process_case(i):
    # ========= Load data ========
    filename = gt_path_list[i]
    gt_path = os.path.join(gt_folder, filename)
    pred_path = os.path.join(pred_folder, filename)
    new_centerline_path = os.path.join(centerlines_folder, filename.replace(".png", ".json"))

    gt_np = np.array(Image.open(gt_path).convert("L")) > 0
    pred_np = np.array(Image.open(pred_path).convert("L")) > 0

    # ======== Train data creation ========
    # We reverse the gt and pred in the function because we want to get the parts of the pred skeleton that are not in the gt
    pred_combined_graph: nx.Graph = get_combined_graph(gt=pred_np, pred=gt_np)
    gt_combined_graph: nx.Graph = get_combined_graph(gt=gt_np, pred=pred_np)

    train_positive_centerlines = []
    train_negative_centerlines = []

    for n1, n2, e_data in pred_combined_graph.edges(data=True):
        edge_pred_state = e_data["edge_pred_state"]
        centerline = e_data["centerline"]
        if edge_pred_state == EdgePredState.IN_PREDICTION:
            train_positive_centerlines.append(centerline)
        else:
            train_negative_centerlines.append(centerline)

    for n1, n2, e_data in gt_combined_graph.edges(data=True):
        edge_pred_state = e_data["edge_pred_state"]
        centerline = e_data["centerline"]
        if edge_pred_state == EdgePredState.NOT_IN_PREDICTION:
            train_positive_centerlines.append(centerline)

    train_positive_classes = [1] * len(train_positive_centerlines)
    train_negative_classes = [0] * len(train_negative_centerlines)

    print(f"Case {filename}: {len(train_positive_centerlines)} positive samples, {len(train_negative_centerlines)} negative samples.")

    # Combine positive and negative samples
    train_path_centerlines = train_negative_centerlines + train_positive_centerlines
    train_edges_classes = train_negative_classes + train_positive_classes

    new_train_data =  {
        "path_centerlines": train_path_centerlines,
        "edges_classes": train_edges_classes
    }

    # ======== Eval data ========
    pred_graph: nx.Graph = img_to_graph(pred_np, clean = True, closing_radius=1, return_pixel_graph=False)

    nodes_radius_data = {}
    for id, n_data in pred_graph.nodes(data=True):
        nodes_radius_data[id] = float(n_data["radius"])

    eval_path_centerlines_cut = []
    eval_path_centerlines = []
    eval_edges = []

    for n1, n2, e_data in pred_graph.edges(data=True):
        centerline = e_data["centerline"]
        eval_edges.append((n1, n2))
        eval_path_centerlines_cut.append(centerline)
        eval_path_centerlines.append(centerline)

    new_eval_data = {
        "edges": eval_edges,
        "nodes_radius": nodes_radius_data,
        "path_centerlines": eval_path_centerlines_cut,
        "full_path_centerlines": eval_path_centerlines
    }

    # ======== Save data ========

    centerlines_data = {
        "train": new_train_data,
        "eval": new_eval_data
    }
    with open(new_centerline_path, 'w') as f:
        json.dump(centerlines_data, f, indent=4)


# %%
process_case(0)

# %%
for i in tqdm(range(len(gt_path_list))):
    process_case(i)

# %%
for i in tqdm(range(len(gt_path_list))):
    filename = gt_path_list[i]
    new_centerline_path = os.path.join(centerlines_folder, filename.replace(".png", ".json"))

    with open(new_centerline_path, 'r') as f:
        centerlines_data = json.load(f)

    train_data = centerlines_data["train"]
    train_edges_classes = train_data["edges_classes"]

    train_edges_classes = [1 - c for c in train_edges_classes] # Reverse the classes to get less samples of class 1, and then a better usage of precision-recall curve for evaluation

    train_data["edges_classes"] = train_edges_classes
    centerlines_data["train"] = train_data

    with open(new_centerline_path, 'w') as f:
        json.dump(centerlines_data, f, indent=4)

# %%
