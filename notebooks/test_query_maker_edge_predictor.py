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
import torch.nn as nn
from torch_geometric.data import Batch, Data
from abc import ABC


# %%
class EdgeQuery(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError
        
class EdgeQueryAllNonExisting(EdgeQuery):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_index = batch.edge_index
        num_nodes = batch.num_nodes
        full_edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        existing_edges_set = set((u.item(), v.item()) for u, v in edge_index.t())
        non_existing_edges = [
            (u.item(), v.item()) for u, v in full_edge_index.t()
            if (u.item(), v.item()) not in existing_edges_set and (v.item(), u.item()) not in existing_edges_set
        ]
        if len(non_existing_edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(non_existing_edges, dtype=torch.long).t()
        return edge_index_query


class EdgeQueryMaxDistance(EdgeQuery):
    def __init__(self, max_dist: float) -> None:
        super().__init__()
        self.max_dist = max_dist

    def forward(self, batch: Batch) -> torch.Tensor:
        nodes = torch.arange(batch.num_nodes)
        edge_index = batch.edge_index
        existing_edges = set((u.item(), v.item()) for u, v in edge_index.t())
        edge_index_query = []
        for n1 in nodes:
            for n2 in nodes:
                if n1 >= n2:
                    continue
                if (n1.item(), n2.item()) in existing_edges or (n2.item(), n1.item()) in existing_edges:
                    continue
                pos1 = batch.pos[n1]
                pos2 = batch.pos[n2]
                distance = torch.norm(pos1 - pos2)
                if distance <= self.max_dist:
                    edge_index_query.append((n1.item(), n2.item()))

        if len(edge_index_query) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(edge_index_query, dtype=torch.long).t()
        return edge_index_query
    

class EdgeQueryExtremitiesMaxDistance(EdgeQuery):
    def __init__(self, max_dist: float) -> None:
        super().__init__()
        self.max_dist = max_dist

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_index = batch.edge_index
        nodes = torch.arange(batch.num_nodes)
        existing_edges = set((u.item(), v.item()) for u, v in edge_index.t())

        node_degrees = torch.zeros(batch.num_nodes, dtype=torch.long)
        for u, v in edge_index.t():
            node_degrees[u] += 1
            node_degrees[v] += 1
        if (node_degrees == 1).sum() == 0:
            node_degrees = node_degrees / 2 # undirected graph
        extremity_nodes = nodes[node_degrees <= 1]

        edge_index_query = set()
        for extremity_node in extremity_nodes:
            for n in nodes:
                if extremity_node == n:
                    continue
                if (extremity_node.item(), n.item()) in existing_edges or (n.item(), extremity_node.item()) in existing_edges:
                    continue
                pos1 = batch.pos[extremity_node]
                pos2 = batch.pos[n]
                distance = torch.norm(pos1 - pos2)
                if distance <= self.max_dist:
                    edge_index_query.add((min(extremity_node.item(), n.item()), max(extremity_node.item(), n.item())))

        if len(edge_index_query) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(list(edge_index_query), dtype=torch.long).t()
        return edge_index_query


# %%
edge_index = torch.tensor([[0, 1, 2, 1, 3, 3],
                        [1, 3, 3, 0, 1, 2]], dtype=torch.long)
x = torch.randn((4, 16))  # 4 nodes with 16 features each
pos = torch.tensor([[0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index, pos=pos, num_nodes=x.shape[0])

batch_1 = Batch(
    **data
)

print(batch_1)

# %%
query_maker = EdgeQueryMaxDistance(max_dist=1.5)
query_edge_index = query_maker(batch_1)

print(query_edge_index)

# %%
query_maker = EdgeQueryMaxDistance(max_dist=1.1)
query_edge_index = query_maker(batch_1)

print(query_edge_index)

# %%
query_maker = EdgeQueryExtremitiesMaxDistance(max_dist=1.5)
query_edge_index = query_maker(batch_1)

print(query_edge_index)

# %%
edge_index = torch.tensor([[0, 1, 3, 4, 1, 2, 4, 5],
                        [1, 2, 4, 5, 0, 1, 3, 4]], dtype=torch.long)
x = torch.randn((6, 16))  # 4 nodes with 16 features each
pos = torch.tensor([[1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 2.0]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index, pos=pos, num_nodes=x.shape[0])

batch_2 = Batch(
    **data
)

print(batch_2)

# %%
query_maker = EdgeQueryMaxDistance(max_dist=1.5)
query_edge_index = query_maker(batch_2)

print(query_edge_index)

# %%
query_maker = EdgeQueryMaxDistance(max_dist=1.1)
query_edge_index = query_maker(batch_2)

print(query_edge_index)

# %%
query_maker = EdgeQueryExtremitiesMaxDistance(max_dist=1.5)
query_edge_index = query_maker(batch_2)

print(query_edge_index)


# %%
class EdgeQueryDiffCCMaxDistance(EdgeQuery):
    def __init__(self, max_dist: float) -> None:
        super().__init__()
        self.max_dist = max_dist

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_index = batch.edge_index
        nodes = torch.arange(batch.num_nodes)

        cc_labels = []
        visited = set()
        queue = []
        for node in nodes:
            if node.item() in visited:
                continue
            queue.append(node.item())
            current_cc = []
            while queue:
                current_node = queue.pop(0)
                if current_node in visited:
                    continue
                visited.add(current_node)
                current_cc.append(current_node)
                neighbors = edge_index[1][edge_index[0] == current_node].tolist() + edge_index[0][edge_index[1] == current_node].tolist()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
            cc_labels.append(current_cc)

        # Find pairs of nodes in different connected components
        edge_index_query = []
        for i, cc1 in enumerate(cc_labels):
            for j, cc2 in enumerate(cc_labels):
                if i >= j:
                    continue
                for n1 in cc1:
                    for n2 in cc2:
                        pos1 = batch.pos[n1]
                        pos2 = batch.pos[n2]
                        distance = torch.norm(pos1 - pos2)
                        if distance <= self.max_dist:
                            edge_index_query.append((n1, n2))

        if len(edge_index_query) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(edge_index_query, dtype=torch.long).t()
        return edge_index_query


# %%
query_maker = EdgeQueryDiffCCMaxDistance(max_dist=3.0)

# %%
query_edge_index = query_maker(batch_1)

print(query_edge_index)

# %%
query_edge_index = query_maker(batch_2)
print(query_edge_index)


# %%
class EdgeQueryExtremitiesDiffCCMaxDistance(EdgeQuery):
    def __init__(self, max_dist: float) -> None:
        super().__init__()
        self.max_dist = max_dist

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_index = batch.edge_index
        edge_label_index = batch.edge_label_index
        nodes = torch.arange(batch.num_nodes)

        # Reconstruct adjacency matrix
        adj_matrix = torch.zeros((batch.num_nodes, batch.num_nodes), dtype=torch.bool)
        for u, v in edge_label_index.t():
            adj_matrix[u, v] = True
            adj_matrix[v, u] = True  # undirected graph

        # Find connected components
        cc_labels = []
        visited = set()
        queue = []
        for node in nodes:
            if node.item() in visited:
                continue
            queue.append(node.item())
            current_cc = []
            while queue:
                current_node = queue.pop(0)
                if current_node in visited:
                    continue
                visited.add(current_node)
                current_cc.append(current_node)
                neighbors = adj_matrix[current_node].nonzero(as_tuple=False).view(-1).tolist()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
            cc_labels.append(current_cc)
        print("Number of Connected Components:", len(cc_labels))
        print("Connected Components:", cc_labels)

        # Find extremity nodes
        node_degrees = torch.zeros(batch.num_nodes, dtype=torch.long)
        for u, v in edge_index.t():
            node_degrees[u] += 1
            node_degrees[v] += 1
        if (node_degrees == 1).sum() == 0:
            node_degrees = node_degrees / 2 # undirected graph
        extremity_nodes = nodes[node_degrees <= 1]

        # Find pairs nodes (extremity, other) in different connected components
        edge_index_query = set()
        for extremity_node in extremity_nodes:
            extremity_cc = None
            for i, cc in enumerate(cc_labels):
                if extremity_node.item() in cc:
                    extremity_cc = i
                    break
            for j, cc in enumerate(cc_labels):
                if j == extremity_cc:
                    continue
                for n in cc:
                    if extremity_node == n:
                        continue
                    pos1 = batch.pos[extremity_node]
                    pos2 = batch.pos[n]
                    distance = torch.norm(pos1 - pos2)
                    if distance <= self.max_dist:
                        edge_index_query.add((min(extremity_node.item(), n), max(extremity_node.item(), n)))

        if len(edge_index_query) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(list(edge_index_query), dtype=torch.long).t()
        return edge_index_query


# %%
query_maker = EdgeQueryExtremitiesDiffCCMaxDistance(max_dist=1.5)

# %%
query_edge_index = query_maker(batch_1)
print(query_edge_index)

# %%
query_edge_index = query_maker(batch_2)
print(query_edge_index)


# %%
class EdgePredictor(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, edge_scores: torch.Tensor, query_edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
        
class EdgePredictorThresholded(EdgePredictor):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, edge_scores: torch.Tensor, query_edge_index: torch.Tensor) -> torch.Tensor:
        above_threshold_mask = edge_scores >= self.threshold
        selected_edge_index = query_edge_index[:, above_threshold_mask]
        selected_edge_scores = edge_scores[above_threshold_mask]
        return selected_edge_index, selected_edge_scores


class EdgePredictorMaxThresholded(EdgePredictor):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, edge_scores: torch.Tensor, query_edge_index: torch.Tensor) -> torch.Tensor:
        nodes = query_edge_index.flatten().unique().tolist()

        node_max_coupling = {node: None for node in nodes}
        node_max_coupling_score = {node: float('-inf') for node in nodes}

        above_threshold_mask = edge_scores >= self.threshold
        above_threshold_scores = edge_scores[above_threshold_mask]
        above_threshold_edges = query_edge_index[:, above_threshold_mask]

        if above_threshold_edges.size(1) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=edge_scores.dtype)

        for i, (u, v) in enumerate(above_threshold_edges.t().tolist()):
            score = above_threshold_scores[i].item()
            if score > node_max_coupling_score[u]:
                node_max_coupling_score[u] = score
                node_max_coupling[u] = v
            if score > node_max_coupling_score[v]:
                node_max_coupling_score[v] = score
                node_max_coupling[v] = u
        
        selected_edge_index = []
        selected_edge_scores = []
        for i, (node, max_coupling) in enumerate(zip(nodes, node_max_coupling.values())):
            if max_coupling is not None:
                friends = (min(node, max_coupling), max(node, max_coupling))
                if friends not in selected_edge_index:
                    selected_edge_index.append(friends)
                    selected_edge_scores.append(node_max_coupling_score[node])

        selected_edge_index = torch.tensor(selected_edge_index, dtype=torch.long).t()
        selected_edge_scores = torch.tensor(selected_edge_scores, dtype=edge_scores.dtype)
        return selected_edge_index, selected_edge_scores

class EdgePredictorReciprocalMaxThresholded(EdgePredictor):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, edge_scores: torch.Tensor, query_edge_index: torch.Tensor) -> torch.Tensor:
        nodes = query_edge_index.flatten().unique().tolist()

        node_best_friend = {node: None for node in nodes}
        node_best_friend_score = {node: float('-inf') for node in nodes}

        above_threshold_mask = edge_scores >= self.threshold
        above_threshold_scores = edge_scores[above_threshold_mask]
        above_threshold_edges = query_edge_index[:, above_threshold_mask]

        for i, (u, v) in enumerate(above_threshold_edges.t().tolist()):
            score = above_threshold_scores[i].item()
            if score > node_best_friend_score[u]:
                node_best_friend_score[u] = score
                node_best_friend[u] = v
            if score > node_best_friend_score[v]:
                node_best_friend_score[v] = score
                node_best_friend[v] = u
        print(node_best_friend)

        selected_edge_index = []
        selected_edge_scores = []
        for i, (node, best_friend) in enumerate(zip(nodes, node_best_friend.values())):
            if best_friend is not None:
                best_friend_best_friend = node_best_friend[best_friend]
                if best_friend_best_friend == node:
                    # best friend are reciprocal
                    friends = (min(node, best_friend), max(node, best_friend))
                    if friends not in selected_edge_index:
                        selected_edge_index.append(friends)
                        selected_edge_scores.append(node_best_friend_score[node])

        selected_edge_index = torch.tensor(list(selected_edge_index), dtype=torch.long).t()
        selected_edge_scores = torch.tensor(selected_edge_scores, dtype=edge_scores.dtype)
        return selected_edge_index, selected_edge_scores



# %%
query_maker = EdgeQueryMaxDistance(max_dist=1.5)
query_edge_index = query_maker(batch_1)
edges_scores = torch.tensor([0.2, 0.6, 0.8], dtype=torch.float)
print(query_edge_index)
print(edges_scores)

edge_predictor = EdgePredictorThresholded(threshold=0.5)
predicted_edge_index, predicted_edge_scores = edge_predictor(edges_scores, query_edge_index)
print(predicted_edge_index)
print(predicted_edge_scores)

# %%
edge_predictor = EdgePredictorMaxThresholded(threshold=0.5)
predicted_edge_index, predicted_edge_scores = edge_predictor(edges_scores, query_edge_index)
print(predicted_edge_index)
print(predicted_edge_scores)

# %%
edge_predictor = EdgePredictorReciprocalMaxThresholded(threshold=0.5)
predicted_edge_index, predicted_edge_scores = edge_predictor(edges_scores, query_edge_index)
print(predicted_edge_index)
print(predicted_edge_scores)

# %%
query_maker = EdgeQueryExtremitiesDiffCCMaxDistance(max_dist=1.5)
query_edge_index = query_maker(batch_2)
edges_scores = torch.tensor([0.9, 0.95, 0.15, 0.99, 0.60, 0.001], dtype=torch.float)
print(query_edge_index)
print(edges_scores)

edge_predictor = EdgePredictorThresholded(threshold=0.5)
predicted_edge_index, predicted_edge_scores = edge_predictor(edges_scores, query_edge_index)
print(predicted_edge_index)
print(predicted_edge_scores)

# %%
edge_predictor = EdgePredictorMaxThresholded(threshold=0.5)
predicted_edge_index, predicted_edge_scores = edge_predictor(edges_scores, query_edge_index)
print(predicted_edge_index)
print(predicted_edge_scores)

# %%
edge_predictor = EdgePredictorReciprocalMaxThresholded(threshold=0.5)
predicted_edge_index, predicted_edge_scores = edge_predictor(edges_scores, query_edge_index)
print(predicted_edge_index)
print(predicted_edge_scores)

# %%
from PIL import Image
import numpy as np
from graph.graph_creation import img_to_graph

img = Image.open("/home/morand/afs/EVAPORE/data/FIVES/pred/FIVES_001.png").convert("L")
#img = Image.open("/home/morand/afs/EVAPORE/data/FIVES/gt/FIVES_001.png").convert("L")
img = np.array(img)

print(np.unique(img))

graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)
print(graph)

# %%
from graph_neural_networks.data.dataset.dynamic.graph_transforms.oversample_nodes import OversampleNodesTransform
from graph_neural_networks.data.dataset.dynamic.graph_transforms.add_edge_closest_cc import AddEdgeClosestCCTransform
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
import networkx as nx

graph_wrapper = GraphWrapper(graph)
oversample_transform = OversampleNodesTransform(max_dist=100.0, remove_original_edges=True)
virtual_edge_transform = AddEdgeClosestCCTransform(only_one_edge_per_cc=True)

graph_oversampled = oversample_transform(graph_wrapper)
graph_with_virtual_edges = virtual_edge_transform(graph_oversampled)

new_graph = graph_with_virtual_edges.get_graph()
print(new_graph)

mapping = {old_id: new_id for new_id, old_id in enumerate(new_graph.nodes)}
new_graph = nx.relabel_nodes(new_graph, mapping)

# %%
from graph.graph_visualization import display_graph_overlay

display_graph_overlay(img, new_graph, figsize=(50,50))

# %%
from graph_neural_networks.data.utils.networkx import networkx_to_pyg

pyg_data = networkx_to_pyg(new_graph)
print(pyg_data)

# %%
import io
import os.path as osp
import pickle
import re
import sys
import warnings
from typing import Any, Dict, List, Literal, Optional, Union, overload
from uuid import uuid4

import fsspec
import torch

def torch_load(path: str, map_location: Any = None) -> Any:
    try:
        with fsspec.open(path, 'rb') as f:
            return torch.load(f, map_location, weights_only=False)
    except pickle.UnpicklingError as e:
        error_msg = str(e)
        if "add_safe_globals" in error_msg:
            warn_msg = ("Weights only load failed. Please file an issue "
                        "to make `torch.load(weights_only=True)` "
                        "compatible in your case.")
            match = re.search(r'add_safe_globals\(.*?\)', error_msg)
            if match is not None:
                warnings.warn(
                    f"{warn_msg} Please use "
                    f"`torch.serialization.{match.group()}` to "
                    f"allowlist this global.", stacklevel=2)
            else:
                warnings.warn(warn_msg, stacklevel=2)

            with fsspec.open(path, 'rb') as f:
                return torch.load(f, map_location, weights_only=False)
        else:
            raise e


# %%
data_path = "/home/morand/afs/EVAPORE/data/FIVES/processed/dynamic_dataset/dynamic_dataset_21/FIVES_001.pt"
pyg_data = torch_load(data_path)
print(pyg_data)

# %%
from graph_neural_networks.data.edge_splits.no_edge_split import NoEdgeSplit

edge_split = NoEdgeSplit()
split_data = edge_split(pyg_data)

print(split_data)

# %%
batch_3 = Batch(
    **split_data
)

query_maker = EdgeQueryExtremitiesDiffCCMaxDistance(max_dist=200)
query_edge_index = query_maker(batch_3)

print(query_edge_index)
print(query_edge_index.shape)

# %%
