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
import networkx as nx

adj_matrix = nx.adjacency_matrix(graph)
print(adj_matrix.shape)

# %%
print(adj_matrix)

# %%
full_adj_matrix = nx.to_numpy_array(graph)
plt.imshow(full_adj_matrix, cmap='gray')
plt.show()

# %%
N = full_adj_matrix.shape[0]

M_bis = full_adj_matrix.copy()
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        if full_adj_matrix[i, j] == 0:
            continue
        new_i = j
        for new_j in range(N):
            if new_j == i:
                continue
            if new_i == new_j:
                continue
            if full_adj_matrix[new_i, new_j] == 1:
                if full_adj_matrix[i, new_j] == 0:
                    print(f"Path found: {i} -> {j} -> {new_j}")
                    M_bis[i, new_j] = 1

# %%
plt.imshow(M_bis, cmap='hot')


# %%
def propagate_connections(adj_matrix, steps=5):
    N = adj_matrix.shape[0]
    M = np.ones_like(adj_matrix) * -1.0
    M[adj_matrix > 0] = 1.0
    M = M * (1 - np.eye(N))
    plt.figure(figsize=(10,10))
    plt.imshow(M, interpolation='none')
    plt.colorbar()
    plt.show()
    for step_i in range(steps):
        M_temp = M.copy()
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if M[i, j] == -1:
                    continue
                new_i = j
                for new_j in range(N):
                    if new_j == i:
                        continue
                    if new_i == new_j:
                        continue
                    if M[new_i, new_j] > 0:
                        if M[i, new_j] == -1:
                            M_temp[i, new_j] = M[i, j] + M[new_i, new_j]
        if np.array_equal(M, M_temp):
            print(f"No more updates at step {step_i}, stopping propagation.")
            break
        M = M_temp
        plt.figure(figsize=(10,10))
        plt.imshow(M, interpolation='none')
        plt.colorbar()
        plt.show()
    return M


# %%
propagate_connections(full_adj_matrix, steps=10)


# %%
def propagate_connections(adj_matrix, max_steps=None, plot=True):
    N = adj_matrix.shape[0]
    if max_steps is None or max_steps > N:
        max_steps = N
    M = adj_matrix.copy()
    if plot:
        plt.figure(figsize=(10,10))
        plt.imshow(M, interpolation='none')
        plt.colorbar()
        plt.show()
    for step_i in range(max_steps):
        M_temp = M.copy()
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if M[i, j] == -1:
                    continue
                new_i = j
                for new_j in range(N):
                    if new_j == i:
                        continue
                    if new_i == new_j:
                        continue
                    if M[new_i, new_j] > 0:
                        new_dist = M[i, j] + M[new_i, new_j]
                        if M[i, new_j] == -1 or new_dist < M[i, new_j]:
                            M_temp[i, new_j] = new_dist
        if np.array_equal(M, M_temp):
            print(f"No more updates at step {step_i}, stopping propagation.")
            break
        M = M_temp
        if plot:
            plt.figure(figsize=(10,10))
            plt.imshow(M, interpolation='none')
            plt.colorbar()
            plt.show()
    return M


# %%
mini_adj_matrix = np.array([
    [0, 1, -1],
    [1, 0, 2],
    [-1, 2, 0]
])

M = propagate_connections(mini_adj_matrix, max_steps=3)


# %%
def graph_to_length_matrix(graph):
    N = graph.number_of_nodes()
    length_matrix = np.zeros((N, N)) - 1
    length_matrix = length_matrix + np.eye(length_matrix.shape[0])
    nodes_to_index = {n: i for i, n in enumerate(graph.nodes())}
    for n_1, n_2, data in graph.edges(data=True):
        l = data['length']
        i, j = nodes_to_index[n_1], nodes_to_index[n_2]
        length_matrix[i, j] = l
        length_matrix[j, i] = l
    return length_matrix


length_matrix = graph_to_length_matrix(graph)
plt.figure(figsize=(10,10))
plt.imshow(length_matrix)
plt.colorbar()
plt.show()

M = propagate_connections(length_matrix)

# %%
a = 0.01
y = np.exp(- a * M)

plt.figure(figsize=(10,10))
plt.imshow(y)
plt.colorbar()
plt.show()

# %%
test_matrix = np.array([
    [0, 3, -1, -1, -1],
    [3, 0, 1, -1, 1],
    [-1, 1, 0, 2, -1],
    [-1, -1, 2, 0, -1],
    [-1, 1, -1, -1, 0]
])

M_test = propagate_connections(test_matrix, plot=False)

print(M_test)

# %%
test_cycle_matrix = np.array([
    [0, 2, -1, -1, -1, -1],
    [2, 0, 1, -1, -1, 1],
    [-1, 1, 0, 1, -1, -1],
    [-1, -1, 1, 0, 4, -1],
    [-1, -1, -1, 4, 0, 2],
    [-1, 1, -1, -1, 2, 0]
])

M_cycle = propagate_connections(test_cycle_matrix, plot=False)

print(M_cycle)


# %%
def get_y(M, a=0.01):
    return np.exp(- a * M)

y_test = get_y(M_test, a=0.5)

plt.figure(figsize=(10,10))
plt.imshow(y_test)
plt.colorbar()
plt.show()


# %%
def plot_geodesic_distance_histogram(graph, max_steps=10, bins=50):
    length_matrix = graph_to_length_matrix(graph)
    geodesic_matrix = propagate_connections(length_matrix, max_steps=max_steps, plot=False)
    hist, bin_edges = np.histogram(geodesic_matrix[geodesic_matrix > 0], bins=bins)
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges))
    plt.xlabel('Geodesic Distance')
    plt.ylabel('Frequency')
    plt.title('Geodesic Distance Histogram')
    plt.show()



# %%
histo = plot_geodesic_distance_histogram(graph, max_steps=10)

# %%
import networkx as nx
import numpy as np

class GraphWrapper():
    def __init__(self, 
                 graph: nx.Graph, 
                 distance_matrix: np.ndarray = None,
                 root_node: int = None):
        self.graph = graph
        self.distance_matrix = distance_matrix
        self.root_node = root_node

    def get_distance(self, node1: int, node2: int) -> float:
        if self.distance_matrix is None:
            raise ValueError("Distance matrix is not defined.")
        return self.distance_matrix[node1, node2]
    
    def are_node_connected(self, node1: int, node2: int) -> bool:
        if self.distance_matrix is None:
            raise ValueError("Distance matrix is not defined.")
        return self.distance_matrix[node1, node2] != -1
        

class OversampleNodesTransform:
    def __init__(self,
                 max_dist: float = 10.0,
                 remove_original_edges: bool = True):
        self.max_dist = max_dist
        self.remove_original_edges = remove_original_edges

    def __call__(self, graph: nx.Graph) -> nx.Graph:
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

                if n != len(split_points) + 1:
                    warnings.warn(f"Expected {n-1} split points, but got {len(split_points)}.")

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
                    )
                    edge_counter += 1
                    new_edge_sampled = True

            if self.remove_original_edges and new_edge_sampled:
                G.remove_edge(u, v)
                
        return G, old_nodes, new_nodes, new_nodes_parent_distances

    def _build_config(self) -> dict:
        return {
            "_target_": self.__class__.__name__,
            "max_dist": self.max_dist,
            "remove_original_edges": self.remove_original_edges,
        }


# %%
graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)
print(graph)

# %%
new_oversample_transform = OversampleNodesTransform(max_dist=20)

oversampled_graph, old_nodes, new_nodes, new_nodes_parent_distances = new_oversample_transform(graph)

# %%
from graph.graph_visualization import display_graph_overlay

display_graph_overlay(img, oversampled_graph, figsize=(20,20))

# %%
print(len(old_nodes), "+", len(new_nodes), "=", len(old_nodes) + len(new_nodes))

new_N = oversampled_graph.number_of_nodes()
new_distance_matrix = np.ones((new_N, new_N)) * -1

print(new_distance_matrix.shape)

# %%
print(new_nodes_parent_distances)

# %%
distance_matrix = M

new_distance_matrix[0:N, 0:N] = distance_matrix

plt.figure(figsize=(10,10))
plt.imshow(new_distance_matrix)
plt.colorbar()
plt.show()

# %%

oti = {n: i for i, n in enumerate(old_nodes)}

for i, new_node in enumerate(new_nodes):
    (p1, d1), (p2, d2) = new_nodes_parent_distances[new_node]
    for j, old_node in enumerate(old_nodes):
        if old_node == p1:
            new_distance_matrix[N + i, j] = d1
            new_distance_matrix[j, N + i] = d1
        elif old_node == p2:
            new_distance_matrix[N + i, j] = d2
            new_distance_matrix[j, N + i] = d2
        else:
            closest_parent = p1
            dist_to_closest_parent = d1
            p1_to_old = distance_matrix[oti[old_node], oti[p1]]
            p2_to_old = distance_matrix[oti[old_node], oti[p2]]
            if p1_to_old == -1 and p2_to_old == -1:
                new_distance_matrix[N + i, j] = -1
                new_distance_matrix[j, N + i] = -1
                continue
            if (p2_to_old + d2 < p1_to_old + d1) or p1_to_old == -1:
                closest_parent = p2
                dist_to_closest_parent = d2
            new_distance_matrix[N + i, j] = dist_to_closest_parent + distance_matrix[oti[old_node], oti[closest_parent]]
            new_distance_matrix[j, N + i] = new_distance_matrix[N + i, j]

for i, new_node in enumerate(new_nodes):
    (p1, d1), (p2, d2) = new_nodes_parent_distances[new_node]
    for j, other_new_node in enumerate(new_nodes):
        if i == j:
            new_distance_matrix[N + i, N + j] = 0.0
            new_distance_matrix[N + j, N + i] = 0.0
            continue
        other_new_to_p1 = new_distance_matrix[N + j, oti[p1]]
        other_new_to_p2 = new_distance_matrix[N + j, oti[p2]]
        if other_new_to_p1 == -1 and other_new_to_p2 == -1:
            new_distance_matrix[N + i, N + j] = -1
            new_distance_matrix[N + j, N + i] = -1
            continue
        closest_parent = p1
        dist_to_closest_parent = d1
        if (other_new_to_p2 + d2 < other_new_to_p1 + d1) or other_new_to_p1 == -1:
            closest_parent = p2
            dist_to_closest_parent = d2
        new_distance_matrix[N + i, N + j] = dist_to_closest_parent + new_distance_matrix[N + j, oti[closest_parent]]
        new_distance_matrix[N + j, N + i] = new_distance_matrix[N + i, N + j]
        

plt.figure(figsize=(10,10))
plt.imshow(new_distance_matrix)
plt.colorbar()
plt.show()

# %%
img = np.array(Image.open("/home/morand/afs/tests/test_oversampling_length_matrix.png"))[:,:,0]

graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)

print(graph)

# %%
display_graph_overlay(img, graph, figsize=(10,10))

# %%
transform = OversampleNodesTransform(max_dist=10)

oversampled_graph, old_nodes, new_nodes, new_nodes_parent_distances = transform(graph)

# %%
display_graph_overlay(img, oversampled_graph, figsize=(10,10))

# %%
distance_matrix = graph_to_length_matrix(graph)

print(distance_matrix)

plt.imshow(distance_matrix)
plt.colorbar()
plt.show()

# %%
distance_matrix = propagate_connections(distance_matrix, plot=False)

print(distance_matrix)

plt.imshow(distance_matrix)
plt.colorbar()
plt.show()

# %%
brutforce_oversampled_distance_matrix = graph_to_length_matrix(oversampled_graph)
plt.imshow(brutforce_oversampled_distance_matrix)
plt.colorbar()
plt.show()

brutforce_oversampled_distance_matrix = propagate_connections(brutforce_oversampled_distance_matrix, plot=False)
plt.imshow(brutforce_oversampled_distance_matrix)
plt.colorbar()
plt.show()

# %%
for k, v in new_nodes_parent_distances.items():
    print(f"Node {k}: Parents {v}")

# %%
import time

start_time = time.time()

N = distance_matrix.shape[0]
new_N = oversampled_graph.number_of_nodes()
new_distance_matrix = np.ones((new_N, new_N)) * -1
new_distance_matrix[0:N, 0:N] = distance_matrix

print(new_distance_matrix)
plt.imshow(new_distance_matrix)
plt.colorbar()
plt.show()

oti = {n: i for i, n in enumerate(old_nodes)}

for i, new_node in enumerate(new_nodes):
    (p1, d1), (p2, d2) = new_nodes_parent_distances[new_node]
    for j, old_node in enumerate(old_nodes):
        if old_node == p1:
            new_distance_matrix[N + i, j] = d1
            new_distance_matrix[j, N + i] = d1
        elif old_node == p2:
            new_distance_matrix[N + i, j] = d2
            new_distance_matrix[j, N + i] = d2
        else:
            closest_parent = p1
            dist_to_closest_parent = d1
            p1_to_old = distance_matrix[oti[old_node], oti[p1]]
            p2_to_old = distance_matrix[oti[old_node], oti[p2]]
            if p1_to_old == -1 and p2_to_old == -1:
                new_distance_matrix[N + i, j] = -1
                new_distance_matrix[j, N + i] = -1
                continue
            if (p2_to_old + d2 < p1_to_old + d1) or p1_to_old == -1:
                closest_parent = p2
                dist_to_closest_parent = d2
            new_distance_matrix[N + i, j] = dist_to_closest_parent + distance_matrix[oti[old_node], oti[closest_parent]]
            new_distance_matrix[j, N + i] = new_distance_matrix[N + i, j]

print(new_distance_matrix)
plt.imshow(new_distance_matrix)
plt.colorbar()
plt.show()

for i, new_node in enumerate(new_nodes):
    (p1, d1), (p2, d2) = new_nodes_parent_distances[new_node]
    for j, other_new_node in enumerate(new_nodes):
        if i == j:
            new_distance_matrix[N + i, N + j] = 0.0
            new_distance_matrix[N + j, N + i] = 0.0
            continue
        (other_p1, other_d1), (other_p2, other_d2) = new_nodes_parent_distances[other_new_node]
        if p1 == other_p1 and p2 == other_p2:
            inter_dist = abs(other_d1 - d1)
            new_distance_matrix[N + i, N + j] = inter_dist
            new_distance_matrix[N + j, N + i] = inter_dist
            continue
        other_new_to_p1 = new_distance_matrix[N + j, oti[p1]]
        other_new_to_p2 = new_distance_matrix[N + j, oti[p2]]
        if other_new_to_p1 == -1 and other_new_to_p2 == -1:
            new_distance_matrix[N + i, N + j] = -1
            new_distance_matrix[N + j, N + i] = -1
            continue
        closest_parent = p1
        dist_to_closest_parent = d1
        if (other_new_to_p2 + d2 < other_new_to_p1 + d1) or other_new_to_p1 == -1:
            closest_parent = p2
            dist_to_closest_parent = d2
        new_distance_matrix[N + i, N + j] = dist_to_closest_parent + new_distance_matrix[N + j, oti[closest_parent]]
        new_distance_matrix[N + j, N + i] = new_distance_matrix[N + i, N + j]
        
print(new_distance_matrix)
plt.imshow(new_distance_matrix)
plt.colorbar()
plt.show()

end_time = time.time()

print(f"Oversampled distance matrix computed in {end_time - start_time:.4f} seconds.")

# %%
oversampled_y = get_y(new_distance_matrix, a=0.1)
plt.imshow(oversampled_y)
plt.colorbar()
plt.show()

# %%
import time

start_time = time.time()

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

end_time = time.time()

print(new_distance_matrix)

print(f"Oversampled distance matrix computed in {end_time - start_time:.4f} seconds.")


# %%
def gaussian_dist(distance_matrix, a=0.01):
    return np.exp(-(np.square(distance_matrix)) / (2 * a**2))


# %%
y = gaussian_dist(new_distance_matrix, a=5)

plt.imshow(y)
plt.colorbar()
plt.show()


# %%
def compute_distance_matrix_by_graph_breadth_first(graph: nx.Graph, skip_first_neighbor: bool) -> np.ndarray:
    N = graph.number_of_nodes()
    distance_matrix = np.ones((N, N)) * -1
    nodes_to_index = {n: i for i, n in enumerate(graph.nodes())}
    for start_node in graph.nodes():
        start_index = nodes_to_index[start_node]
        #distance_matrix[start_index, start_index] = 0.0
        visited = set()
        queue = [(start_node, 0.0, 0)]
        while queue:
            current_node, current_dist, depth_level = queue.pop(0)
            current_index = nodes_to_index[current_node]

            if distance_matrix[start_index, current_index] == -1 or current_dist < distance_matrix[start_index, current_index]:
                distance_matrix[start_index, current_index] = current_dist
                distance_matrix[current_index, start_index] = current_dist

            for neighbor in graph.neighbors(current_node):
                edge_data = graph.get_edge_data(current_node, neighbor)
                min_length = float('inf')
                min_e = None
                for e, data in edge_data.items():
                    edge_length = data['length']
                    if edge_length < min_length:
                        min_length = edge_length
                        min_e = e
                if min_e is not None:
                    edge_length = edge_data[min_e]['length']
                    new_dist = current_dist + edge_length
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if skip_first_neighbor and depth_level == 0:
                            queue.append((neighbor, 0.0, 1))
                        else:
                            queue.append((neighbor, new_dist, depth_level + 1))
    return distance_matrix


# %%
import numpy as np
import networkx as nx
import heapq

def compute_distance_matrix_by_graph_breadth_first(graph: nx.Graph, skip_first_neighbor: bool) -> np.ndarray:
    N = graph.number_of_nodes()
    distance_matrix = np.ones((N, N)) * -1
    nodes_to_index = {n: i for i, n in enumerate(graph.nodes())}

    # Use Dijkstra for each start node
    for start_node in graph.nodes():
        start_index = nodes_to_index[start_node]

        # Min heap for priority queue
        pq = [(0.0, start_node, 0)]  # (distance, node, depth_level)
        distances = {n: float('inf') for n in graph.nodes()}
        distances[start_node] = 0.0

        while pq:
            current_dist, current_node, depth_level = heapq.heappop(pq)
            current_index = nodes_to_index[current_node]

            # If already found a shorter path skip
            if current_dist > distances[current_node]:
                continue

            # Write symmetric distances
            distance_matrix[start_index, current_index] = current_dist
            if not skip_first_neighbor:
                distance_matrix[current_index, start_index] = current_dist

            for neighbor in graph.neighbors(current_node):
                edge_data = graph.get_edge_data(current_node, neighbor)

                if skip_first_neighbor and depth_level == 0:
                    min_length = 0.0
                else:
                    # Find shortest edge between the two nodes
                    min_length = min(d['length'] for d in edge_data.values())
                    
                new_dist = current_dist + min_length

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor, depth_level + 1))

    return distance_matrix


# %%


all_edges_data = graph.edges(data=True)
for u, v, data in all_edges_data:
    print(f"Edge {u}-{v}: length = {data['length']}")

# %%
distance_matrix_bis = compute_distance_matrix_by_graph_breadth_first(graph, skip_first_neighbor=False)

# %%
print(distance_matrix_bis)

# %%
img = np.array(Image.open("/home/morand/afs/EVAPORE/data/FIVES/gt/1_A.png"))[:,:,0]
graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)

# %%
distance_matrix_bis = compute_distance_matrix_by_graph_breadth_first(graph, skip_first_neighbor=False)
plt.imshow(distance_matrix_bis)
plt.colorbar()
plt.show()

distance_matrix_bis_bis = compute_distance_matrix_by_graph_breadth_first(graph, skip_first_neighbor=True)
plt.imshow(distance_matrix_bis_bis)
plt.colorbar()
plt.show()
print(distance_matrix_bis_bis)

diff = distance_matrix_bis - distance_matrix_bis_bis
plt.imshow(diff)
plt.colorbar()
plt.show()

# %%
import os

dir = "/home/morand/afs/EVAPORE/data/FIVES/gt/"

gt_names = os.listdir(dir)
gt_paths = [os.path.join(dir, n) for n in gt_names if n.endswith('.png')]


# %%
def benchmark_methods(graph):
    # Methode 1, matrix propagation
    start_time = time.time()
    length_matrix = graph_to_length_matrix(graph)
    geodesic_matrix = propagate_connections(length_matrix, plot=False)
    end_time = time.time()
    print(f"Matrix propagation time: {end_time - start_time:.4f} seconds")

    # Methode 2, breadth-first search
    start_time = time.time()
    geodesic_matrix_bis = compute_distance_matrix_by_graph_breadth_first(graph)
    end_time = time.time()
    print(f"Breadth-first search time: {end_time - start_time:.4f} seconds")

    # Verify that both methods yield the same result
    if np.allclose(geodesic_matrix, geodesic_matrix_bis):
        print("Both methods yield the same geodesic distance matrix.")
        print(geodesic_matrix)
    else:
        print("Discrepancy found between the two methods' results.")
        print(geodesic_matrix)
        print(geodesic_matrix_bis)

        plt.imshow(geodesic_matrix - geodesic_matrix_bis)
        plt.colorbar()
        plt.show()

    plt.imshow(geodesic_matrix)
    plt.colorbar()
    plt.show()

    plt.imshow(geodesic_matrix_bis)
    plt.colorbar()
    plt.show()


# %%
img = np.array(Image.open("/home/morand/afs/tests/test_oversampling_length_matrix.png"))[:,:,0]

graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)

print(graph)

benchmark_methods(graph)

# %%
transform = OversampleNodesTransform(max_dist=10)

oversampled_graph, old_nodes, new_nodes, new_nodes_parent_distances = transform(graph)

benchmark_methods(oversampled_graph)

# %%
img = np.array(Image.open("/home/morand/afs/tests/simple_loop_two.png"))[:,:,0]
graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)

display_graph_overlay(img, graph, figsize=(10,10)) 
benchmark_methods(graph)

# %%
img = np.array(Image.open("/home/morand/afs/tests/New Piskel.png"))[:,:,0]
graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)

benchmark_methods(graph)

# %%
import time

gt_paths = gt_paths[:5]  # limit to first 5 for testing

for gt_path in gt_paths:
    print(f"Processing {gt_path}...")
    img = np.array(Image.open(gt_path))[:,:,0]
    graph = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)

    benchmark_methods(graph)

# %%
