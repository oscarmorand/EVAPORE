import networkx as nx
import numpy as np

from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
from graph_neural_networks.data.utils.pred_state import EdgePredState

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
                if edge_pred_state in [EdgePredState.NOT_IN_PREDICTION, EdgePredState.NOT_IN_PREDICTION.value]:
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
                    name = f"edge_{edge_counter}"
                    local_length = float(lengths[i])
                    local_centerline = centerline[i0:i1 + 1].tolist()
                    local_radius = radius[i0:i1 + 1]
                    min_radius = np.min(local_radius)
                    max_radius = np.max(local_radius)
                    mean_radius = np.mean(local_radius)
                    local_radius = local_radius.tolist()

                    if edge_pred_state is not None:
                        G.add_edge(n0, n1,id=edge_counter, name=name, centerline=local_centerline, radius=local_radius, length=local_length, min_radius=min_radius, max_radius=max_radius, mean_radius=mean_radius, edge_pred_state=edge_pred_state)
                    else:
                        G.add_edge(n0, n1,id=edge_counter, name=name, centerline=local_centerline, radius=local_radius, length=local_length, min_radius=min_radius, max_radius=max_radius, mean_radius=mean_radius)
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
        graph_wrapper.setup_pred_state()
        return graph_wrapper


    def _build_config(self) -> dict:
        return {
            "_target_": self.__class__.__name__,
            "max_dist": self.max_dist,
            "remove_original_edges": self.remove_original_edges,
        }