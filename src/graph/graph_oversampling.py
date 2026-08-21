import networkx as nx
import numpy as np

from graph.graph_wrapper import GraphWrapper
from graph.graph_pred_state import EdgePredState

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
                    local_radius = local_radius.tolist()

                    if edge_pred_state is not None:
                        G.add_edge(n0, n1,id=edge_counter, name=name, centerline=local_centerline, radius=local_radius, length=local_length, edge_pred_state=edge_pred_state)
                    else:
                        G.add_edge(n0, n1,id=edge_counter, name=name, centerline=local_centerline, radius=local_radius, length=local_length)
                    edge_counter += 1
                    new_edge_sampled = True

            if self.remove_original_edges and new_edge_sampled:
                G.remove_edge(u, v)
                
        return G, old_nodes, new_nodes, new_nodes_parent_distances

    def __call__(self, graph_wrapper: GraphWrapper) -> GraphWrapper:
        graph = graph_wrapper.get_graph()

        oversampled_graph, old_nodes, new_nodes, new_nodes_parent_distances = self.oversample_graph(graph)
        graph_wrapper.old_nodes = old_nodes
        graph_wrapper.new_nodes = new_nodes
        graph_wrapper.new_nodes_parent_distances = new_nodes_parent_distances

        graph_wrapper.is_oversampled = True

        graph_wrapper.set_graph(oversampled_graph)
        graph_wrapper.setup_pred_state()
        return graph_wrapper


    def _build_config(self) -> dict:
        return {
            "_target_": self.__class__.__name__,
            "max_dist": self.max_dist,
            "remove_original_edges": self.remove_original_edges,
        }