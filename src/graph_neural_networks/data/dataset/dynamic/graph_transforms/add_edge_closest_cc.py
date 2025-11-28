import networkx as nx
import numpy as np

from graph_neural_networks.data.dataset.dynamic.graph_transforms.graph_transform import GraphTransform
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
from graph_neural_networks.data.utils.virtual_edges import add_virtual_edge, setup_non_virtual_edges

class AddEdgeClosestCCTransform(GraphTransform):
    def __init__(self, only_one_edge_per_cc: bool = True):
        self.only_one_edge_per_cc = only_one_edge_per_cc

    def add_edge_closest_cc(self, graph: nx.Graph, in_pred_graph: nx.Graph = None) -> nx.Graph:
        G = graph.copy()
        in_pred_G = (in_pred_graph.copy()) if in_pred_graph is not None else None

        edge_counter = max([data['id'] for _, _, data in graph.edges(data=True)]) + 1

        # Classify the connected components if cc_condition is True
        connected_components = list(nx.connected_components(graph if in_pred_graph is None else in_pred_graph))

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
                        G = add_virtual_edge(G, node1_id, closest_node_id, edge_id=edge_counter, length=distance_att)
                        if in_pred_G is not None:
                            in_pred_G = add_virtual_edge(in_pred_G, node1_id, closest_node_id)
                        edge_counter += 1

                if self.only_one_edge_per_cc:
                    G = add_virtual_edge(G, global_closest_node1_id, global_closest_node2_id, edge_id=edge_counter, length=global_closest_distance)
                    if in_pred_G is not None:
                        in_pred_G = add_virtual_edge(in_pred_G, global_closest_node1_id, global_closest_node2_id)
                    edge_counter += 1

            connected_components = list(nx.connected_components(G if in_pred_G is None else in_pred_G))

        return G

    def __call__(self, graph_wrapper: GraphWrapper) -> GraphWrapper:
        """
        Adds virtual edges between the closest nodes of different connected components in the graph.

        Args:
            graph_wrapper (GraphWrapper): A wrapper containing the graph to be transformed.

        Returns:
            GraphWrapper: The transformed graph wrapper with added virtual edges.
        """

        if not graph_wrapper.all_edges_have_attribute('virtual_edge'):
            graph_wrapper.setup_non_virtual_edges()

        graph = graph_wrapper.get_graph()
        transformed_graph = self.add_edge_closest_cc(graph, graph_wrapper.in_pred_graph)

        graph_wrapper.set_graph(transformed_graph)
        return graph_wrapper
        

    def _build_config(self) -> dict:
        return {
            "_target_": self.__class__.__name__,
            "only_one_edge_per_cc": self.only_one_edge_per_cc,
        }