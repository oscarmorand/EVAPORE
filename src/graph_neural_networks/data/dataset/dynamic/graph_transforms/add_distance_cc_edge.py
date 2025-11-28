import networkx as nx
import numpy as np

from graph_neural_networks.data.dataset.dynamic.graph_transforms.graph_transform import GraphTransform
from graph_neural_networks.data.utils.virtual_edges import add_virtual_edge, setup_non_virtual_edges

class AddDistanceCCEdgeTransform(GraphTransform):
    def __init__(self, 
                 distance_threshold: float = 20.0,
                 cc_condition: bool = True,
                 only_connect_extremities: bool = False,
                 only_keep_closest: bool = False,
                 distance_att: str = "euclidean"):
        """
        Graph transform that adds edges between nodes that are within a certain distance threshold
        and optionally belong to the same connected component (CC).

        Args:
            distance_threshold (float): The maximum distance between nodes to add an edge.
            cc_condition (bool): If True, only add edges between nodes in the same connected component.
        """
        self.distance_threshold = distance_threshold
        self.cc_condition = cc_condition
        self.only_connect_extremities = only_connect_extremities
        self.only_keep_closest = only_keep_closest
        self.distance_att = distance_att

    def get_distance_att(self, distance: float) -> float:
        if self.distance_att == "euclidean":
            return distance
        elif isinstance(self.distance_att, float):
            return self.distance_att
        else:
            return None

    def __call__(self, graph: nx.Graph) -> nx.Graph:
        """
        Apply the transform to the given graph.

        Args:
            graph: The input graph object.

        Returns:
            The transformed graph with additional edges.
        """

        # Classify the connected components if cc_condition is True
        if self.cc_condition:
            connected_components = list(nx.connected_components(graph))
            cc_map = {}
            for cc in connected_components:
                for node in cc:
                    cc_map[node] = cc

        if not all('virtual_edge' in attrs for _, _, attrs in graph.edges(data=True)):
            graph = setup_non_virtual_edges(graph)

        # Add edges based on distance and connected component condition
        for (node1_id, node1_data) in graph.nodes(data=True):
            if self.only_connect_extremities and graph.degree(node1_id) > 1:
                continue
            closest_distance = float('inf')
            closest_node_id = None
            for (node2_id, node2_data) in graph.nodes(data=True):
                if node1_id == node2_id:
                    continue

                if self.only_connect_extremities and graph.degree(node2_id) > 1:
                    continue

                pos1 = np.array(node1_data.get('pos', None))
                pos2 = np.array(node2_data.get('pos', None))

                if pos1 is None or pos2 is None:
                    continue

                distance = np.linalg.norm(pos1 - pos2)

                # Add edge if within distance threshold
                if distance <= self.distance_threshold:
                    if not self.cc_condition or (self.cc_condition and cc_map[node1_id] != cc_map[node2_id]):
                        if self.only_keep_closest:
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_node_id = node2_id
                        else:
                            distance_att = self.get_distance_att(distance)
                            graph = add_virtual_edge(graph, node1_id, node2_id, length=distance_att)
            
            if self.only_keep_closest and closest_node_id is not None:
                distance_att = self.get_distance_att(closest_distance)
                graph = add_virtual_edge(graph, node1_id, closest_node_id, length=distance_att)

        return graph
    
    def _build_config(self) -> dict:
        return {
            "_target_": self.__class__.__name__,
            "distance_threshold": self.distance_threshold,
            "cc_condition": self.cc_condition,
            "only_connect_extremities": self.only_connect_extremities,
            "distance_att": self.distance_att,
        }