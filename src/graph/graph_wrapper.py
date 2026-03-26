import networkx as nx
import numpy as np

from graph.graph_pred_state import EdgePredState

class GraphWrapper():
    def __init__(self, 
                 graph: nx.Graph,
                 is_oversampled: bool = False,
                 initialized_edge_attributes: set = None,
                 old_nodes: list = None,
                 new_nodes: list = None,
                 new_nodes_parent_distances: dict = None):
        self.graph = graph

        self.is_oversampled = is_oversampled
        self.old_nodes = old_nodes
        self.new_nodes = new_nodes
        self.new_nodes_parent_distances = new_nodes_parent_distances

        self.initialized_edge_attributes = initialized_edge_attributes or set()

        self.in_pred_edges = set()
        self.not_in_pred_edges = set()
        self.in_pred_graph = None

    def get_graph(self) -> nx.Graph:
        return self.graph
    
    def set_graph(self, graph: nx.Graph) -> None:
        self.graph = graph

    def all_edges_have_attribute(self, attr_name: str) -> bool:
        if attr_name in self.initialized_edge_attributes:
            return True
        all_edges_have_attr = all(attr_name in attrs for _, _, attrs in self.graph.edges(data=True))
        if all_edges_have_attr:
            self.initialized_edge_attributes.add(attr_name)
        return all_edges_have_attr

    def setup_pred_state(self) -> None:
        if not self.all_edges_have_attribute('edge_pred_state'):
            return
        
        self.in_pred_graph = nx.MultiGraph()
        for u, v, data in self.graph.edges(data=True):
            pred_state = data['edge_pred_state']
            if pred_state in [EdgePredState.IN_PREDICTION.value, EdgePredState.IN_PREDICTION]:
                self.in_pred_edges.add((u, v))

                if u not in self.in_pred_graph:
                    self.in_pred_graph.add_node(u, **self.graph.nodes[u])
                if v not in self.in_pred_graph:
                    self.in_pred_graph.add_node(v, **self.graph.nodes[v])
                self.in_pred_graph.add_edge(u, v, **data)

            elif pred_state in [EdgePredState.NOT_IN_PREDICTION.value, EdgePredState.NOT_IN_PREDICTION]:
                self.not_in_pred_edges.add((u, v))
