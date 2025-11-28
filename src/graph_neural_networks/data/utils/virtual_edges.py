import networkx as nx
import torch
from graph_neural_networks.data.utils.pred_state import EdgePredState

def setup_non_virtual_edges(graph: nx.Graph) -> nx.Graph:
    """Sets up existing edges in the graph as non-virtual edges.
    Args:
        graph (nx.Graph): A NetworkX graph.
    Returns:
        nx.Graph: the initialized graph
    """

    G = graph.copy()

    for u, v, c in graph.edges:
        if G.get_edge_data(u, v).get('virtual_edge') is None:
            G[u][v][c]['virtual_edge'] = False

    return G
    

def add_virtual_edge(graph: nx.Graph, node1_id, node2_id, edge_id: int = None, length=None) -> nx.Graph:
    """Adds a virtual edge to the graph with specified attributes.

    Args:
        graph: A NetworkX graph.
        virtual_edge_attr: A dictionary of attributes to assign to the virtual edge.

    Returns:
        nx.Graph: the modified graph
    """

    G = graph.copy()

    G.add_edge(node1_id, 
                   node2_id,
                   id=edge_id,
                   name=f"virtual_edge_{edge_id}",
                   virtual_edge=True, 
                   length=length,
                   centerline=[],
                   radius=[],
                   min_radius=None,
                   max_radius=None,
                   mean_radius=None,
                   edge_pred_state=EdgePredState.IN_PREDICTION)
    
    return G

def get_virtual_edges_index(graph: nx.Graph) -> torch.Tensor:
    virtual_edges_index = []
    for src, dst, attrs in graph.edges(data=True):
        if attrs.get('virtual_edge', False):
            virtual_edges_index.append([src, dst])
    return torch.tensor(virtual_edges_index, dtype=torch.long).T