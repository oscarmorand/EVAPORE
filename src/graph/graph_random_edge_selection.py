'''
Author: Oscar Morand (LRE, CREATIS)
Date: October 2025
Description: Functions to randomly select edges in a graph based on different criteria (uniform, depth-based, topological class).
'''

import networkx as nx
import logging
import numpy as np

from graph.graph_labeling import generate_graph_depth, generate_graph_topological_classification
from graph.graph_utils import TopologicalClass

def uniform_edge_selection(graph: nx.MultiGraph) -> tuple:
    """Select a random edge from the graph uniformly.

    Args:
        graph (nx.MultiGraph): The input graph.

    Returns:
        tuple: A tuple (u, v, e) where u and v are the endpoints of the selected edge
               and e is the edge key (for MultiGraph).
    """
    n_edges = graph.number_of_edges()
    random_index = np.random.randint(0, n_edges)
    u, v, data = list(graph.edges(data=True))[random_index]
    id = data['id']

    e = [key for key, val in graph.get_edge_data(u, v).items() if id == val['id']][0]
    return u, v, e

def depth_edge_selection(graph: nx.MultiGraph,
                         depth: int
                         ) -> tuple:

    if not any('depth' in d for _, _, d in graph.edges(data=True)):
        logging.warning("No edges have 'depth' attribute, computing graph depth...")
        graph = generate_graph_depth(graph)

    # Filter edges by depth level
    all_edges = [(u, v) for u, v, d in graph.edges(data=True)]
    edges_id_in_level = [i for i, (u, v, d) in enumerate(graph.edges(data=True)) if d.get('depth') == depth]

    if not edges_id_in_level:
        logging.warning(f"No edges found at depth level {depth}.")
        return graph

    # Randomly select an edge to remove
    edge_id = np.random.choice(edges_id_in_level)
    selected_edge = all_edges[edge_id]

    return selected_edge

def topological_edge_selection(graph: nx.MultiGraph, topological_class: TopologicalClass) -> tuple:
    if not any('topological_class' in d for _, _, d in graph.edges(data=True)):
        logging.warning("No edges have 'topological_class' attribute, computing classification...")
        graph = generate_graph_topological_classification(graph)
    
    # Filter edges by topological class
    all_edges = [(u, v) for u, v, d in graph.edges(data=True)]
    edges_id_in_class = [i for i, (u, v, d) in enumerate(graph.edges(data=True)) if d.get('topological_class') == topological_class]

    # Randomly select an edge to remove
    edge_id = np.random.choice(edges_id_in_class)
    selected_edge = all_edges[edge_id]

    return selected_edge