'''
Author: Oscar Morand (LRE, CREATIS)
Date: October 2025
Description: Functions to degrade graphs by removing or modifying edges.
'''

import logging
import numpy as np
import networkx as nx
import enum

from graph.graph_labeling import generate_graph_topological_classification
from graph.graph_random_edge_selection import uniform_edge_selection, depth_edge_selection, topological_edge_selection
from graph.graph_cleaning import merge_two_edges

class EdgeSelectionMode(enum.Enum):
    UNIFORM = 'uniform'
    DEPTH = 'depth'
    TOPOLOGICAL = 'topological'

class EdgeDegradationMode(enum.Enum):
    DELETE = 'delete'
    SPLIT = 'split'
    ERODE = 'erode'


# =============================================================================
# DEGRADATION FUNCTION
# =============================================================================

def delete_edge(graph: nx.MultiGraph, selected_edge: tuple):
    """
    Delete an edge from the graph and remove any isolated nodes.

    Args:
        graph (nx.MultiGraph): The input graph.
        selected_edge (tuple): The edge to delete (u, v).

    Returns:
        nx.MultiGraph: The modified graph with the edge removed.
    """

    G = graph.copy()

    u, v, e = selected_edge
    G.remove_edge(u, v, e)
    
    # Merge neighbour edges if possible
    for node in [u, v]:
        if G.degree(node) == 2:
            G = merge_two_edges(G, node)
            print(f"Merged edges at node {node} before deleting edge {selected_edge}.")

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        G.remove_nodes_from(isolated_nodes)

    return G
    

# =============================================================================
# DICTIONNARIES
# =============================================================================

edge_selection_mode_to_function = {
    EdgeSelectionMode.UNIFORM: uniform_edge_selection,
    EdgeSelectionMode.DEPTH: depth_edge_selection,
    EdgeSelectionMode.TOPOLOGICAL: topological_edge_selection,
}

edge_degradation_mode_to_function = {
    EdgeDegradationMode.DELETE: delete_edge,
}


# =============================================================================
# EDGE DEGRADATION MAIN FUNCTION
# =============================================================================

def edge_degradation(graph: nx.MultiGraph,
                     selection_mode: EdgeSelectionMode = EdgeSelectionMode.UNIFORM,
                     degradation_mode: EdgeDegradationMode = EdgeDegradationMode.DELETE,
                     n_degradation: int = 1,
                     *args
                     ) -> nx.MultiGraph:
    
    """
    Apply edge degradation to the graph.

    Parameters:
        graph (nx.MultiGraph): The input graph.
        selection_mode (EdgeSelectionMode): The edge selection mode.
        degradation_mode (EdgeDegradationMode): The edge degradation mode.
        n_degradation (int): number of time to repeat the function.
        *args: Additional arguments for the selection function.

    Returns:
        nx.MultiGraph: The modified graph after edge degradation.
    """
    
    if graph.number_of_edges() == 0:
        logging.warning("Graph has no edges to delete.")
        return graph
    
    if selection_mode not in edge_selection_mode_to_function:
        logging.warning(f"Unknown edge selection mode: {selection_mode}")
        return graph
    
    if degradation_mode not in edge_degradation_mode_to_function:
        logging.warning(f"Unknown edge degradation mode: {degradation_mode}")
        return graph

    for i in range(n_degradation):
        # Select the edge
        selection_function = edge_selection_mode_to_function[selection_mode]
        selected_edge = selection_function(graph, *args)
        if selected_edge == ():
            break

        # Apply the degradation function
        degradation_function = edge_degradation_mode_to_function[degradation_mode]
        graph = degradation_function(graph, selected_edge)

        graph = generate_graph_topological_classification(graph)

    return graph
