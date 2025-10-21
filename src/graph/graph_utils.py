'''
Author: Oscar Morand (LRE, CREATIS)
Date: October 2025
Description: Utility functions for graph processing.
'''

import networkx as nx
import logging
import enum
import numpy as np


logger = logging.getLogger(__name__)


class TopologicalClass(enum.Enum):
    NON_TOPOLOGICAL = 'non_topological'
    T0 = 't0'
    T1 = 't1'


def find_max_radius_node(graph: nx.MultiGraph) -> int:
    """
    Finds the node with the maximum radius attribute in the graph.
    Parameters:
        graph (nx.MultiGraph): Input vascular graph.
    Returns:
        int: Node ID with the maximum radius.
    """

    max_radius = 0
    max_node = None

    for n in graph.nodes(data=True):
        radius = n[1].get('radius', 0)
        if radius > max_radius:
            max_radius = radius
            max_node = n[0]

    return max_node


def get_parallel_edges(graph: nx.MultiGraph) -> set:
    """
    Get all edges connected to a node that have parallel edges.

    Args:
        graph (nx.MultiGraph): The input graph.
        node (int): The node to check for parallel edges.

    Returns:
        set: A set of edges that have parallel edges.
    """
    parallel_edges = set()
    for n in graph.nodes():
        for m in graph.neighbors(n):
            n_edges = graph.number_of_edges(n, m)
            if n_edges > 1 and (n, m) not in parallel_edges and (m, n) not in parallel_edges:
                parallel_edges.add((n,m))
    return parallel_edges


def get_cycles_edges(graph: nx.MultiGraph) -> set:
    """
    Identify edges that are part of cycles in the graph.

    Parameters:
        graph (nx.MultiGraph): Input vascular graph.

    Returns:
        set: A set of edges (tuples) that are part of cycles.
    """

    if graph.number_of_edges() == 0:
        logging.info("Graph has no edges to delete.")
        return graph
    
    cycles_edges = set()

    # Include self-loops as cycle edges
    cycles_edges.update(nx.selfloop_edges(graph))

    # Include parallel edges as cycle edges
    parallel_edges = get_parallel_edges(graph)
    cycles_edges.update(parallel_edges)
                
    # Find all cycles in the simple version of the graph
    # Convert to simple graph for cycle detection
    G_simpl = nx.Graph(graph)
    cycles = list(nx.cycle_basis(G_simpl))
    
    for cycle in cycles:
        for i in range(len(cycle)):
            n_0, n_1 = cycle[i], cycle[(i + 1) % len(cycle)]
            cycles_edges.add((n_0, n_1))

    return cycles_edges


def get_terminal_edges(graph: nx.MultiGraph) -> set:
    """
    Identify edges that are terminal (connected to nodes with degree 1).

    Parameters:
        graph (nx.MultiGraph): Input vascular graph.

    Returns:
        set: A set of edges (tuples) that are terminal edges.
    """

    terminal_edges = set()
    for node in graph.nodes():
        if graph.degree(node) == 1:
            neighbor = list(graph.neighbors(node))[0]
            if graph.degree(neighbor) > 1:
                terminal_edges.add((node, neighbor))

    if terminal_edges is None or len(terminal_edges) == 0:
        logging.info("No terminal edges found in the graph.")

    return terminal_edges