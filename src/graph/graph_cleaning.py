'''
Authors: Oscar Morand (LRE, CREATIS), Morgane Des-Ligneris (CREATIS)
Date: October 2025
Description: Functions for cleaning vascular graphs
'''

import logging
import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def remove_degree_2_nodes(graph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Simplify a NetworkX graph by removing all nodes with degree 2,
    connecting their neighbors directly.

    Args:
        graph (networkx.MultiGraph): Input graph

    Returns:
        simplified_graph (networkx.MultiGraph): Simplified graph with degree-2 nodes removed
    """

    G = graph.copy()

    # Find all degree-2 nodes
    nodes_to_remove = [n for n in G.nodes if G.degree(n) == 2]

    for node in nodes_to_remove:

        # Get the two neighbors of the degree-2 node
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 1:
            neighbors = [neighbors[0], neighbors[0]]

        # Get edge data for both edges connected to the degree-2 node
        edge_0 = G.get_edge_data(node, neighbors[0])[0]
        edge_1 = G.get_edge_data(node, neighbors[1])[0]

        # Extract properties from both edges
        centerline_0, radius_0, length_0 = edge_0["centerline"], edge_0["radius"], edge_0["length"]
        centerline_1, radius_1, length_1 = edge_1["centerline"], edge_1["radius"], edge_1["length"]

        # Use the ID of the first edge
        id = edge_0["id"]

        # Combine properties
        centerline = centerline_0 + centerline_1
        radius = radius_0 + radius_1
        length = length_0 + length_1

        # Add new edge connecting the two neighbors directly
        G.add_edge(
            neighbors[0],
            neighbors[1],
            id=id,
            name=f"edge_{id}",
            centerline=centerline,
            radius=radius,
            length=length,
            min_radius=min(radius),
            max_radius=max(radius),
            mean_radius=float(np.mean(radius)) if radius else 0.0,
        )

        # Remove the degree-2 node and its edges
        G.remove_edge(node, neighbors[0])
        G.remove_edge(node, neighbors[1])
        G.remove_node(node)
        
    return G


def remove_small_self_loops(graph: nx.MultiGraph,
                            min_length: float = 5.0
                            ) -> nx.MultiGraph:
    """
    Remove self-loops in the graph that are shorter than a specified minimum length.

    Args:
        graph (networkx.MultiGraph): Input graph
        min_length (float): Minimum length threshold for self-loops to be retained

    Returns:
        cleaned_graph (networkx.MultiGraph): Graph with short self-loops removed
    """

    G = graph.copy()

    # Identify and remove self-loops shorter than min_length
    self_loops = [(u, v) for u, v, d in graph.edges(data=True) if u == v]

    for u, v in self_loops:
        length = G.get_edge_data(u, v)[0]['length']

        # Remove the self-loop if its length is less than the threshold
        if length < min_length:
            logger.info("Removing self-loop at node %d with length %f", u, length)
            G.remove_edge(u, v)

    return G