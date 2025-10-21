'''
Authors: Oscar Morand (LRE, CREATIS), Morgane Des-Ligneris (CREATIS)
Date: October 2025
Description: Functions for cleaning vascular graphs
'''

import logging
import networkx as nx
import numpy as np


logger = logging.getLogger(__name__)


def merge_centerlines(c0: list, c1: list) -> list:
    """
    Merge two centerlines ensuring correct orientation.

    Args:
        c1 (list): First centerline (list of points).
        c2 (list): Second centerline (list of points).

    Returns:
        list: Merged centerline.
    """

    if c0[-1] != c1[0]:
        if c0[-1] == c1[-1]:
            c1 = c1[::-1]
        else:
            c0 = c0[::-1]
            if c0[-1] != c1[0]:
                c1 = c1[::-1]

    return c0 + c1

def merge_two_edges(graph: nx.MultiGraph,
                    node: int
                    ) -> nx.MultiGraph:
    """
    Merge two edges connected to a degree-2 node into a single edge.

    Args:
        graph (nx.MultiGraph): The input graph.
        node (int): The degree-2 node to merge.

    Returns:
        nx.MultiGraph: The graph with the merged edge.
    """
    G = graph.copy()

    node_degree = G.degree(node)
    if node_degree != 2:
        logging.warning("The node {node} has a degree of {node_degree}, the merging is not possible")
        return G

    # Get the two neighbors of the degree-2 node
    neighbors = list(G.neighbors(node))
    if len(neighbors) == 1:
        if neighbors[0] == node: 
            logger.info("Self-loop detected, cannot merge edge with itself.")
            return G
        
        # Handle parallel edges
        n0, n1 = neighbors[0], neighbors[0]
        edge_0, edge_1 = (node, n0, 0), (node, n1, 1)
        data_0, data_1 = G.get_edge_data(*edge_0), G.get_edge_data(*edge_1)
    else:
        n0, n1 = neighbors[0], neighbors[1]
        edge_0 = (node, n0)
        edge_1 = (node, n1)
        data_0 = list(G.get_edge_data(*edge_0).values())[0]
        data_1 = list(G.get_edge_data(*edge_1).values())[0]

    # Extract properties from both edges
    centerline_0, radius_0, length_0 = data_0["centerline"], data_0["radius"], data_0["length"]
    centerline_1, radius_1, length_1 = data_1["centerline"], data_1["radius"], data_1["length"]

    # Use the ID of the first edge
    id = data_0["id"]

    # Combine properties
    centerline = merge_centerlines(centerline_0, centerline_1)
    radius = radius_0 + radius_1
    length = length_0 + length_1

    # Add new edge connecting the two neighbors directly
    G.add_edge(
        n0,
        n1,
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
    G.remove_edge(*edge_0)
    G.remove_edge(*edge_1)
    G.remove_node(node)

    return G


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
        G = merge_two_edges(G, node)
        
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


# =========================================================
# Branch removal based on morphological criteria
# =========================================================

def should_remove_branch(
    edge_data: dict,
    parent_radius: float,
    end_radius: float,
    radius_ratio_threshold: float,
    length_ratio_threshold: float,
) -> bool:
    """
    Apply morphological criteria to determine if a terminal branch is artifactual.

    This function evaluates terminal branches against anatomical plausibility
    criteria. Branches that are very short or both thin and relatively short
    are considered artifacts from segmentation or skeletonization.

    Args:
        edge_data: Dictionary containing edge properties (length, etc.)
        parent_radius: Radius at the parent node
        end_radius: Radius at the terminal node
        radius_ratio_threshold: Minimum ratio end_radius/parent_radius to keep. default 0.3
        length_ratio_threshold: Length threshold relative to parent diameter. default 1.5

    Returns:
        bool: True if branch should be removed, False if it should be kept

    Criteria:
        1. Very short: length < parent_diameter → remove
        2. Thin and relatively short:
           length < parent_diameter * length_ratio AND
           radius_ratio < radius_ratio_threshold → remove
    """
    length = edge_data.get("length", 0)

    # Ensure parent_radius is a scalar
    if hasattr(parent_radius, "item"):
        parent_radius = parent_radius.item()
    if hasattr(end_radius, "item"):
        end_radius = end_radius.item()

    parent_diameter = 2 * float(parent_radius)
    radius_ratio = float(end_radius) / float(parent_radius) if parent_radius > 0 else 0

    # Case 1: Very short branch (much shorter than parent diameter)
    if length < parent_diameter:
        logger.info(f"Removing very short branch (ID {edge_data.get('id', 'unknown')}): length={length:.2f} < diameter={parent_diameter:.2f}")
        return True

    # Case 2: Slightly longer branch but with very thin end
    elif (
        length < parent_diameter * length_ratio_threshold
        and radius_ratio < radius_ratio_threshold
    ):
        logger.info(f"Removing thin long branch (ID {edge_data.get('id', 'unknown')}): length={length:.2f}, ratio={radius_ratio:.3f}")
        return True

    return False

def find_branches_to_remove(
        graph: nx.Graph,
        radius_ratio_threshold: float,
        length_ratio_threshold: float
        ) -> list[tuple[int, int, str]]:
    """
    Identify all terminal branches that meet removal criteria.

    This function efficiently scans all terminal nodes (degree 1) and
    evaluates their connecting edges against morphological criteria for
    artifact removal.

    Args:
        graph: NetworkX graph to analyze
        radius_ratio_threshold: Minimum end/parent radius ratio to preserve
        length_ratio_threshold: Length threshold factor relative to parent diameter

    Returns:
        list: Tuples of (terminal_node, parent_node, edge_id) for removal
              Empty list if no branches meet removal criteria
    """
    to_remove = []

    # Pre-filter terminal nodes for better performance
    terminal_nodes = [node for node in graph.nodes if graph.degree(node) == 1]

    for node in terminal_nodes:
        parent = next(graph.neighbors(node))
        edge_data = graph.get_edge_data(node, parent)[0]

        # Safely extract radius values
        parent_radius = graph.nodes[parent]["radius"]
        end_radius = graph.nodes[node]["radius"]

        # Ensure radius values are scalars
        if hasattr(parent_radius, "item"):
            parent_radius = parent_radius.item()
        if hasattr(end_radius, "item"):
            end_radius = end_radius.item()

        if should_remove_branch(
            edge_data,
            parent_radius,
            end_radius,
            radius_ratio_threshold,
            length_ratio_threshold,
        ):
            to_remove.append((node, parent, edge_data.get("id", "unknown")))

    return to_remove


def remove_artefact_branches(
        graph: nx.Graph,
        radius_ratio_threshold: float,
        length_ratio_threshold: float
        ) -> int:
    """
    Remove terminal branches identified as artifacts with safety protections.

    This is the main branch cleaning function that removes short, thin, or
    otherwise artifactual terminal branches while implementing safety checks
    to preserve important anatomical structures.

    Args:
        graph: NetworkX graph to clean (modified in-place)
        radius_ratio_threshold: Minimum end/parent radius ratio to preserve branches
        length_ratio_threshold: Length factor relative to parent diameter

    Returns:
        int: Number of branches removed
    """
    to_remove = find_branches_to_remove(
        graph, radius_ratio_threshold, length_ratio_threshold
    )
    logger.info(f"Found {len(to_remove)} branches to remove")

    G = graph.copy()
    # Apply the removals to the actual graph
    for node, parent, _ in to_remove:
        if G.has_edge(node, parent):
            G.remove_edge(node, parent)

    G.remove_nodes_from(list(nx.isolates(G)))

    return G


# ========================================================
# Full cleaning pipeline
# ========================================================

def clean_step(graph: nx.MultiGraph, 
               min_length: float, 
               radius_ratio_threshold: float, 
               length_ratio_threshold: float
               ) -> nx.MultiGraph:
    """
    Perform a single cleaning step on the graph.

    This function applies a sequence of cleaning operations:
    1. Remove small self-loops
    2. Remove artefact branches based on morphological criteria
    3. Remove degree-2 nodes to simplify the graph

    Args:
        graph (nx.MultiGraph): Input vascular graph
        min_length (float): Minimum length threshold for self-loops
        radius_ratio_threshold (float): Radius ratio threshold for branch removal
        length_ratio_threshold (float): Length ratio threshold for branch removal

    Returns:
        nx.MultiGraph: Cleaned and simplified graph
    """

    G_cleaned = remove_small_self_loops(graph, min_length)
    G_full_cleaned = remove_artefact_branches(G_cleaned, radius_ratio_threshold, length_ratio_threshold)
    G_simplified = remove_degree_2_nodes(G_full_cleaned)
    return G_simplified


def clean_graph(graph: nx.Graph, 
                min_length: float, 
                radius_ratio_threshold: float, 
                length_ratio_threshold: float
                ) -> nx.MultiGraph:
    """
    Iteratively clean the graph until no further changes occur.
    This function repeatedly applies the cleaning step until the graph stabilizes.

    Args:
        graph (nx.Graph): Input vascular graph
        min_length (float): Minimum length threshold for self-loops
        radius_ratio_threshold (float): Radius ratio threshold for branch removal
        length_ratio_threshold (float): Length ratio threshold for branch removal

    Returns:
        nx.MultiGraph: Fully cleaned and simplified graph
    """

    G_m = nx.MultiGraph()
    G_m.add_nodes_from(graph.nodes(data=True))
    G_m.add_edges_from(graph.edges(data=True))

    G_cleaned = clean_step(G_m, min_length=min_length, radius_ratio_threshold=radius_ratio_threshold, length_ratio_threshold=length_ratio_threshold)
    while (G_cleaned.number_of_edges() < G_m.number_of_edges()) or (G_cleaned.number_of_nodes() < G_m.number_of_nodes()):
        G_m = G_cleaned
        G_cleaned = clean_step(G_m, min_length=min_length, radius_ratio_threshold=radius_ratio_threshold, length_ratio_threshold=length_ratio_threshold)

    return G_cleaned