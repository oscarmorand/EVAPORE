'''
Author: Oscar Morand (LRE, CREATIS)
Date: October 2025
Description: Functions to label graph edges and nodes with different attributes (depth, hierarchy, etc.)
'''

import networkx as nx
import numpy as np

from graph.graph_utils import get_cycles_edges, get_terminal_edges, TopologicalClass

def generate_graph_depth(graph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Generate a new graph with depth information for each edge.
    Depth is defined as the minimum number of edges from any terminal node.

    Parameters:
        graph (nx.MultiGraph): Input vascular graph.

    Returns:
        nx.MultiGraph: Graph with 'depth' attribute added to each edge.
    """

    # Get all terminal nodes
    terminals = [n for n, d in graph.degree() if d == 1]

    # BFS from all terminal nodes
    q = [(n, 0) for n in terminals]

    depths = {edge_data['id']: -1 for _, _, edge_data in graph.edges(data=True)}
    visited = {n: False for n in graph.nodes()}
    edges = {edge_data['id']: (u, v) for u, v, edge_data in graph.edges(data=True)}

    # BFS loop
    while len(q) > 0:
        n, prev_depth = q.pop(0)
        if visited[n]:
            continue
        visited[n] = True

        new_depth = prev_depth + 1

        for neighbor in graph.neighbors(n):
            edge = (n, neighbor) if (n, neighbor) in graph.edges else (neighbor, n)
            multi_edge = graph.get_edge_data(*edge)
            for val in multi_edge.values():
                edge_id = val['id']
                edge_depth = depths[edge_id]

                if new_depth < edge_depth or edge_depth == -1:
                    depths[edge_id] = new_depth

                q.append((neighbor, new_depth))

    G = graph.copy()
    # add depth to graph edges as attribute
    for edge_id, depth in depths.items():
        u, v = edges[edge_id]
        for e, val in G.get_edge_data(u, v).items():
            real_id = (u, v, e)
            G.edges[real_id]['depth'] = depth

    return G


def generate_graph_hierarchy(graph: nx.MultiGraph,
                             root: int
                             ) -> nx.MultiGraph:
    """
    Generate a graph representing the hierarchy from the given root node, with a hierarchy attribute on edges and nodes.

    Parameters:
        graph (nx.MultiGraph): The input undirected graph.
        root (int): The root node from which to generate the hierarchy.

    Returns:
        nx.MultiGraph: A graph representing the hierarchy.
    """

    G = graph.copy()

    q = [(root, 0)]

    while len(q) > 0:
        n, hierarchy = q.pop(0)

        # Set hierarchy for the node
        G.nodes[n]['hierarchy'] = hierarchy

        new_hierarchy = hierarchy + 1

        for neighbor in G.neighbors(n):
            edge = (n, neighbor) if (n, neighbor) in G.edges else (neighbor, n)
            edge_data = G.get_edge_data(*edge)

            for e, val in edge_data.items():
                # If the edge already has a hierarchy and it's less than or equal to current, skip
                if 'hierarchy' in val and val['hierarchy'] <= new_hierarchy:
                    continue

                # Set hierarchy for the edge
                real_id = (*edge, e)
                G.edges[real_id]['hierarchy'] = new_hierarchy

                q.append((neighbor, new_hierarchy))

    return G


def generate_graph_topological_classification(graph: nx.MultiGraph) -> nx.MultiGraph:
    """Generate a graph with topological classification of edges.
    Classify edges into three categories:
        - 't0': edges that are neither part of a cycle nor terminal edges, their removal increases the number of connected components,
            so the betti number b0 increases by 1.
        - 't1': edges that are part of at least one cycle, their removal decreases the number of cycles, so the betti number b1 decreases by 1,
            and the number of connected components remains the same.
        - 'non_topological': terminal edges, their removal does not change the number of connected components or cycles.

    Parameters:
        graph (nx.MultiGraph): Input vascular graph.

    Returns:
        nx.MultiGraph: Graph with 'topological_class' attribute added to each edge.
    """
    cycles_edges = get_cycles_edges(graph)
    terminal_edges = get_terminal_edges(graph)
    t0_edges = set()
    for u, v, d in graph.edges(data=True):
        if (u,v) not in cycles_edges and (v,u) not in cycles_edges and (u,v) not in terminal_edges and (v,u) not in terminal_edges:
            t0_edges.add((u,v))
    
    classification = {
        TopologicalClass.NON_TOPOLOGICAL: terminal_edges,
        TopologicalClass.T0: t0_edges,
        TopologicalClass.T1: cycles_edges,
    }
    
    G = graph.copy()
    for class_name, edges_set in classification.items():
        for u, v in edges_set:
            edge_data = G.get_edge_data(u, v)
            for e, val in edge_data.items():
                real_id = (u, v, e)
                G.edges[real_id]['topological_class'] = class_name

    return G


def generate_graph_edge_radius_decay(graph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Generate a graph with 'radius_decay' attribute for each edge.
    The 'radius_decay' is defined as (max_radius - min_radius) / max_radius.

    Parameters:
        graph (nx.MultiGraph): Input vascular graph with 'min_radius' and 'max_radius' attributes on edges.

    Returns:
        nx.MultiGraph: Graph with 'radius_decay' attribute added to each edge.
    """

    G = graph.copy()

    for u, v, d in G.edges(data=True):
        for key, val in G.get_edge_data(u, v).items():
            real_id = (u, v, key)
            if 'min_radius' not in val or 'max_radius' not in val:
                raise ValueError(f"Edge {real_id} does not have 'min_radius' or 'max_radius' attribute. Please compute graph depth first.")

            min_radius, max_radius = val['min_radius'], val['max_radius']
            radius_decay = (max_radius - min_radius) / max_radius
        
            G.edges[real_id]['radius_decay'] = radius_decay

    return G