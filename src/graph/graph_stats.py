'''
Author: Oscar Morand (LRE, CREATIS)
Date: October 2025
Description: Functions to compute statistics and histograms of graph attributes.
'''


import logging
import networkx as nx
import numpy as np


logger = logging.getLogger(__name__)

def graph_attribute_histogram(graph: nx.MultiGraph, attribute: str) -> dict:
    """
    Compute the histogram of a specified edge attribute in the graph.

    Parameters:
        graph (nx.MultiGraph): The input vascular graph.
        attribute (str): The edge attribute to compute the histogram for.

    Returns:
        dict: A dictionary with attribute values as keys and their counts as values.
    """

    attribute_counts = {}
    for _, _, data in graph.edges(data=True):
        attr_value = data.get(attribute, None)
        if attr_value is not None:
            if attr_value in attribute_counts:
                attribute_counts[attr_value] += 1
            else:
                attribute_counts[attr_value] = 1
        else:
            logger.warning(f"Edge without '{attribute}' attribute found.")

    if not attribute_counts or len(attribute_counts) == 0:
        logger.warning(f"No '{attribute}' data found in edges.")

    return attribute_counts

def graph_depth_histogram(graph: nx.MultiGraph) -> dict:
    """
    Compute the histogram of depth levels in the graph.

    Parameters:
        graph (nx.MultiGraph): The input vascular graph with 'depth' attribute on edges.

    Returns:
        dict: A dictionary with depth levels as keys and their counts as values.
    """

    return graph_attribute_histogram(graph, 'depth')


def graph_hierarchy_histogram(graph: nx.MultiGraph) -> dict:
    """
    Compute the histogram of hierarchy levels in the graph.

    Parameters:
        graph (nx.MultiGraph): The input vascular graph with 'hierarchy' attribute on edges.

    Returns:
        dict: A dictionary with hierarchy levels as keys and their counts as values.
    """

    return graph_attribute_histogram(graph, 'hierarchy')


def graph_nodes_degree_histogram(graph: nx.MultiGraph) -> dict:
    """
    Compute the histogram of node degrees in the graph.

    Parameters:
        graph (nx.MultiGraph): The input vascular graph.

    Returns:
        dict: A dictionary with node degrees as keys and their counts as values.
    """

    degree_counts = {}
    for _, degree in graph.degree():
        if degree in degree_counts:
            degree_counts[degree] += 1
        else:
            degree_counts[degree] = 1

    if not degree_counts or len(degree_counts) == 0:
        logger.warning("No node degree data found.")

    return degree_counts


def graph_radius_histogram(graph: nx.MultiGraph) -> dict:
    """
    Compute the histogram of radius values in the graph.

    Parameters:
        graph (nx.MultiGraph): The input vascular graph with 'radius' attribute on edges.

    Returns:
        dict: A dictionary with radius values as keys and their counts as values.
    """
    total_radius = []
    for _, _, data in graph.edges(data=True):
        radius = data.get('radius', 0.0)
        total_radius += radius

    hist, bin_edges = np.histogram(total_radius, bins='auto')

    radius_hist = {f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}": int(hist[i]) for i in range(len(hist))}

    return radius_hist


def get_graph_total_length(graph: nx.MultiGraph) -> float:
    """
    Compute the total length of all edges in the graph.

    Parameters:
        graph (nx.MultiGraph): The input vascular graph with 'length' attribute on edges.

    Returns:
        float: The total length of all edges.
    """

    total_length = 0.0
    for _, _, data in graph.edges(data=True):
        length = data.get('length', 0.0)
        total_length += length

    return total_length


def get_graph_average_radius(graph: nx.MultiGraph) -> float:
    """
    Compute the average radius of all edges in the graph.

    Parameters:
        graph (nx.MultiGraph): The input vascular graph with 'radius' attribute on edges.

    Returns:
        float: The average radius of all edges.
    """

    total_radius = []
    for _, _, data in graph.edges(data=True):
        radius = data.get('radius', 0.0)
        total_radius += radius

    return np.mean(total_radius)


def get_histogram_difference(hist1: dict, hist2: dict) -> dict:
    """
    Compute the difference between two histograms.

    Parameters:
        hist1 (dict): The first histogram.
        hist2 (dict): The second histogram.

    Returns:
        dict: A dictionary with keys from both histograms and their differences as values.
    """

    all_keys = set(hist1.keys()).union(set(hist2.keys()))
    diff_hist = {key: hist1.get(key, 0) - hist2.get(key, 0) for key in all_keys}

    return diff_hist