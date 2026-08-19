'''
Author: Oscar Morand (LRE, CREATIS)
Date: October 2025
Description: Functions to compute Betti numbers for graphs and 2D masks.
'''

from skimage.measure import label
import numpy as np
import warnings
import networkx as nx


# ============================================================================
# Betti numbers for 2D masks
# ============================================================================

def betti_0_2D(mask: np.ndarray) -> int:
    '''
    Compute the 0th Betti number (number of connected components) for a 2D binary mask.

    Parameters:
        mask (np.ndarray): 2D binary mask where foreground pixels are True.

    Returns:
        int: The 0th Betti number (number of connected components).
    '''
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array.")
    if mask.dtype != bool:
        warnings.warn("Input mask is not boolean. Converting to boolean.")
        mask = (mask > 0)

    # Label connected components in the mask, using 8-connectivity
    _, num_cc = label(mask, return_num=True, connectivity=2)
    return num_cc


def betti_1_2D(mask: np.ndarray) -> int:
    '''
    Compute the 1st Betti number (number of holes) for a 2D binary mask.

    Parameters:
        mask (np.ndarray): 2D binary mask where foreground pixels are True.

    Returns:
        int: The 1st Betti number (number of holes).
    '''
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array.")
    if mask.dtype != bool:
        warnings.warn("Input mask is not boolean. Converting to boolean.")
        mask = (mask > 0)

    # Invert the mask to find holes
    inverse_mask = np.logical_not(mask)
    # Label connected components in the inverted mask, using 4-connectivity
    _, num_cc = label(inverse_mask, return_num=True, connectivity=1)
    # Subtract 1 to exclude the outer background component
    return num_cc - 1


def betti_numbers_2D(mask: np.ndarray) -> tuple[int, int]:
    '''
    Compute the Betti numbers (0th and 1st) for a 2D binary mask.

    Parameters:
        mask (np.ndarray): 2D binary mask where foreground pixels are True.

    Returns:
        tuple: A tuple containing the 0th and 1st Betti numbers.
    '''
    b0 = betti_0_2D(mask)
    b1 = betti_1_2D(mask)
    return b0, b1


# ============================================================================
# Betti numbers for graphs
# ============================================================================

def betti_0_graph(G: nx.Graph) -> int:
    """
    Compute the 0th Betti number (number of connected components) for a graph.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        int: The 0th Betti number (number of connected components).
    """
    return nx.number_connected_components(G)


def betti_1_graph(G: nx.Graph) -> int:
    """
    Compute the 1st Betti number (number of holes) for a graph.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        int: The 1st Betti number (number of holes).
    """
    return nx.number_of_edges(G) - nx.number_of_nodes(G) + nx.number_connected_components(G)


def betti_numbers_graph(G: nx.Graph) -> tuple[int, int]:
    """
    Compute the Betti numbers (0th and 1st) for a graph.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        tuple: A tuple containing the 0th and 1st Betti numbers.
    """

    b0 = betti_0_graph(G)
    b1 = betti_1_graph(G)
    return b0, b1