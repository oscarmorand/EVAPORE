'''
Authors: Oscar Morand (LRE, CREATIS), Morgane Des-Ligneris (CREATIS)
Date: October 2025
Description: Functions to read/write graphs in various formats (JSON, DOT, PNG).
'''

import logging
import numpy as np
import json
import networkx as nx
from PIL import Image

from graph.graph_visualization import get_graph_overlay_img


logger = logging.getLogger(__name__)


def ndarray_to_list(obj):
    """
    Convert numpy arrays to lists recursively for JSON serialization.
    
    Parameters:
        - obj: The object to convert (can be a numpy array, dict, list, or tuple).

    Returns:
        - The converted object with numpy arrays replaced by lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ndarray_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(ndarray_to_list(item) for item in obj)
    else:
        return obj


def save_graph_to_json(graph: nx.Graph,
                       path: str
                       ) -> None:
    """
    Save a NetworkX graph (directed or undirected) to JSON format.

    Parameters:
        - graph (NetworkX.Graph): graph to save
        - path (str): Output file path (e.g., 'graph.json')
    """
    # Convert graph to node-link data format
    data = nx.readwrite.json_graph.node_link_data(graph, edges="edges")

    # Convert all numpy arrays to lists recursively
    data_list = ndarray_to_list(data)

    # Save to JSON file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4)


def save_graph_to_dot(graph: nx.Graph,
                      path: str
                      ) -> None:
    """
    Save a NetworkX graph to DOT format.

    Parameters:
        - graph (NetworkX.Graph): graph to save
        - path (str): Output file path (e.g., 'graph.dot')
    """
    try:
        nx.nx_pydot.write_dot(graph, path)
        logging.info(f"Graph saved to {path} in DOT format.")
    except ImportError as e:
        logging.error("pydot is required to save graphs in DOT format. Please install it via 'pip install pydot'.")
        raise e
    

def save_graph_to_img(img: np.ndarray,
                      graph: nx.Graph,
                      path: str
                      ) -> None:
    """
    Save a graph overlayed on the original image.

    Parameters:
        - img (np.ndarray): Original image
        - graph (nx.Graph): Graph to overlay
        - path (str): Output file path (e.g., 'graph.png')
    """
    overlay = get_graph_overlay_img(img, graph)
    Image.fromarray(overlay).save(path)