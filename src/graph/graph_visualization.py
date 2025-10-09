'''
Author: Oscar Morand (LRE, CREATIS)
Date: October 2025
Description: Functions for visualizing vascular graphs
'''

import numpy as np
import networkx as nx

def get_graph_overlay_img(img: np.ndarray,
                          graph: nx.Graph
                          ) -> np.ndarray:
    """
    Overlay the graph on the original image for visualization.

    Args:
        img (np.ndarray): Original binary image.
        graph (nx.Graph): Vascular graph to overlay.

    Returns:
        np.ndarray: Image in RGB format with graph overlay.
    """

    # Create an RGB version of the original image
    viz = np.zeros((*img.shape, 3), dtype=np.uint8)
    viz[img > 0] = [255, 255, 255]

    # Overlay edges in red
    for u, v, data in graph.edges(data=True):
        centerline = np.array(data['centerline'])
        viz[centerline[:,0], centerline[:,1]] = [255, 0, 0]

    # Overlay nodes in green
    for n, data in graph.nodes(data=True):
        pos = data['pos']
        viz[pos[0], pos[1]] = [0, 255, 0]

    return viz