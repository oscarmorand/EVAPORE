from enum import Enum
import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation

from graph.graph_creation import img_to_graph

class EdgePredState(Enum):
    NOT_IN_PREDICTION = 0
    IN_PREDICTION = 1

source_name_to_attr = {
        "not_in_p_graph": EdgePredState.NOT_IN_PREDICTION,
        "in_p_graph": EdgePredState.IN_PREDICTION
} 

def add_edges_from_graph(combined_graph, source_graph, source_name, combined_node_coords_to_node): 
    G = combined_graph.copy()
    for u, v, data in source_graph.edges(data=True):
        coord_u = tuple(source_graph.nodes[u]['pos'])
        coord_v = tuple(source_graph.nodes[v]['pos'])
        if coord_u not in combined_node_coords_to_node or coord_v not in combined_node_coords_to_node:
            print(f"Skipping edge from {source_name} with unknown nodes:", coord_u, coord_v)
            continue
        combined_u = combined_node_coords_to_node[coord_u]
        combined_v = combined_node_coords_to_node[coord_v]
        G.add_edge(
            combined_u, 
            combined_v, 
            **data, 
            edge_pred_state=source_name_to_attr[source_name]
        )  
    return G

def combine_graphs(in_p_graph, not_in_p_graph, radius_map):
    combined_graph = nx.MultiGraph()

    in_p_node_coords = set([tuple(data['pos']) for _, data in in_p_graph.nodes(data=True)])
    not_in_p_node_coords = set([tuple(data['pos']) for _, data in not_in_p_graph.nodes(data=True)])

    combined_node_coords = in_p_node_coords.union(not_in_p_node_coords)

    for combined_node_i, coord in enumerate(combined_node_coords):
        radius = radius_map[coord[0], coord[1]]
        combined_graph.add_node(combined_node_i, pos=coord, radius=radius)

    combined_node_coords_to_node = {tuple(data['pos']): node for node, data in combined_graph.nodes(data=True)}

    combined_graph = add_edges_from_graph(combined_graph, in_p_graph, "in_p_graph", combined_node_coords_to_node)
    combined_graph = add_edges_from_graph(combined_graph, not_in_p_graph, "not_in_p_graph", combined_node_coords_to_node)

    return combined_graph


def get_combined_graph(gt, pred):
    gt_graph = img_to_graph(gt, clean=True)
    radius_map = np.zeros_like(gt, dtype=np.float32)
    skel_g = np.zeros_like(gt, dtype=bool)

    for u, v, data in gt_graph.edges(data=True):
        centerline = np.array(data['centerline'])
        radius = np.array(data['radius'])
        skel_g[centerline[:,0], centerline[:,1]] = True
        radius_map[centerline[:,0], centerline[:,1]] = radius

    g_skel_in_p = np.logical_and(skel_g, pred)
    g_skel_not_in_p = np.logical_and(skel_g, np.logical_not(pred))

    footprint = np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]])
    g_skel_not_in_p_dilated = np.logical_and(binary_dilation(g_skel_not_in_p, footprint), skel_g)

    g_in_p_graph = img_to_graph(g_skel_in_p, clean=False)
    g_not_in_p_graph = img_to_graph(g_skel_not_in_p_dilated, clean=False)

    combined_graph = combine_graphs(g_in_p_graph, g_not_in_p_graph, radius_map)
    return combined_graph
