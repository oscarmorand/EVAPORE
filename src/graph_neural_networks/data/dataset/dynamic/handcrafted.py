import networkx as nx
import torch
from graph.graph_utils import find_max_radius_node, find_max_radius_node_patch_based

def get_node_degree(nx_graph: nx.Graph, 
                    node_idx: int,
                    **kwargs
) -> int:
    return nx_graph.degree[node_idx]

def get_node_coord_x(nx_graph: nx.Graph, 
                     node_idx: int,
                     **kwargs
) -> float:
    return nx_graph.nodes[node_idx]['pos'][1]

def get_node_coord_y(nx_graph: nx.Graph, 
                     node_idx: int,
                     **kwargs
) -> float:
    return nx_graph.nodes[node_idx]['pos'][0]

def get_centered_normalized_x(nx_graph: nx.Graph,
                              node_idx: int,
                              base_image_shape: tuple[int, int] = None
) -> float:
    max_radius_node = find_max_radius_node(nx_graph)
    max_node_pos = nx_graph.nodes[max_radius_node]['pos']
    centered_x = nx_graph.nodes[node_idx]['pos'][1] - max_node_pos[1]
    normalized_x = centered_x / (base_image_shape[1] if base_image_shape else 1)
    return normalized_x

def get_centered_normalized_y(nx_graph: nx.Graph,
                              node_idx: int,
                              base_image_shape: tuple[int, int] = None
) -> float:
    max_radius_node = find_max_radius_node(nx_graph)
    max_node_pos = nx_graph.nodes[max_radius_node]['pos']
    centered_y = nx_graph.nodes[node_idx]['pos'][0] - max_node_pos[0]
    normalized_y = centered_y / (base_image_shape[0] if base_image_shape else 1)
    return normalized_y

def get_reference_x(nx_graph: nx.Graph,
                    node_idx: int,
                    base_image_shape: tuple[int, int] = None
) -> float:
    max_radius_node = find_max_radius_node_patch_based(nx_graph, patch_size=100, base_image_shape=base_image_shape)
    max_node_pos = nx_graph.nodes[max_radius_node]['pos']
    centered_x = nx_graph.nodes[node_idx]['pos'][1] - max_node_pos[1]
    normalized_x = centered_x / (base_image_shape[1] if base_image_shape else 1)
    reference_x = normalized_x
    if not base_image_shape is None:
        width = base_image_shape[1]
        if max_node_pos[1] > (width / 2):
            reference_x = -normalized_x
    return reference_x

def get_reference_y(nx_graph: nx.Graph,
                    node_idx: int,
                    base_image_shape: tuple[int, int] = None
) -> float:
    max_radius_node = find_max_radius_node_patch_based(nx_graph, patch_size=100, base_image_shape=base_image_shape)
    max_node_pos = nx_graph.nodes[max_radius_node]['pos']
    centered_y = nx_graph.nodes[node_idx]['pos'][0] - max_node_pos[0]
    reference_y = centered_y / (base_image_shape[0] if base_image_shape else 1)
    return reference_y
    

def get_node_radius(nx_graph: nx.Graph, 
                    node_idx: int,
                    **kwargs
) -> float:
    return nx_graph.nodes[node_idx]['radius']


def get_edge_length(nx_graph: nx.Graph,
                    src_idx: int,
                    dst_idx: int
) -> float:
    return nx_graph.get_edge_data(src_idx, dst_idx)['length']

def get_edge_mean_radius(nx_graph: nx.Graph,
                         src_idx: int,
                         dst_idx: int
) -> float:
    return nx_graph.get_edge_data(src_idx, dst_idx)['mean_radius']

def get_edge_min_radius(nx_graph: nx.Graph,
                        src_idx: int,
                        dst_idx: int
) -> float:
    return nx_graph.get_edge_data(src_idx, dst_idx)['min_radius']

def get_edge_max_radius(nx_graph: nx.Graph,
                        src_idx: int,
                        dst_idx: int
) -> float:
    return nx_graph.get_edge_data(src_idx, dst_idx)['max_radius']

def get_edge_radius_decay(nx_graph: nx.Graph,
                         src_idx: int,
                         dst_idx: int
) -> float:
    return nx_graph.get_edge_data(src_idx, dst_idx)['radius_decay']

# ===============================================================
# HANDCRAFTED FEATURE METHODS DICTIONARIES
# ===============================================================

handcrafted_node_feature_methods_dict = {
    "degree": get_node_degree,
    "coord_x": get_node_coord_x,
    "coord_y": get_node_coord_y,
    "centered_normalized_x": get_centered_normalized_x,
    "centered_normalized_y": get_centered_normalized_y,
    "reference_x": get_reference_x,
    "reference_y": get_reference_y,
    "radius": get_node_radius,
}

handcrafted_edge_feature_methods_dict = {
    "length": get_edge_length,
    "mean_radius": get_edge_mean_radius,
    "min_radius": get_edge_min_radius,
    "max_radius": get_edge_max_radius,
    "radius_decay": get_edge_radius_decay
}


# ===============================================================
# HANDCRAFTED FEATURE FUNCTIONS
# ===============================================================

def handcraft_node_feature(feature_name: str,
                           nx_graph: nx.Graph,
                           node_idx: int,
                           **kwargs) -> float:
    if feature_name not in handcrafted_node_feature_methods_dict:
        raise ValueError(f"Handcrafted node feature {feature_name} not recognized.")
    feature_func = handcrafted_node_feature_methods_dict[feature_name]
    return feature_func(nx_graph, node_idx, **kwargs)


# ===============================================================
# REQUIREMENTS FOR HANDCRAFTED FEATURES
# ===============================================================

required_node_attr_dict = {
    "degree": [],
    "coord_x": ['pos'],
    "coord_y": ['pos'],
    "centered_normalized_x": ['pos'],
    "centered_normalized_y": ['pos'],
    "radius": ['radius'],
}

required_edge_attr_dict = {
    "length": ['length'],
    "mean_radius": ['mean_radius'],
    "min_radius": ['min_radius'],
    "max_radius": ['max_radius'],
    "radius_decay": ['radius_decay']
}

required_node_attr_transform_dict = {
    "degree": None,
    "coord_x": None,
    "coord_y": None,
    "radius": None
}

required_edge_attr_transform_dict = {
    "length": None,
    "mean_radius": None,
    "min_radius": None,
    "max_radius": None,
    "radius_decay": None
}

def check_required_node_attrs(feature_name: str,
                              nx_graph: nx.Graph) -> bool:
    if feature_name not in required_node_attr_dict:
        raise ValueError(f"Handcrafted node feature {feature_name} not recognized.")
    
    # Get required attributes for the feature
    required_attrs = required_node_attr_dict.get(feature_name, [])
    if len(required_attrs) == 0:
        return nx_graph
    
    # Check if all required attributes are present
    actual_attrs = [True for _ in required_attrs]
    for (_, attrs) in nx_graph.nodes(data=True):
        for attr_i, req_attr in enumerate(required_attrs):
            if req_attr not in attrs:
                actual_attrs[attr_i] = False

    for attr_i, req_attr in enumerate(required_attrs):
        if not actual_attrs[attr_i]:
            # We missing a required attribute on at least one node, if possible apply the required transform, else raise error
            if req_attr not in required_node_attr_transform_dict:
                raise ValueError(f"Handcrafted node feature {req_attr} not recognized.")

            transform_func = required_node_attr_transform_dict.get(req_attr, None)
            if transform_func is not None:
                nx_graph = transform_func(nx_graph)
            else:
                raise ValueError(f"Node attribute {req_attr} required for handcrafted feature {feature_name} is missing and no transform is available to compute it.")

    return nx_graph

 # TODO: implement for edge features