import networkx as nx

from graph_neural_networks.data.dataset.dynamic.graph_transforms.graph_transform import GraphTransform
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper

transforms_orders = [
    "compute_distance_matrix_transform",
    "oversample_nodes_transform", 
    "add_edge_closest_cc_transform", 
    "add_distance_cc_edge_transform"
]

def apply_graph_transforms(graph_wrapper: GraphWrapper,
                           graph_transforms: dict[str, GraphTransform]
) -> GraphWrapper:
    """
    Apply a series of graph transforms to the input graph.

    Args:
        graph_wrapper: (GraphWrapper) The input graph wrapper object.
        graph_transforms: (dict[str, GraphTransform]) A dictionary of graph transforms to apply.

    Returns:
        (GraphWrapper) The transformed graph wrapper after applying all specified transforms.
    """

    transform_names = list(graph_transforms.keys())
    ordered_graph_transforms = []
    for transform_name in transforms_orders:
        if transform_name in transform_names:
            ordered_graph_transforms.append(graph_transforms[transform_name])
    for transform_name in transform_names:
        if graph_transforms[transform_name] not in ordered_graph_transforms:
            ordered_graph_transforms.append(graph_transforms[transform_name])

    for transform in ordered_graph_transforms:
        graph_wrapper = transform(graph_wrapper)

    return graph_wrapper