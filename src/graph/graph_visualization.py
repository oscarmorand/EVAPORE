'''
Author: Oscar Morand (LRE, CREATIS)
Date: October 2025
Description: Functions for visualizing vascular graphs
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging
import plotly.graph_objects as go
from skimage import measure

from graph.graph_labeling import generate_graph_edge_radius_decay, generate_graph_topological_classification, generate_graph_depth
from graph.graph_utils import find_max_radius_node, TopologicalClass
from graph.graph_stats import graph_attribute_histogram, graph_depth_histogram, graph_hierarchy_histogram, graph_nodes_degree_histogram, graph_radius_histogram, get_histogram_difference
from graph.graph_pred_state import EdgePredState

logger = logging.getLogger(__name__)


def get_virtual_centerline(pos_start: np.ndarray, pos_goal: np.ndarray) -> np.ndarray:
    '''
    Generates a straight-line (DDA-style) sequence of integer coordinates
    between pos_start and pos_goal, in mask-index order (e.g. (row, col) for 2D,
    (d, row, col) for 3D). Works for any number of dimensions.

    Args:
        pos_start (np.ndarray): shape (D,) starting coordinate.
        pos_goal (np.ndarray): shape (D,) goal coordinate.

    Returns:
        np.ndarray: shape (L, D) sequence of integer coordinates from start to goal.
    '''
    p0 = np.asarray(pos_start, dtype=float)
    p1 = np.asarray(pos_goal, dtype=float)

    diff = p1 - p0
    steps = int(np.max(np.abs(diff)))

    if steps == 0:
        return p0[None, :].astype(np.int32)

    increments = diff / steps
    t = np.arange(steps + 1)[:, None]           # (steps+1, 1)
    points = p0[None, :] + t * increments[None, :]  # (steps+1, D)
    points[-1] = p1  # avoid float drift on the last point

    return points.astype(np.int32)


def display_graph_overlay_vol(gt: np.ndarray, graph: nx.Graph) -> None:
    """
    Visualize the graph and the ground truth mesh using Plotly.

    Args:
        graph (nx.Graph): The graph to visualize.
        gt (np.ndarray): The ground truth 3D array for mesh extraction.
    """

    # extract surface
    verts, faces, _, _ = measure.marching_cubes(gt, level=0.5)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:,0],
                y=verts[:,1],
                z=verts[:,2],
                i=faces[:,0],
                j=faces[:,1],
                k=faces[:,2],
                opacity=0.5
            )
        ]
    )

    skeleton_indices = []
    node_index = None

    for u, v, data in graph.edges(data=True):
        centerline = np.array(data["centerline"])
        if centerline is None:
            continue
        edge_pred_state = data.get("edge_pred_state", None)

        color = "red"
        if edge_pred_state is not None:
            if edge_pred_state.value == EdgePredState.IN_PREDICTION.value:
                color = "green"
            elif edge_pred_state.value == EdgePredState.NOT_IN_PREDICTION.value:
                color = "red"
            else:
                color = "blue"

        fig.add_trace(
            go.Scatter3d(
                x=centerline[:,0],
                y=centerline[:,1],
                z=centerline[:,2],
                mode="lines",
                line=dict(color=color, width=5)
            )
        )

        skeleton_indices.append(len(fig.data)-1)

    node_positions = []
    for node, data in graph.nodes(data=True):
        pos = np.array(data["pos"])
        node_positions.append(pos)
    node_positions = np.array(node_positions)

    node_index = len(fig.data)-1
    
    fig.add_trace(
        go.Scatter3d(
            x=node_positions[:,0],
            y=node_positions[:,1],
            z=node_positions[:,2],
            mode="markers",
            marker=dict(size=3, color="orange"),
            name="nodes",
            showlegend=False
        )
    )

    fig.update_layout(
        showlegend=False,
        width=1200,
        height=800,
        updatemenus=
        [
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Hide mesh",
                        method="update",
                        args=[{"visible": [False] + [True] * len(skeleton_indices) + [True]}]
                    ),
                    dict(
                        label="Hide skeleton",
                        method="update",
                        args=[{"visible": [True] + [False] * len(skeleton_indices) + [True]}]
                    ),
                    dict(
                        label="Hide nodes",
                        method="update",
                        args=[{"visible": [True] + [True] * len(skeleton_indices) + [False]}]
                    ),
                    dict(
                        label="Show all",
                        method="update",
                        args=[{"visible": [True] * (len(skeleton_indices) + 2)}]
                    )
                ]
            )
        ]
    )

    fig.show()


def get_graph_overlay_img(img: np.ndarray,
                          graph: nx.Graph,
                          show_edges: bool = True
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
    if show_edges:
        for u, v, data in graph.edges(data=True):
            centerline = np.array(data['centerline'])
            edge_pred_state = data.get('edge_pred_state', None)
            virtual_edge = data.get('virtual_edge', False)
            color = [255, 255, 255] # Default white
            if edge_pred_state is not None:
                if edge_pred_state in [EdgePredState.IN_PREDICTION.value, EdgePredState.IN_PREDICTION]:
                    color = [0, 0, 255]  # Blue for IN_PREDICTION
                elif edge_pred_state in [EdgePredState.NOT_IN_PREDICTION.value, EdgePredState.NOT_IN_PREDICTION]:
                    color = [255, 0, 0]  # Red for NOT_IN_PREDICTION
                else:
                    color = [255, 255, 0]  # Yellow for unknown
            else:
                color = [0, 0, 255]  # Blue for edges without prediction state
            if virtual_edge:
                color = [0, 255, 255]  # Cyan for virtual edges
                x0, y0 = graph.nodes[u]['pos']
                x1, y1 = graph.nodes[v]['pos']
                centerline = get_virtual_centerline((x0, y0), (x1, y1))

            if centerline is None:
                print(centerline, edge_pred_state, virtual_edge)

            viz[centerline[:,0], centerline[:,1]] = color

    # Overlay nodes in green
    for n, data in graph.nodes(data=True):
        pos = data['pos']
        viz[pos[0], pos[1]] = [0, 255, 0]

    return viz

def display_graph_overlay_img(img: np.ndarray,
                       graph: nx.Graph,
                       show_edges: bool = True,
                       figsize: tuple[int, int] = (8, 8)
                       ) -> None:
    """
    Display the graph overlayed on the original image.

    Args:
        img (np.ndarray): Original binary image.
        graph (nx.Graph): Vascular graph to overlay.
    """
    viz = get_graph_overlay_img(img, graph, show_edges)
    plt.figure(figsize=figsize)
    plt.imshow(viz)
    plt.axis('off')
    plt.show()

def display_graph_overlay(img: np.ndarray,
                          graph: nx.Graph,
                          show_edges: bool = True,
                          figsize: tuple[int, int] = (8, 8)
) -> None:
    if img.ndim == 2:
        display_graph_overlay_img(img, graph, show_edges, figsize)
    else:
        display_graph_overlay_vol(img, graph)

def display_edges_set(graph: nx.MultiGraph,
                      edges: set,
                      img: np.ndarray
                      ) -> None:
    """
    Display a set of edges overlayed on the original image.

    Args:
        graph (nx.MultiGraph): The input graph.
        edges (set): A set of edges to display.
        img (np.ndarray): The original image for background.
    """

    background = np.zeros((*img.shape, 3), dtype=np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(background, cmap='gray')

    for u, v in edges:
        if graph.has_edge(u, v):
            data = graph.get_edge_data(u, v)[0]
            centerline = np.array(data['centerline'])
            plt.plot(centerline[:,1], centerline[:,0], color='blue', linewidth=2)

    plt.axis('off')
    plt.show()


def display_graph_difference(img: np.ndarray,
                          graph1: nx.Graph,
                          graph2: nx.Graph,
                          bounding_box_width: int = 5
                          ) -> None:
    """
    Display the differences between two graphs overlayed on the original image.

    Args:
        img (np.ndarray): Original binary image.
        graph1 (nx.Graph): First vascular graph.
        graph2 (nx.Graph): Second vascular graph.
    """
    viz = np.zeros((*img.shape, 3), dtype=np.uint8)
    viz[img > 0] = [255, 255, 255]

    # Edges in graph1 but not in graph2 in red
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    diff_edges = edges1.symmetric_difference(edges2)

    for u, v in diff_edges:
        if (u, v) in edges1:
            data = graph1.get_edge_data(u, v)[0]
        else:
            data = graph2.get_edge_data(u, v)[0]
        centerline = np.array(data['centerline'])
        viz[centerline[:,0], centerline[:,1]] = [255, 0, 0]

        # Highlight bounding box of differences
        all_coords = np.array(data['centerline'])
        min_row, min_col = np.min(all_coords, axis=0) + (-bounding_box_width, -bounding_box_width)
        max_row, max_col = np.max(all_coords, axis=0) + (bounding_box_width, bounding_box_width)
        viz[min_row:max_row+1, min_col - bounding_box_width:min_col + bounding_box_width] = [0, 255, 255]
        viz[min_row:max_row+1, max_col - bounding_box_width:max_col + bounding_box_width] = [0, 255, 255]
        viz[min_row - bounding_box_width:min_row + bounding_box_width, min_col:max_col+1] = [0, 255, 255]
        viz[max_row - bounding_box_width:max_row + bounding_box_width, min_col:max_col+1] = [0, 255, 255]

    plt.figure(figsize=(8, 8))
    plt.imshow(viz)
    plt.axis('off')
    plt.show()


def display_graph_edge_depth(graph: nx.MultiGraph,
                          img: np.ndarray
                          ) -> None:
    """
    Plot the graph overlayed on the image, coloring edges based on their depth.

    Parameters:
        graph (nx.MultiGraph): The input graph.
        img (np.ndarray): The original image for background.
    """

    if 'depth' not in next(iter(graph.edges(data=True)))[2]:
        logger.info("Graph edges do not have 'depth' attribute, computing depth...")
        graph = generate_graph_depth(graph)

    background = np.zeros((*img.shape, 3), dtype=np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(background, cmap='gray')

    # Normalize depth values for coloring
    depths = list(nx.get_edge_attributes(graph, 'depth').values())
    min_depth = min(depths)
    max_depth = max(depths)
    norm = plt.Normalize(vmin=min_depth, vmax=max_depth)
    cmap = plt.get_cmap('coolwarm')

    for u, v, data in graph.edges(data=True):
        depth = graph.get_edge_data(u, v)[0]['depth']
        color = cmap(norm(depth))

        coords = np.array(data['centerline'])
        plt.plot(coords[:, 1], coords[:, 0], color=color, linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.axis('off')
    plt.colorbar(sm, label='Edge Depth', ax=plt.gca())
    plt.show()


def display_graph_mean_radius(graph: nx.MultiGraph,
                           img: np.ndarray
                           ) -> None:
    """
    Plot the graph, coloring edges based on their mean radius.

    Args:
        graph (nx.MultiGraph): The input graph with 'mean_radius' attribute on edges.
        img (np.ndarray): The original image for background.
    """

    if 'mean_radius' not in next(iter(graph.edges(data=True)))[2]:
        logger.warning("Graph edges do not have 'mean_radius' attribute.")
        return

    background = np.zeros((*img.shape, 3), dtype=np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(background, cmap='gray')

    # Extract mean radius values for coloring
    radii = [data['mean_radius'] for _, _, data in graph.edges(data=True)]
    max_radius = max(radii)
    min_radius = min(radii)
    norm = plt.Normalize(vmin=min_radius, vmax=max_radius)
    cmap = plt.get_cmap('plasma')

    for u, v, data in graph.edges(data=True):
        mean_radius = data['mean_radius']
        color = cmap(norm(mean_radius))

        coords = np.array(data['centerline'])
        plt.plot(coords[:, 1], coords[:, 0], color=color, linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.axis('off')
    plt.colorbar(sm, label='Mean Radius', ax=plt.gca())
    plt.show()


def display_max_radius_node(graph: nx.Graph,
                            img: np.ndarray
                            ) -> None:
    """Display the node with the maximum radius on the given image.

    Args:
        graph (nx.Graph): The input graph.
        img (np.ndarray): The original image.
    """

    max_node = find_max_radius_node(graph)
    if max_node is None:
        logger.warning("No nodes in the graph.")
        return

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')

    # Plot all edges
    for u, v, data in graph.edges(data=True):
        coords = np.array(data['centerline'])
        plt.plot(coords[:, 1], coords[:, 0], color='blue', linewidth=1)

    # Highlight the max radius node
    coords = graph.nodes[max_node]['pos']
    plt.scatter(coords[1], coords[0], color='red', s=100, label='Max Radius Node')
    plt.legend()
    plt.axis('off')
    plt.show()


def display_edges_hierarchy(graph: nx.MultiGraph,
                            img: np.ndarray
                            ) -> None:
    """
    Display the graph hierarchy overlayed on the image.

    Parameters:
        hierarchy (nx.MultiGraph): The graph representing the hierarchy.
        img (np.ndarray): The original image for background.
    """

    background = np.zeros((*img.shape, 3), dtype=np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(background, cmap='gray')

    # Normalize hierarchy values for coloring
    hierarchies = [data['hierarchy'] for _, _, data in graph.edges(data=True) if 'hierarchy' in data]
    if not hierarchies:
        logger.warning("No hierarchy data found in edges.")
        return
    
    max_hierarchy = max(hierarchies)
    norm = plt.Normalize(vmin=0, vmax=max_hierarchy)
    cmap = plt.get_cmap('coolwarm')

    for u, v in graph.edges():
        for val in graph.get_edge_data(u, v).values():
            if 'centerline' in val:
                hierarchy_level = val.get('hierarchy', 0)
                color = cmap(norm(hierarchy_level))

                coords = np.array(val['centerline'])
            plt.plot(coords[:, 1], coords[:, 0], color=color, linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.axis('off')
    plt.colorbar(sm, label='Hierarchy Level', ax=plt.gca())
    plt.show()


def display_topological_edge_classification(graph: nx.MultiGraph,
                                            img: np.ndarray
                                            ) -> None:
    """Display the topological classification of edges on the given image.

    Args:
        graph (nx.MultiGraph): The input graph.
        img (np.ndarray): The original image.
    """

    if graph.number_of_edges() == 0:
        logging.warning("Graph has no edges to display.")
        return

    if not any('topological_class' in d for _, _, d in graph.edges(data=True)):
        logging.info("At least one edge is missing 'topological_class' attribute, computing classification...")
        graph = generate_graph_topological_classification(graph)

    background = np.zeros((*img.shape, 3), dtype=np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(background, cmap='gray')
    
    for u, v in graph.edges():
        edge_data = graph.get_edge_data(u, v)
        for e, val in edge_data.items():
            class_name = val.get('topological_class', 'unknown')
            centerline = np.array(val.get('centerline', []))
            if class_name == TopologicalClass.NON_TOPOLOGICAL:
                color = [255, 0, 0]  # Red
            elif class_name == TopologicalClass.T0:
                color = [0, 255, 0]  # Green
            elif class_name == TopologicalClass.T1:
                color = [0, 0, 255]  # Blue
            else:
                color = [255, 255, 0]  # Yellow for unknown
            plt.plot(centerline[:,1], centerline[:,0], color=np.array(color)/255, linewidth=2)

    plt.axis('off')
    plt.show()

def display_edge_radius_decay(graph: nx.MultiGraph,
                              img: np.ndarray
                              ) -> None:
        """
        Display the graph with edges colored based on their radius decay.
    
        Parameters:
            graph (nx.MultiGraph): The input graph with 'radius_decay' attribute on edges.
            img (np.ndarray): The original image for background.
        """
    
        if 'radius_decay' not in next(iter(graph.edges(data=True)))[2]:
            logger.info("Graph edges do not have 'radius_decay' attribute.")
            graph = generate_graph_edge_radius_decay(graph)
    
        background = np.zeros((*img.shape, 3), dtype=np.uint8)
        plt.figure(figsize=(8, 8))
        plt.imshow(background, cmap='gray')
    
        # Extract radius decay values for coloring
        decays = [data['radius_decay'] for _, _, data in graph.edges(data=True)]
        max_decay = max(decays)
        min_decay = min(decays)
        norm = plt.Normalize(vmin=min_decay, vmax=max_decay)
        cmap = plt.get_cmap('viridis')
    
        for u, v, data in graph.edges(data=True):
            radius_decay = data['radius_decay']
            color = cmap(norm(radius_decay))
    
            coords = np.array(data['centerline'])
            plt.plot(coords[:, 1], coords[:, 0], color=color, linewidth=2)
    
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.axis('off')
        plt.colorbar(sm, label='Radius Decay', ax=plt.gca())
        plt.show()


def display_histogram(data: dict,
                          title: str,
                          xlabel: str,
                          ylabel: str
                          ) -> None:
    """
    Display a histogram from the given data.

    Parameters:
        data (dict): A dictionary with keys as categories and values as counts.
        title (str): The title of the histogram.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """

    plt.figure(figsize=(8, 6))
    plt.bar(data.keys(), data.values())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def display_attribute_histogram(graph: nx.MultiGraph,
                                attribute: str
                                ) -> None:
    """
    Display the histogram of a specified edge attribute in the graph.

    Parameters:
        graph (nx.MultiGraph): The input vascular graph.
        attribute (str): The edge attribute to compute the histogram for.
    """
    hist = graph_attribute_histogram(graph, attribute)
    display_histogram(hist, f'Graph {attribute.capitalize()} Histogram', attribute.capitalize(), 'Number of Edges')


def display_graph_statistics(graph: nx.MultiGraph) -> None:
    """
    Display various statistics of the graph including histograms for depth and hierarchy.

    Parameters:
        graph (nx.MultiGraph): The input vascular graph.
    """
    depth_hist = graph_depth_histogram(graph)
    hierarchy_hist = graph_hierarchy_histogram(graph)
    degree_hist = graph_nodes_degree_histogram(graph)
    radius_hist = graph_radius_histogram(graph)

    display_histogram(depth_hist, 'Graph Depth Histogram', 'Depth Level', 'Number of Edges')
    display_histogram(hierarchy_hist, 'Graph Hierarchy Histogram', 'Hierarchy Level', 'Number of Edges')
    display_histogram(degree_hist, 'Graph Node Degree Histogram', 'Node Degree', 'Number of Nodes')
    display_histogram(radius_hist, 'Graph Radius Histogram', 'Radius', 'Number of Edges')


def display_histogram_difference(hist_1: dict,
                                 hist_2: dict,
                                 title: str,
                                 xlabel: str,
                                 ylabel: str
                                 ) -> None:
    """
    Display the difference between histograms of two graphs.

    Parameters:
        graph_1 (nx.MultiGraph): The first input vascular graph.
        graph_2 (nx.MultiGraph): The second input vascular graph.
        title (str): The title of the histogram.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """

    diff_hist = get_histogram_difference(hist_1, hist_2)
    display_histogram(diff_hist, title, xlabel, ylabel)


def display_attribute_histogram_difference(graph_1: nx.MultiGraph,
                                           graph_2: nx.MultiGraph,
                                           attribute: str
                                           ) -> None:
     """
     Display the difference in histograms of a specified edge attribute between two graphs.
    
     Parameters:
          graph_1 (nx.MultiGraph): The first input vascular graph.
          graph_2 (nx.MultiGraph): The second input vascular graph.
          attribute (str): The edge attribute to compute the histogram for.
     """
    
     hist_1 = graph_attribute_histogram(graph_1, attribute)
     hist_2 = graph_attribute_histogram(graph_2, attribute)
    
     display_histogram_difference(hist_1, hist_2,
                                    f'{attribute.capitalize()} Histogram Difference',
                                    attribute.capitalize(),
                                    'Difference in Number of Edges')
