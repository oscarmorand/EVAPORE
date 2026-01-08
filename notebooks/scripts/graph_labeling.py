# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: test_env
#     language: python
#     name: python3
# ---

# %%
import logging
import numpy as np
import json
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt

# %%
img = np.array(Image.open("/home/morand/afs/datasets/FIVES/train/Ground truth/1_A.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/New Piskel.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/simple_loop_two.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/error_loop_branch.png"))[:,:,0]

print(img.shape)

plt.imshow(img, cmap='gray')
plt.show()

# %%
from graph.graph_creation import img_to_graph

graph, pixel_G = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=True)

# %%
from graph.graph_io import save_graph_to_dot

save_graph_to_dot(graph, "/home/morand/afs/tests/test.dot")

# %%
print(nx.get_edge_attributes(graph, 'mean_radius'))

# %%
from graph.graph_visualization import display_graph_overlay

display_graph_overlay(img, graph)

# %%
display_graph_overlay(img, pixel_G, show_edges=False)


# %%
def generate_graph_depth(graph: nx.Graph) -> None:
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


# %%
graph = generate_graph_depth(graph)

print(nx.get_edge_attributes(graph, 'depth'))


# %%
def plot_graph_edge_depth(graph: nx.Graph, img: np.ndarray) -> None:
    """
    Plot the graph overlayed on the image, coloring edges based on their depth.

    Parameters:
    - graph: The input graph.
    - degrees: A dictionary mapping edge IDs to their depth values.
    - img: The original image for background.
    """
    if 'depth' not in next(iter(graph.edges(data=True)))[2]:
        print("Graph edges do not have 'depth' attribute.")
        return

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


# %%
plot_graph_edge_depth(graph, img)

# %%
from graph.graph_stats import graph_depth_histogram

hist = graph_depth_histogram(graph)

plt.bar(hist.keys(), hist.values())
plt.xlabel('Depth Level')
plt.ylabel('Number of Edges')
plt.title('Graph Depth Histogram')
plt.show()

# %%
from graph.graph_io import save_graph_to_json

save_graph_to_json(graph, "/home/morand/afs/tests/new_piskel.json")


# %%
def plot_graph_mean_radius(graph: nx.Graph, img: np.ndarray) -> None:
    """
    Plot the graph, coloring edges based on their mean radius.

    Args:
        graph (nx.Graph): The input graph with 'mean_radius' attribute on edges.
        img (np.ndarray): The original image for background.
    """

    if 'mean_radius' not in next(iter(graph.edges(data=True)))[2]:
        print("Graph edges do not have 'mean_radius' attribute.")
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


# %%
plot_graph_mean_radius(graph, img)


# %%
def find_max_radius_node(graph: nx.Graph) -> int:
    max_radius = 0
    max_node = None

    for n in graph.nodes(data=True):
        radius = n[1].get('radius', 0)
        if radius > max_radius:
            max_radius = radius
            max_node = n[0]

    return max_node


# %%
max_node = find_max_radius_node(graph)
print(f"Node with maximum radius: {max_node}, Radius: {graph.nodes[max_node]['radius']}")


# %%
def display_max_radius_node(graph: nx.Graph, img: np.ndarray) -> None:
    max_node = find_max_radius_node(graph)
    if max_node is None:
        print("No nodes in the graph.")
        return

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')

    # Plot all edges
    for u, v, data in graph.edges(data=True):
        coords = np.array(data['centerline'])
        plt.plot(coords[:, 1], coords[:, 0], color='blue', linewidth=1)

    # Highlight the max radius node
    print(graph.nodes[max_node])
    coords = graph.nodes[max_node]['pos']
    plt.scatter(coords[1], coords[0], color='red', s=100, label='Max Radius Node')
    plt.legend()
    plt.axis('off')
    plt.show()


# %%
display_max_radius_node(graph, img)


# %%
def generate_graph_hierarchy(graph: nx.Graph, root: int) -> nx.DiGraph:
    """
    Generate a graph representing the hierarchy from the given root node, with a hierarchy attribute on edges and nodes.

    Parameters:
    - graph: The input undirected graph.
    - root: The root node from which to generate the hierarchy.

    Returns:
    - A graph representing the hierarchy.
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


# %%
graph_hierarchy = generate_graph_hierarchy(graph, max_node)

print(nx.get_edge_attributes(graph_hierarchy, 'hierarchy'))


# %%
def display_edges_hierarchy(graph: nx.Graph, img: np.ndarray) -> None:
    """
    Display the graph hierarchy overlayed on the image.

    Parameters:
    - hierarchy: The graph representing the hierarchy.
    - img: The original image for background.
    """
    background = np.zeros((*img.shape, 3), dtype=np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(background, cmap='gray')

    # Normalize hierarchy values for coloring
    hierarchies = [data['hierarchy'] for _, _, data in graph.edges(data=True) if 'hierarchy' in data]
    if not hierarchies:
        print("No hierarchy data found in edges.")
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


# %%
display_edges_hierarchy(graph_hierarchy, img)

# %%
from graph.graph_stats import graph_hierarchy_histogram

hist = graph_hierarchy_histogram(graph_hierarchy)

plt.bar(hist.keys(), hist.values())
plt.xlabel('Hierarchy Level')
plt.ylabel('Number of Edges')
plt.title('Graph Hierarchy Histogram')
plt.show()

# %%
from graph.graph_visualization import display_graph_statistics

display_graph_statistics(graph_hierarchy)

# %%
from graph.graph_stats import get_graph_average_radius, get_graph_total_length

avg_radius = get_graph_average_radius(graph)
total_length = get_graph_total_length(graph)

print(f"Average Radius: {avg_radius}")
print(f"Total Length: {total_length}")

# %%
from graph.graph_visualization import display_edge_radius_decay

display_edge_radius_decay(graph, img)

# %%
skel = np.zeros_like(img)
for u, v, data in graph.edges(data=True):
    coords = np.array(data['centerline']).astype(int)
    skel[coords[:, 0], coords[:, 1]] = 255

plt.imshow(skel, cmap='gray')
plt.show()

# %%
from scipy.ndimage import sobel, gaussian_filter

plt.imshow(skel[750:1250, 1250:1750], cmap='gray')
plt.show()

skel_g = gaussian_filter(skel.astype(float), sigma=0.5)
plt.imshow(skel_g[750:1250, 1250:1750], cmap='gray')
plt.show()

skel = skel_g

grad_x = sobel(skel, axis=0)
grad_y = sobel(skel, axis=1)

plt.imshow(grad_x[750:1250, 1250:1750], cmap='gray')
plt.colorbar(label='Gradient X')
plt.show()

plt.imshow(grad_y[750:1250, 1250:1750], cmap='gray')
plt.colorbar(label='Gradient Y')
plt.show()

magnitude = (grad_x**2 + grad_y**2)**0.5
angles = np.arctan2(grad_y, grad_x)

plt.imshow(magnitude[750:1250, 1250:1750], cmap='gray')
plt.colorbar(label='Magnitude')
plt.show()

plt.imshow(angles[750:1250, 1250:1750])
plt.colorbar(label='Angle (radians)')
plt.show()


# %%

# %%
def curvature_from_coords(x, y):
    # Compute first and second derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    # Curvature formula for parametric curves
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

curvatures = np.zeros_like(img, dtype=float)

for u, v, data in graph.edges(data=True):
    coords = np.array(data['centerline'])

    x = coords[:, 1]
    y = coords[:, 0]

    curvature = curvature_from_coords(x, y)

    curvatures[coords[:, 0], coords[:, 1]] = curvature

curvatures = gaussian_filter(curvatures, sigma=2.0)
plt.imshow(curvatures[750:1250, 1250:1750], cmap='jet')
plt.colorbar(label='Curvature')
plt.show()


# %%

def fit_circle(x, y):
    """Fit circle to x,y points and return center (xc, yc) and radius R."""
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0], c[1]
    R = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, R

def local_curvature_signed(x, y, window=5):
    """Compute signed curvature using local circle fitting and tangent orientation."""
    n = len(x)
    curvature = np.zeros(n)
    for i in range(window, n-window):
        xw = x[i-window:i+window+1]
        yw = y[i-window:i+window+1]
        try:
            xc, yc, R = fit_circle(xw, yw)
            kappa = 1.0 / R
            # Determine sign via tangent direction
            v1 = np.array([x[i] - x[i-1], y[i] - y[i-1], 0])
            v2 = np.array([x[i+1] - x[i], y[i+1] - y[i], 0])
            cross = np.cross(v1, v2)
            curvature[i] = np.sign(cross[2]) * kappa
        except np.linalg.LinAlgError:
            curvature[i] = 0
    return curvature

curvatures = np.zeros_like(img, dtype=float)
for u, v, data in graph.edges(data=True):
    coords = np.array(data['centerline'])

    x = coords[:, 1]
    y = coords[:, 0]

    curvature = local_curvature_signed(x, y, window=10)

    curvatures[coords[:, 0], coords[:, 1]] = curvature


curvatures = gaussian_filter(curvatures, sigma=1.0)
plt.imshow(curvatures[850:1150, 1350:1650], cmap='seismic', vmin=-0.05, vmax=0.05)
plt.colorbar(label='Curvature (circle fitting)')
plt.show()

# %%
