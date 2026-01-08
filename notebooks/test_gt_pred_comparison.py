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
#     display_name: graph-neural-networks
#     language: python
#     name: python3
# ---

# %%
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

# %%
pred_dir = "/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds"
gt_dir = "/home/morand/afs/EVAPORE/data/FIVES/gt"

pred_filenames = sorted(os.listdir(pred_dir))
gt_filenames = sorted(os.listdir(gt_dir))

pred_filenames = [f for f in pred_filenames if f.endswith('.png')]

pred_paths = [os.path.join(pred_dir, f) for f in pred_filenames]
gt_paths = [os.path.join(gt_dir, f) for f in gt_filenames]

print(f"Number of prediction files: {len(pred_paths)}")
print(f"Number of ground truth files: {len(gt_paths)}")


# %%
pred_path, gt_path = pred_paths[0], gt_paths[0]
print("Prediction path:", pred_path)
print("Ground Truth path:", gt_path)

# %%
gt = np.array(Image.open(gt_path))[:,:,0]
pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))

plt.imshow(pred)
plt.title("Prediction")
plt.show()

plt.imshow(gt)
plt.title("Ground Truth")
plt.show()

# %%
from graph.graph_creation import img_to_graph

pred_graph = img_to_graph(pred)
gt_graph = img_to_graph(gt)

# %%
from graph.graph_visualization import display_graph_overlay

display_graph_overlay(pred, pred_graph)
display_graph_overlay(gt, gt_graph)


# %%
def compute_matches(graph_1, graph_2):
    matches = {}
    for i_1, data_1 in graph_1.nodes(data=True):
        pos_1, rad_1 = np.array(data_1['pos']), data_1['radius']

        min_i = None
        min_dist = float('inf')

        for i_2, data_2 in graph_2.nodes(data=True):
            pos_2, rad_2 = np.array(data_2['pos']), data_2['radius']

            dist = np.linalg.norm(pos_1 - pos_2)

            if dist < (rad_1 + rad_2):
                if dist < min_dist:
                    min_dist = dist
                    min_i = i_2

        if min_i is not None:
            matches[i_1] = min_i

    return matches

p_to_g_matches = compute_matches(pred_graph, gt_graph)
g_to_p_matches = compute_matches(gt_graph, pred_graph)
print("Prediction to Ground Truth matches:", p_to_g_matches)
print("Ground Truth to Prediction matches:", g_to_p_matches)

reciprocal_matches = {p: g for p, g in p_to_g_matches.items() if g in g_to_p_matches and g_to_p_matches[g] == p}
print("Reciprocal matches:", reciprocal_matches)
reciprocal_matches_t = {g: p for p, g in reciprocal_matches.items()}


# %%
def plot_matches(pred, gt, pred_graph, gt_graph, matches):
    plt.figure(figsize=(20,20))

    background = np.zeros(gt.shape + (3,), dtype=np.uint8)
    background[:,:,0] = gt
    background[:,:,1] = pred
    background[:,:,2] = pred
    plt.imshow(background)

    for i_p, i_g in matches.items():
        pos_p = pred_graph.nodes[i_p]['pos']
        pos_g = gt_graph.nodes[i_g]['pos']

        radius = int((pred_graph.nodes[i_p]['radius'] + gt_graph.nodes[i_g]['radius']) / 2)

        circle_p = plt.Circle((pos_p[1], pos_p[0]), radius, color='yellow', fill=False, linewidth=2)
        circle_g = plt.Circle((pos_g[1], pos_g[0]), radius, color='cyan', fill=False, linewidth=2)
        plt.gca().add_patch(circle_p)
        plt.gca().add_patch(circle_g)

    plt.title("Matches between Prediction and Ground Truth")
    plt.show()


# %%
plot_matches(pred, gt, pred_graph, gt_graph, reciprocal_matches)

# %%
edges_match = []

for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
    if reciprocal_matches_t.get(n_g_1) is not None and reciprocal_matches_t.get(n_g_2) is not None:
        n_p_1 = reciprocal_matches_t[n_g_1]
        n_p_2 = reciprocal_matches_t[n_g_2]

        if pred_graph.has_edge(n_p_1, n_p_2):
            edges_match.append((n_g_1, n_g_2))


print("Number of ground truth edges:", gt_graph.number_of_edges())
print("Number of matching edges:", len(edges_match))        


# %%
plt.figure(figsize=(20,20))
background = np.zeros(gt.shape + (3,), dtype=np.uint8)
background[:,:,0] = gt
background[:,:,1] = pred
background[:,:,2] = pred
plt.imshow(background)

for n_g_1, n_g_2 in edges_match:
    edge_data = gt_graph.get_edge_data(n_g_1, n_g_2)[0]
    print(edge_data)
    centerline = np.array(edge_data['centerline'])
    plt.plot(centerline[:,1], centerline[:,0], color='green', linewidth=2)

plt.title("Matching Edges between Prediction and Ground Truth")
plt.show()


# %%
from skimage.morphology import skeletonize

skel_p = skeletonize(pred)
skel_g = skeletonize(gt)

plt.imshow(skel_p)
plt.title("Skeletonized Prediction")
plt.show()

plt.imshow(skel_g)
plt.title("Skeletonized Ground Truth")
plt.show()


# %%
p_skel_in_g = np.logical_and(skel_p, gt)
g_skel_in_p = np.logical_and(skel_g, pred)

plt.figure(figsize=(10,10))
plt.imshow(p_skel_in_g)
plt.title("Prediction Skeleton in Ground Truth")
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(g_skel_in_p)
plt.title("Ground Truth Skeleton in Prediction")
plt.show()

# %%
g_vasc_length = skel_g.sum()
p_vasc_length = skel_p.sum()

g_in_p_vasc_length = g_skel_in_p.sum()
p_in_g_vasc_length = p_skel_in_g.sum()

print("Ground Truth Vascular Length:", g_vasc_length)
print("Prediction Vascular Length:", p_vasc_length)

print("Ground Truth Skeleton in Prediction Length:", g_in_p_vasc_length)
print("Prediction Skeleton in Ground Truth Length:", p_in_g_vasc_length)

# %%
g_vasc_length_ratio = g_in_p_vasc_length / g_vasc_length if g_vasc_length > 0 else 0
print("Vascular Length Ratio (GT in Pred):", g_vasc_length_ratio)

p_vasc_length_ratio = p_in_g_vasc_length / p_vasc_length if p_vasc_length > 0 else 0
print("Vascular Length Ratio (Pred in GT):", p_vasc_length_ratio)

# %%
radius_map = np.zeros_like(pred, dtype=np.float32)
for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
    centerline = np.array(data_g['centerline'])
    radius = np.array(data_g['radius'])

    radius_map[centerline[:,0], centerline[:,1]] = radius

# %%
plt.figure(figsize=(10,10))
plt.imshow(radius_map, cmap='hot')
plt.title("Ground Truth Radius Map")
plt.colorbar(label='Radius')
plt.show()

# %%
radius_hist = radius_map[radius_map > 0]
plt.figure(figsize=(10,5))
plt.hist(radius_hist, bins=30, color='blue', alpha=0.7)
plt.title("Histogram of Vessel Radii in Ground Truth")
plt.xlabel("Radius")
plt.ylabel("Frequency")
plt.show()

# %%
g_skel_not_in_p = np.logical_and(skel_g, np.logical_not(pred))
plt.figure(figsize=(10,10))
plt.imshow(g_skel_not_in_p)
plt.title("Ground Truth Skeleton not in Prediction")
plt.show()


# %%
error_radius = (radius_map * g_skel_not_in_p)
radius_hist = (error_radius)[error_radius > 0]
plt.figure(figsize=(10,5))
plt.hist(radius_hist, bins=30, color='blue', alpha=0.7)
plt.title("Histogram of Vessel Radii in Ground Truth")
plt.xlabel("Radius")
plt.ylabel("Frequency")
plt.show()

# %%
from tqdm import tqdm

full_error_histogram = np.zeros(50, dtype=np.int64)
full_histogram = np.zeros(50, dtype=np.int64)
raw_radius_values = []
raw_error_radius_values = []

for pred_path, gt_path in tqdm(zip(pred_paths, gt_paths), total=len(pred_paths)):
    pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))
    gt = np.array(Image.open(gt_path))[:,:,0]

    pred_graph = img_to_graph(pred)
    gt_graph = img_to_graph(gt)

    skel_p = skeletonize(pred)
    skel_g = skeletonize(gt)

    radius_map = np.zeros_like(pred, dtype=np.float32)
    for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
        centerline = np.array(data_g['centerline'])
        radius = np.array(data_g['radius'])

        radius_map[centerline[:,0], centerline[:,1]] = radius

    radius_hist = radius_map[radius_map > 0]
    #raw_radius_values.extend(radius_hist)
    full_histogram += np.histogram(radius_hist, bins=50, range=(0, 50))[0]

    g_skel_not_in_p = np.logical_and(skel_g, np.logical_not(pred))
    error_radius = (radius_map * g_skel_not_in_p)
    error_radius_hist = error_radius[error_radius > 0]
    #raw_error_radius_values.extend(error_radius_hist)
    full_error_histogram += np.histogram(error_radius_hist, bins=50, range=(0, 50))[0]


# %%
limit = 30

normalized_histogram = (full_histogram / full_histogram.sum())[:limit]
normalized_error_histogram = (full_error_histogram / full_error_histogram.sum())[:limit]

plt.figure(figsize=(10,5))
plt.bar(np.arange(limit), normalized_histogram, color='blue', alpha=0.6)
plt.bar(np.arange(limit), normalized_error_histogram, color='red', alpha=0.6)
plt.title("Normalized Cumulative Histogram of Vessel Radii in Ground Truth across Dataset")
plt.xlabel("Radius")
plt.ylabel("Normalized Frequency")
plt.show()

# %%
from scipy.stats import ks_2samp

print(len(raw_radius_values), len(raw_error_radius_values))

stat, p = ks_2samp(raw_radius_values, raw_error_radius_values)
print("KS statistic:", stat)
print("p-value:", p)

# %%
from graph.graph_labeling import generate_graph_depth

full_depth_error_histogram = np.zeros(50, dtype=np.int64)
full_depth_histogram = np.zeros(50, dtype=np.int64)
raw_depth_values = []
raw_error_depth_values = []

for pred_path, gt_path in tqdm(zip(pred_paths, gt_paths), total=len(pred_paths)):
    pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))
    gt = np.array(Image.open(gt_path))[:,:,0]

    pred_graph = img_to_graph(pred)
    gt_graph = img_to_graph(gt)

    pred_graph = generate_graph_depth(pred_graph)
    gt_graph = generate_graph_depth(gt_graph)

    skel_p = skeletonize(pred)
    skel_g = skeletonize(gt)

    depth_map = np.zeros_like(pred, dtype=np.float32)
    for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
        centerline = np.array(data_g['centerline'])
        depth = np.array(data_g['depth'])

        depth_map[centerline[:,0], centerline[:,1]] = depth

    depth_hist = depth_map[depth_map > 0]
    #raw_depth_values.extend(depth_hist)
    full_depth_histogram += np.histogram(depth_hist, bins=50, range=(0, 50))[0]

    g_skel_not_in_p = np.logical_and(skel_g, np.logical_not(pred))
    error_depth = (depth_map * g_skel_not_in_p)
    error_depth_hist = error_depth[error_depth > 0]
    #raw_error_depth_values.extend(error_depth_hist)
    full_depth_error_histogram += np.histogram(error_depth_hist, bins=50, range=(0, 50))[0]

# %%
limit = 30
normalized_depth_histogram = (full_depth_histogram / full_depth_histogram.sum())[:limit]
normalized_depth_error_histogram = (full_depth_error_histogram / full_depth_error_histogram.sum())[:limit]

plt.figure(figsize=(10,5))
plt.bar(np.arange(limit), normalized_depth_histogram, color='blue', alpha=0.6)
plt.bar(np.arange(limit), normalized_depth_error_histogram, color='red', alpha=0.6)
plt.title("Normalized Cumulative Histogram of Vessel Depth in Ground Truth across Dataset")
plt.xlabel("Depth")
plt.ylabel("Normalized Frequency")
plt.show()

# %%
print(len(raw_depth_values), len(raw_error_depth_values))

stat, p = ks_2samp(raw_depth_values, raw_error_depth_values)
print("KS statistic:", stat)
print("p-value:", p)

# %%
from graph.graph_labeling import generate_graph_hierarchy
from graph.graph_utils import find_max_radius_node

full_hierarchy_error_histogram = np.zeros(50, dtype=np.int64)
full_hierarchy_histogram = np.zeros(50, dtype=np.int64)
raw_hierarchy_values = []
raw_error_hierarchy_values = []

for pred_path, gt_path in tqdm(zip(pred_paths, gt_paths), total=len(pred_paths)):
    pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))
    gt = np.array(Image.open(gt_path))[:,:,0]

    pred_graph = img_to_graph(pred)
    gt_graph = img_to_graph(gt)

    pred_root = find_max_radius_node(pred_graph)
    gt_root = find_max_radius_node(gt_graph)

    pred_graph = generate_graph_hierarchy(pred_graph, pred_root)
    gt_graph = generate_graph_hierarchy(gt_graph, gt_root)

    skel_p = skeletonize(pred)
    skel_g = skeletonize(gt)

    hierarchy_map = np.zeros_like(pred, dtype=np.float32)
    for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
        centerline = np.array(data_g['centerline'])
        if 'hierarchy' not in data_g:
            continue
        hierarchy = np.array(data_g['hierarchy'])

        hierarchy_map[centerline[:,0], centerline[:,1]] = hierarchy

    hierarchy_hist = hierarchy_map[hierarchy_map > 0]
    #raw_hierarchy_values.extend(hierarchy_hist)
    full_hierarchy_histogram += np.histogram(hierarchy_hist, bins=50, range=(0, 50))[0]

    g_skel_not_in_p = np.logical_and(skel_g, np.logical_not(pred))
    error_hierarchy = (hierarchy_map * g_skel_not_in_p)
    error_hierarchy_hist = error_hierarchy[error_hierarchy > 0]
    #raw_error_hierarchy_values.extend(error_hierarchy_hist)
    full_hierarchy_error_histogram += np.histogram(error_hierarchy_hist, bins=50, range=(0, 50))[0]

# %%
limit = 30
normalized_hierarchy_histogram = (full_hierarchy_histogram / full_hierarchy_histogram.sum())[:limit]
normalized_hierarchy_error_histogram = (full_hierarchy_error_histogram / full_hierarchy_error_histogram.sum())[:limit]

plt.figure(figsize=(10,5))
plt.bar(np.arange(limit), normalized_hierarchy_histogram, color='blue', alpha=0.6)
plt.bar(np.arange(limit), normalized_hierarchy_error_histogram, color='red', alpha=0.6)
plt.title("Normalized Cumulative Histogram of Vessel Hierarchy in Ground Truth across Dataset")
plt.xlabel("Hierarchy")
plt.ylabel("Normalized Frequency")
plt.show()

# %%
print(len(raw_hierarchy_values), len(raw_error_hierarchy_values))

stat, p = ks_2samp(raw_hierarchy_values, raw_error_hierarchy_values)
print("KS statistic:", stat)
print("p-value:", p)

# %%
from graph.graph_labeling import generate_graph_topological_classification
from graph.graph_utils import TopologicalClass
from graph.graph_visualization import display_topological_edge_classification

full_topological_error_histogram = np.zeros(3, dtype=np.int64)
full_topological_histogram = np.zeros(3, dtype=np.int64)
raw_topology_values = []
raw_error_topology_values = []

topological_class_index = {
    TopologicalClass.NON_TOPOLOGICAL: 0,
    TopologicalClass.T0: 1,
    TopologicalClass.T1: 2
}

for pred_path, gt_path in tqdm(zip(pred_paths, gt_paths), total=len(pred_paths)):
    pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))
    gt = np.array(Image.open(gt_path))[:,:,0]

    pred_graph = img_to_graph(pred)
    gt_graph = img_to_graph(gt)

    pred_graph = generate_graph_topological_classification(pred_graph)
    gt_graph = generate_graph_topological_classification(gt_graph)

    skel_p = skeletonize(pred)
    skel_g = skeletonize(gt)

    topological_map = np.zeros_like(pred, dtype=np.float32)
    for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
        centerline = np.array(data_g['centerline'])
        topological_class = topological_class_index[data_g['topological_class']]

        topological_map[centerline[:,0], centerline[:,1]] = topological_class + 1  # +1 to avoid zero

    topological_hist = topological_map[topological_map > 0]
    #raw_topology_values.extend(topological_hist)
    full_topological_histogram += np.histogram(topological_hist, bins=3, range=(1, 4))[0]

    g_skel_not_in_p = np.logical_and(skel_g, np.logical_not(pred))
    error_topological = (topological_map * g_skel_not_in_p)
    error_topological_hist = error_topological[error_topological > 0]
    #raw_error_topology_values.extend(error_topological_hist)
    full_topological_error_histogram += np.histogram(error_topological_hist, bins=3, range=(1, 4))[0]

# %%
normalized_topological_histogram = full_topological_histogram / full_topological_histogram.sum()
normalized_topological_error_histogram = full_topological_error_histogram / full_topological_error_histogram.sum()

plt.figure(figsize=(10,5))
plt.bar(np.arange(3), normalized_topological_histogram, color='blue', alpha=0.6)
plt.bar(np.arange(3), normalized_topological_error_histogram, color='red', alpha=0.6)
plt.title("Normalized Cumulative Histogram of Vessel Topology in Ground Truth across Dataset")
plt.xlabel("Topology")
plt.ylabel("Normalized Frequency")
plt.xticks([0,1,2], ['NON_TOPOLOGICAL', 'T0', 'T1'])
plt.show()

# %%
print(len(raw_topology_values), len(raw_error_topology_values))

stat, p = ks_2samp(raw_topology_values, raw_error_topology_values)
print("KS statistic:", stat)
print("p-value:", p)

# %% [markdown]
# # Correlation between attributes

# %%
radius_values = []
depth_values = []
#hierarchy_values = []
topology_values = []

for pred_path, gt_path in tqdm(zip(pred_paths, gt_paths), total=len(pred_paths)):
    pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))
    gt = np.array(Image.open(gt_path))[:,:,0]

    pred_graph = img_to_graph(pred)
    gt_graph = img_to_graph(gt)

    pred_graph = generate_graph_topological_classification(pred_graph)
    gt_graph = generate_graph_topological_classification(gt_graph)

    #pred_graph = generate_graph_hierarchy(pred_graph, find_max_radius_node(pred_graph))
    #gt_graph = generate_graph_hierarchy(gt_graph, find_max_radius_node(gt_graph))

    pred_graph = generate_graph_depth(pred_graph)
    gt_graph = generate_graph_depth(gt_graph)

    for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
        radius = data_g['radius']
        depth = data_g['depth']
        #hierarchy = data_g['hierarchy']
        topology = topological_class_index[data_g['topological_class']]

        for i in range(len(data_g['centerline'])):
            radius_values.append(radius[i])
            depth_values.append(depth)
            #hierarchy_values.append(hierarchy)
            topology_values.append(topology)

    break

# %%
print(len(radius_values), len(depth_values),len(topology_values))

# %%
from scipy.stats import pearsonr

r_radius_depth, p_value_radius_depth = pearsonr(radius_values, depth_values)
r_radius_topology, p_value_radius_topology = pearsonr(radius_values, topology_values)
r_depth_topology, p_value_depth_topology = pearsonr(depth_values, topology_values)

print("Correlation between Radius and Depth: r =", r_radius_depth, ", p-value =", p_value_radius_depth)
print("Correlation between Radius and Topology: r =", r_radius_topology, ", p-value =", p_value_radius_topology)
print("Correlation between Depth and Topology: r =", r_depth_topology, ", p-value =", p_value_depth_topology)

# %% [markdown]
# # Test de sampling en fonction du radius

# %%
from scipy.interpolate import interp1d

x = np.arange(50)
y = full_error_histogram.astype(np.float64)

plt.plot(x, y, 'o', )
plt.show()

# %%
weighted_y = y
weighted_y[full_histogram != 0] /= full_histogram[full_histogram != 0]

plt.bar(x, weighted_y)
plt.show()

# %%
f = interp1d(x, weighted_y, kind='linear')

xnew = np.arange(0, 49, 0.1)
ynew = f(xnew)

plt.plot(x, weighted_y, 'o', )
plt.plot(xnew, ynew, '-')
plt.show()

# %%
gt = np.array(Image.open(gt_path))[:,:,0]
pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))

pred_graph = img_to_graph(pred)
gt_graph = img_to_graph(gt)

# %%
remove_ratio = 0.1

n_to_remove = int(remove_ratio * gt_graph.number_of_edges())

ids = []
mean_radius = []

for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
    ids.append(data_g['id'])
    mean_radius.append(data_g['mean_radius'])

print(ids)
print(mean_radius)

mean_radius = np.array(mean_radius)
weights = f(mean_radius) 
weights /= weights.sum()

to_remove = np.random.choice(ids, size=n_to_remove, replace=False, p=weights)

print("Number of edges to remove:", n_to_remove)
print("Edges to remove:", to_remove)

# %%
radius_to_remove = []

for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
    id = data_g['id']
    radius = data_g['mean_radius']

    if id in to_remove:
        radius_to_remove.append(radius)

# %%
plt.hist(radius_to_remove, bins=30, color='blue', alpha=0.7)

# %%
remove_ratio = 0.1

full_remove_histogram = np.zeros(50, dtype=np.int64)

for pred_path, gt_path in tqdm(zip(pred_paths, gt_paths), total=len(pred_paths)):
    pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))
    gt = np.array(Image.open(gt_path))[:,:,0]

    pred_graph = img_to_graph(pred)
    gt_graph = img_to_graph(gt)

    n_to_remove = int(remove_ratio * gt_graph.number_of_edges())

    ids = []
    mean_radius = []

    for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
        ids.append(data_g['id'])
        mean_radius.append(data_g['mean_radius'])

    mean_radius = np.array(mean_radius)
    weights = f(mean_radius) 
    weights /= weights.sum()

    to_remove = np.random.choice(ids, size=n_to_remove, replace=False, p=weights)

    radius_to_remove = []

    for n_g_1, n_g_2, data_g in gt_graph.edges(data=True):
        id = data_g['id']
        radius = data_g['mean_radius']

        if id in to_remove:
            radius_to_remove.append(radius)

    full_remove_histogram += np.histogram(radius_to_remove, bins=50, range=(0, 50))[0]

# %%
plt.figure(figsize=(10,5))
plt.bar(np.arange(50), full_remove_histogram, color='blue', alpha=0.6)
plt.show()

# %%
