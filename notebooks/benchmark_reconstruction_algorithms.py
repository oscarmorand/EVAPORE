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
#     display_name: graph-neural-networks (3.12.11)
#     language: python
#     name: python3
# ---

# %%
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from graph.graph_creation import img_to_graph
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
from graph_neural_networks.data.dataset.dynamic.graph_transforms.oversample_nodes import OversampleNodesTransform
from graph_neural_networks.data.dataset.dynamic.graph_transforms.compute_distance_matrix import ComputeDistanceMatrixTransform
from graph_neural_networks.data.dataset.dynamic.apply_graph_transforms import apply_graph_transforms
from graph_neural_networks.data.utils.pred_state import get_combined_graph
from graph_neural_networks.data.utils.pred_state import EdgePredState


# %%
def compute_geodesic_length(path: np.ndarray) -> float:
    diffs = np.diff(path, axis=0)              # (N-1, 2)
    return np.linalg.norm(diffs, axis=1).sum()

def compute_geodesic_length_difference_metric(ref_path: np.ndarray,
                                     pred_path: np.ndarray
) -> float:
    ref_length = compute_geodesic_length(ref_path)
    pred_length = compute_geodesic_length(pred_path)
    return abs(ref_length - pred_length)

def compute_path_distances(ref_path: np.ndarray, 
                          pred_path: np.ndarray
) -> float:
    # pred_path: (P, 2)
    # ref_path:  (R, 2)

    diff = pred_path[:, None, :] - ref_path[None, :, :]   # (P, R, 2)
    dists = np.linalg.norm(diff, axis=2)                  # (P, R)

    min_dists = np.min(dists, axis=1)                      # (P,)

    average_min_distance = np.mean(min_dists)
    hausdorff_distance = np.max(min_dists)

    return average_min_distance, hausdorff_distance

def compute_metrics(ref_path, pred_path):
    if not isinstance(ref_path, np.ndarray):
        ref_path = np.array(ref_path)
    if not isinstance(pred_path, np.ndarray):
        pred_path = np.array(pred_path)

    ref_n_pixels = ref_path.shape[0] - 1
    euclidean_dist = np.linalg.norm(ref_path[0] - ref_path[-1])

    average_min_distance, hausdorff_distance = compute_path_distances(ref_path, pred_path)
    average_min_distance_inv, hausdorff_distance_inv = compute_path_distances(pred_path, ref_path)
    average_min_distance_sym = (average_min_distance + average_min_distance_inv) / 2.0
    hausdorff_distance_sym = max(hausdorff_distance, hausdorff_distance_inv)

    ref_length = compute_geodesic_length(ref_path)
    pred_length = compute_geodesic_length(pred_path)
    geodesic_length_diff_metric = abs(ref_length - pred_length)

    metrics = {
        "average_min_distance": average_min_distance_sym,
        "hausdorff_distance": hausdorff_distance_sym,
        "geodesic_length_diff_metric": geodesic_length_diff_metric
    }
    distances = {
        "euclidean": euclidean_dist,
        "n_pixels": ref_n_pixels,
        "geodesic": ref_length
    }

    return metrics, distances


# %%
path = [
    [0, 0],
    [1, 1],
    [1, 2],
    [0, 3]
]
path = np.array(path)

print("n_pixels:", path.shape[0] - 1)
print("euclidean:", np.linalg.norm(path[0] - path[-1]))
print("geodesic:", compute_geodesic_length(path))

# %%
used_metrics = ["average_min_distance","hausdorff_distance", "geodesic_length_diff_metric"]

# %%
dataset = "FIVES"
data_dir = f"/home/morand/afs/EVAPORE/data/{dataset}/"
gt_folder = f"{data_dir}/gt/"
pred_folder = f"{data_dir}/pred/"
img_folder = f"{data_dir}/img/"
prob_map_folder = f"{data_dir}/probability_maps/"

gt_path_list = os.listdir(gt_folder)
gt_path_list.sort()
print(len(gt_path_list))

# %%
train_transforms = {
        "compute_distance_matrix_transform": ComputeDistanceMatrixTransform(skip_first_neighbor=True),
        "oversample_nodes_transform": OversampleNodesTransform(max_dist=50, remove_original_edges=True),
}
eval_transforms = {
    "oversample_nodes_transform": OversampleNodesTransform(max_dist=50, remove_original_edges=True),
}

# %%
from graph_neural_networks.reconstruction.reconstruction_method import PathReconstructionMethod
from graph_neural_networks.reconstruction.path_reconstruction.min_energy_path_reconstruction import SquaredMinEnergyPathReconstructionMethod
from graph_neural_networks.reconstruction.path_reconstruction.min_energy_path_reconstruction import ClassicMinEnergyPathReconstructionMethod
from graph_neural_networks.reconstruction.path_reconstruction.min_energy_path_reconstruction import MultiChannelSquaredMinEnergyPathReconstructionMethod
from graph_neural_networks.reconstruction.path_reconstruction.dahu_distance_path_reconstruction import DahuDistancePathReconstructionMethod
from graph_neural_networks.reconstruction.path_reconstruction.euclidean_path_reconstruction import EuclideanPathReconstructionMethod

from skimage.filters import gaussian

class PathReconstructionMethodInstance:
    def __init__(self, name: str, method: PathReconstructionMethod, map_fn: callable):
        self.name = name
        self.method = method
        self.map_fn = map_fn

'''
reconstruction_methods: list[PathReconstructionMethodInstance] = [
    PathReconstructionMethodInstance(
        "Euclidean", 
        EuclideanPathReconstructionMethod(), 
        lambda prob_map, img: prob_map
    ),

    #PathReconstructionMethodInstance(
    #    "DahuDistanceOnProbMap", 
    #    DahuDistancePathReconstructionMethod(), 
    #    lambda prob_map, img: prob_map
    #),

    #PathReconstructionMethodInstance(
    #    "DahuDistanceOnGreenChannel", 
    #    DahuDistancePathReconstructionMethod(), 
    #    lambda prob_map, img: img[1]
    #),
    
    PathReconstructionMethodInstance(
        "ClassicMinEnergyOnProbMap", 
        ClassicMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: prob_map
    ),

    PathReconstructionMethodInstance(
        "ClassicMinEnergyOnGreenChannel", 
        ClassicMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: img[1]
    ),

    PathReconstructionMethodInstance(
        "SquaredMinEnergyOnProbMap", 
        SquaredMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: prob_map
    ),

    PathReconstructionMethodInstance(
        "SquaredMinEnergyOnGreenChannel", 
        SquaredMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: img[1]
    ),
    PathReconstructionMethodInstance(
        "BluredSquaredMinEnergyOnProbMap", 
        SquaredMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: torch.tensor(gaussian(prob_map, sigma=5))
    ),

    PathReconstructionMethodInstance(
        "BluredSquaredMinEnergyOnGreenChannel", 
        SquaredMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: torch.tensor(gaussian(img[1], sigma=5))
    ),

    PathReconstructionMethodInstance(
        "MultiChannelSquaredMinEnergyOnProbMapAndGreenChannel", 
        MultiChannelSquaredMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: torch.cat([prob_map.unsqueeze(0), img[1].unsqueeze(0)], dim=0)
    )
]
'''

'''
reconstruction_methods: list[PathReconstructionMethodInstance] = [
    PathReconstructionMethodInstance(
        "Euclidean", 
        EuclideanPathReconstructionMethod(), 
        lambda prob_map, img: prob_map
    ),

    PathReconstructionMethodInstance(
        "DahuDistanceOnProbMap", 
        DahuDistancePathReconstructionMethod(), 
        lambda prob_map, img: prob_map
    ),

    PathReconstructionMethodInstance(
        "DahuDistanceOnGreenChannel", 
        DahuDistancePathReconstructionMethod(), 
        lambda prob_map, img: img[1]
    ),
    
    PathReconstructionMethodInstance(
        "ClassicMinEnergyOnProbMap", 
        ClassicMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: prob_map
    ),

    PathReconstructionMethodInstance(
        "ClassicMinEnergyOnGreenChannel", 
        ClassicMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: img[1]
    ),

    PathReconstructionMethodInstance(
        "SquaredMinEnergyOnProbMap", 
        SquaredMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: prob_map
    ),

    PathReconstructionMethodInstance(
        "SquaredMinEnergyOnGreenChannel", 
        SquaredMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: img[1]
    ),

    PathReconstructionMethodInstance(
        "MultiChannelSquaredMinEnergyOnProbMapAndGreenChannel", 
        MultiChannelSquaredMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img: torch.cat([prob_map.unsqueeze(0), img[1].unsqueeze(0)], dim=0)
    )
]
'''

reconstruction_methods: list[PathReconstructionMethodInstance] = [
    PathReconstructionMethodInstance(
        "Euclidean", 
        EuclideanPathReconstructionMethod(), 
        lambda prob_map, img: prob_map
    )
]


# %%
def show_path(map: torch.Tensor, 
              ref_path: list[tuple[int, int]],
              pred_path: list[tuple[int, int]], 
              margins=20
):
    path_x, path_y = np.array(ref_path)[:, 1], np.array(ref_path)[:, 0]
    min_x, max_x = path_x.min(), path_x.max()
    min_y, max_y = path_y.min(), path_y.max()

    if map.ndim == 2:
        map = ((map - map.min()) / (map.max() - map.min()) * 255).numpy().astype(np.uint8)
        img = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8)
        img[:, :, 0] = map
        img[:, :, 1] = map
        img[:, :, 2] = map
    else:
        map_min = map.amin(dim=(1, 2), keepdim=True)
        map_max = map.amax(dim=(1, 2), keepdim=True)
        map = (map - map_min) / (map_max - map_min + 1e-8)
        map = (map * 255).clamp(0, 255).byte().cpu().numpy()
        map = map.transpose(1, 2, 0)  # C,H,W -> H,W,C
        img = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8)
        if map.shape[2] == 3:  # H, W, 3
            img = map
        elif map.shape[2] == 2:  # H, W, 2
            img[:, :, 0] = map[:, :, 0]
            img[:, :, 1] = map[:, :, 1]

    ref_img, pred_img = img.copy(), img.copy()
    for (y, x) in ref_path:
        img[y, x] = [0, 255, 0]  # Green for reference path
        ref_img[y, x] = [0, 255, 0]
    for (y, x) in pred_path:
        img[y, x] = np.clip(img[y, x].astype(np.uint16) + np.array([255, 0, 0], dtype=np.uint16), 0, 255).astype(np.uint8)  # Red for predicted path
        pred_img[y, x] = [255, 0, 0]

    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].imshow(img[max(0, min_y - margins):min(img.shape[0], max_y + margins), max(0, min_x - margins):min(img.shape[1], max_x + margins)]) 
    axs[1].imshow(ref_img[max(0, min_y - margins):min(img.shape[0], max_y + margins), max(0, min_x - margins):min(img.shape[1], max_x + margins)])
    axs[2].imshow(pred_img[max(0, min_y - margins):min(img.shape[0], max_y + margins), max(0, min_x - margins):min(img.shape[1], max_x + margins)])
    plt.show()


# %%
from PIL import Image
from tqdm.auto import tqdm

def process_case(n, reconstruction_methods, display=False):
    metrics_per_method = {method_instance.name: [] for method_instance in reconstruction_methods}
    ref_lengths = {"euclidean": [], "n_pixels": [], "geodesic": []}
    
    if n is None or n < 0:
        n = len(gt_path_list)
    for i in tqdm(range(n), position=0):
        # ========= Load data ========
        filename = gt_path_list[i]
        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, filename)
        img_path = os.path.join(img_folder, filename)
        prob_map_path = os.path.join(prob_map_folder, filename.replace(".png", ".pt"))

        img = torch.tensor(np.array(Image.open(img_path).convert("RGB")), dtype=torch.float32).permute(2, 0, 1)
        prob_map = torch.load(prob_map_path, map_location="cpu", weights_only=False)[1]
        gt_np = np.array(Image.open(gt_path).convert("L")) > 0
        segmentation_mask_np = np.array(Image.open(pred_path).convert("L")) > 0

        # ========= Create graph and apply transforms ========
        combined_graph: nx.Graph = get_combined_graph(gt_np, segmentation_mask_np)
        graph_wrapper = GraphWrapper(combined_graph)
        new_nx_graph: nx.Graph = apply_graph_transforms(graph_wrapper, train_transforms).get_graph()

        # ========== Extract true path edges and their centerlines ========
        ref_edges = []
        ref_paths = []
        for u, v, d in new_nx_graph.edges(data=True):
            edge_pred_state = d.get("edge_pred_state", None)
            u_pos = np.array(new_nx_graph.nodes[u]["pos"])
            v_pos = np.array(new_nx_graph.nodes[v]["pos"])
            if edge_pred_state in [EdgePredState.NOT_IN_PREDICTION, EdgePredState.NOT_IN_PREDICTION.value]:
                if new_nx_graph.degree(u) > 1 and new_nx_graph.degree(v) > 1:
                    euc_dist = np.linalg.norm(u_pos - v_pos)
                    if euc_dist > 0:
                        ref_edges.append([u, v])
                        path = list(d["centerline"])
                        ref_paths.append(path)
        ref_edges = torch.tensor(ref_edges, dtype=torch.long).T

        # ========== Compute reconstructions and metrics ============
        for i, method_instance in enumerate(tqdm(reconstruction_methods, leave=False, position=1)):
            name = method_instance.name
            method = method_instance.method
            method_map = method_instance.map_fn(prob_map, img)

            pred_paths = method.reconstruct(map=method_map, graph=new_nx_graph, new_edges=ref_edges)
            for path_i, (ref_path, pred_path) in enumerate(zip(ref_paths, pred_paths)):
                res, distances = compute_metrics(ref_path, pred_path)
                if i == 0:
                    if display:
                        print("Distances:", distances)
                        print("Metrics:", res)
                    for distance, value in distances.items():
                        ref_lengths[distance].append(value)
                metrics = list(res.values())
                if display:
                    show_path(method_map, ref_path, pred_path, margins=50)
                metrics_per_method[name].append(metrics)

    metrics_per_method = {method_name: np.array(metrics_list) for method_name, metrics_list in metrics_per_method.items()}
    ref_lengths = {distance: np.array(distances_list) for distance, distances_list in ref_lengths.items()}
    return metrics_per_method, ref_lengths


# %%
n = -1

# %%
metrics_per_method, ref_lengths = process_case(n, reconstruction_methods, display=True)

# %%
import json

metrics_per_method_list = {k: val.tolist() for k, val in metrics_per_method.items()}
ref_lengths_list = {k: val.tolist() for k, val in ref_lengths.items()}
print(metrics_per_method_list)
print(ref_lengths_list)
full_data = {
    "metrics_per_method": metrics_per_method_list,
    "ref_lengths": ref_lengths_list
}

dir = "/home/morand/afs/EVAPORE/notebooks/"
filepath = os.path.join(dir, f"benchmark_reconstruction_algorithm_{n}.json")
with open(filepath, 'w') as f:
    json.dump(full_data, f, indent=4)

# %%
import json

dir = "/home/morand/afs/EVAPORE/notebooks/"
filepath = os.path.join(dir, f"benchmark_reconstruction_algorithm_{n}.json")
data = None
with open(filepath, 'r') as f:
    data = json.load(f)
metrics_per_method_list = data["metrics_per_method"]
ref_lengths_list = data["ref_lengths"]

metrics_per_method = {k: np.array(val) for k, val in metrics_per_method_list.items()}
ref_lengths = {k: np.array(val) for k, val in ref_lengths_list.items()}


# %%
def plot_metric_by_method(metrics_per_method, used_metrics):
    for method_name, metrics_array in metrics_per_method.items():
        plt.violinplot(metrics_array, showmeans=True, showextrema=True)
        plt.xticks(np.arange(1, len(used_metrics) + 1), used_metrics, rotation=45)
        plt.title(f"Metrics for {method_name}")
        plt.show()


# %%
def plot_metric_by_metric(metrics_per_method, used_metrics, y_percentile=None):
    for i, metric in enumerate(used_metrics):
        metric_values = [metrics_per_method[method_name][:, i] for method_name in metrics_per_method.keys()]
        upper = np.percentile(np.concatenate(metric_values), y_percentile) if y_percentile is not None else None
        plt.violinplot(metric_values, showmeans=True, showextrema=True)
        if upper is not None:
            plt.ylim(0, upper)
        plt.xticks(np.arange(1, len(metrics_per_method) + 1), list(metrics_per_method.keys()), rotation=90)
        plt.title(f"Comparison of {metric} across methods")
        plt.show()


# %%
def plot_metric_by_metric_hparam(metrics_per_method, used_metrics, hparam_name="f", log=False):
    for i, metric in enumerate(used_metrics):
        method_names = list(metrics_per_method.keys())
        metric_values = [metrics_per_method[method_name][:, i] for method_name in metrics_per_method.keys()]
        method_hparam = [float(name.split(f"{hparam_name}=")[-1]) for name in method_names]
        x = np.array(method_hparam)
        y = np.mean(metric_values, axis=1)
        error = np.std(metric_values, axis=1)
        plt.plot(x, y, marker='o')
        if log:
            plt.xscale('log')
        plt.fill_between(x, y - error, y + error, alpha=0.2)
        plt.title(f"Comparison of {metric} across methods")
        plt.show()


# %%
from scipy.stats import spearmanr

def plot_metric_by_length(metrics_per_method, 
                                   ref_lengths, 
                                   used_metrics,
                                   distance='euclidean',
                                   methods_to_plot=None, 
                                   log_length=False, 
                                   log_score=False,
                                   log_bins_height=False,
                                   plot_for_each_method=False,
                                   alpha=1.0,
                                   clip_at=None,
                                   n_bins=50
):
    ref_lengths = ref_lengths[distance]
    x = np.array(ref_lengths)
    indices = np.argsort(x)
    x_in_x_order = x[indices]
    max_length_val = np.max(x)
    method_names = metrics_per_method.keys()
    n_methods = len(method_names)
    if methods_to_plot is not None:
        n_methods = len(methods_to_plot)
        
    for i, metric in enumerate(used_metrics):
        fig, axs = plt.subplots(2,2, figsize=(15,15))
        metric_values = [metrics_per_method[method_name][:, i] for method_name in method_names]
        for method_name, y in zip(method_names, metric_values):
            if (methods_to_plot is None) or (method_name in methods_to_plot):
                max_score_val = np.max(y)
                if clip_at is not None:
                    y = np.clip(y, a_min=clip_at, a_max=max_score_val)

                r = np.corrcoef(x, y)[0,1]
                print("Correlation:", r)
                r, _ = spearmanr(x, y)
                print("Spearman correlation:", r)

                # Create line values
                a, b = np.polyfit(x, y, 1)  # degree 1 = line
                print("Slope:", a, ", intercept:", b)
                x_line = np.linspace(min(x), max(x), 100)
                y_line = a * x_line + b
                axs[0,0].plot(x_line, y_line)

                # Point cloud
                scatter = axs[0,0].scatter(x, y, label=method_name, alpha=alpha)
                axs[0,0].set_xlabel(f'{distance} length')
                axs[0,0].set_ylabel(f'{metric} score')

                method_color = scatter.get_facecolor()[0]
                method_color[3] = 1.0
                
                # Score histograms
                score_bins = n_bins
                if log_score:
                    score_bins = 10**(np.linspace(-1, np.log10(max_score_val), n_bins))
                axs[0,1].hist(y, orientation='horizontal', alpha=(1/n_methods), cumulative=False, bins=score_bins)
                axs[0,1].set_ylabel(f'{metric} score')

                # Max curves
                y_in_x_order = y[indices]
                running_max = np.maximum.accumulate(y_in_x_order)
                running_p95 = np.array([
                    np.percentile(y_in_x_order[:i+1], 95)
                    for i in range(len(y_in_x_order))
                ])
                running_mean = np.cumsum(y_in_x_order) / np.arange(1, len(y_in_x_order) + 1)

                axs[1,1].plot(x_in_x_order, running_mean, linestyle=':',
                            label=f"{method_name} mean", color=method_color)
                axs[1,1].plot(x_in_x_order, running_p95, linestyle='--',
                            label=f"{method_name} 95% max", color=method_color)
                axs[1,1].plot(x_in_x_order, running_max, label=f"{method_name} max", color=method_color)
                axs[1,1].grid(True, which='both')
                axs[1,1].set_xlabel(f'{distance} length')
                axs[1,1].set_ylabel(f'{metric} score')

        length_bins = n_bins
        if log_length:
            length_bins = 10**(np.linspace(0, np.log10(max_length_val), n_bins))
        axs[1,0].hist(x, orientation='vertical', alpha=0.5, cumulative=False, density=True, bins=length_bins)
        axs[1,0].hist(x, orientation='vertical', alpha=0.5, cumulative=True, density=True, bins=length_bins)
        axs[1,0].grid(True, which='both')
        axs[1,0].set_xlabel(f'{distance} length')
        
        if log_length:
            axs[0,0].set_xscale('log')
            axs[1,0].set_xscale('log')
            axs[1,1].set_xscale('log')
        if log_score:
            axs[0,0].set_yscale('log')
            axs[0,1].set_yscale('log')
            axs[1,1].set_yscale('log')
        if log_bins_height:
            axs[0,1].set_xscale('log')
            axs[1,0].set_yscale('log')

        fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05))
        plt.show()

    if plot_for_each_method:
        for i, metric in enumerate(used_metrics):
            metric_values = [metrics_per_method[method_name][:, i] for method_name in method_names]
            for method_name, y in zip(method_names, metric_values):
                if (methods_to_plot is not None) and (method_name in methods_to_plot):
                    plt.scatter(x, y, label=method_name)
                    if log_length:
                        plt.xscale('log')
                    if log_score:
                        plt.yscale('log')
                    plt.title(f"{metric} values for {method_name} method, by {distance} length of reference path")
                    plt.show()


# %%
plot_metric_by_metric(metrics_per_method, used_metrics)
plot_metric_by_metric(metrics_per_method, used_metrics, y_percentile=95)

# %%
methods_to_plot = [
    "Euclidean", 
    "DahuDistanceOnGreenChannel",
    #"DahuDistanceOnProbMap", 
    "SquaredMinEnergyOnGreenChannel",
]
plot_metric_by_length(metrics_per_method, 
                               ref_lengths, 
                               used_metrics,
                               distance='euclidean',
                               methods_to_plot=methods_to_plot, 
                               log_length=True, 
                               log_score=True, 
                               log_bins_height=False,
                               plot_for_each_method=False, 
                               alpha=0.1, clip_at=0.1, n_bins=50)

# %%
x = np.array(ref_lengths["euclidean"])

lt_or_eq_100 = (x <= 100)
lt_or_eq_100_x = x[lt_or_eq_100]
print(len(lt_or_eq_100_x), len(x), "percentage:", len(lt_or_eq_100_x) / len(x) * 100)

# %%
from scipy.stats import spearmanr
from matplotlib.colors import LogNorm

def plot_metric_by_length(metrics_per_method, 
                        ref_lengths, 
                        used_metrics,
                        distance='euclidean',
                        method_to_plot=None, 
                        log_length=False, 
                        log_score=False,
                        log_bins_height=False,
                        alpha=1.0,
                        metric_clip = [0.1, None],
                        length_clip = [1.0, None],
                        n_bins=50,
                        density_plot = True         
):
    ref_lengths = ref_lengths[distance]
    x = np.array(ref_lengths)
    indices = np.argsort(x)
    x_in_x_order = x[indices]
    min_length_val = np.min(x)
    max_length_val = np.max(x)
    if length_clip[0] is not None:
            min_length_val = length_clip[0]
    if length_clip[1] is not None:
            max_length_val = length_clip[1]
    print(min_length_val, max_length_val)
        
    for i, metric in enumerate(used_metrics):
        fig, axs = plt.subplots(2,2, figsize=(15,15))
        y = metrics_per_method[method_to_plot][:, i]

        min_score_val = np.min(y)
        max_score_val = np.max(y)
        if metric_clip[0] is not None and metric_clip[0] > min_score_val:
            min_score_val = metric_clip[0]
        if metric_clip[1] is not None and metric_clip[1] < max_score_val:
            max_score_val = metric_clip[1]
        print(min_score_val, max_score_val)

        r = np.corrcoef(x, y)[0,1]
        print("Correlation:", r)
        r, _ = spearmanr(x, y)
        print("Spearman correlation:", r)

        # Create line values
        a, b = np.polyfit(x, y, 1)  # degree 1 = line
        print("Slope:", a, ", intercept:", b)
        x_line = np.linspace(min(x), max(x), 100)
        y_line = a * x_line + b
        axs[0,0].plot(x_line, y_line)

        if log_score:
            score_bins = 10**(np.linspace(np.log10(min_score_val), np.log10(max_score_val), n_bins))
        else:
            score_bins = n_bins
        if log_length:
            length_bins = 10**(np.linspace(np.log10(min_length_val), np.log10(max_length_val), n_bins))
        else:
            length_bins = n_bins

        # Point cloud
        if density_plot:
            # Point cloud
            bins = [length_bins, score_bins]
            hist = axs[0,0].hist2d(x, y, bins=bins, norm=LogNorm())
            fig.colorbar(hist[3])
        else:
            scatter = axs[0,0].scatter(x, y, label=method_to_plot, alpha=alpha)
        axs[0,0].set_xlabel(f'{distance} length')
        axs[0,0].set_ylabel(f'{metric} score')
        
        # Score histograms
        score_bins = n_bins
        if log_score:
            score_bins = 10**(np.linspace(-1, np.log10(max_score_val), n_bins))
        hist = axs[0,1].hist(y, orientation='horizontal', alpha=1.0, cumulative=False, bins=score_bins)
        axs[0,1].set_ylabel(f'{metric} score')

        method_color = list(hist[2][0].get_facecolor())
        method_color[3] = 1.0

        # Max curves
        y_in_x_order = y[indices]
        running_max = np.maximum.accumulate(y_in_x_order)
        running_p95 = np.array([
            np.percentile(y_in_x_order[:i+1], 95)
            for i in range(len(y_in_x_order))
        ])
        running_mean = np.cumsum(y_in_x_order) / np.arange(1, len(y_in_x_order) + 1)

        axs[1,1].plot(x_in_x_order, running_mean, linestyle=':',
                    label=f"{method_to_plot} mean", color=method_color)
        axs[1,1].plot(x_in_x_order, running_p95, linestyle='--',
                    label=f"{method_to_plot} 95% max", color=method_color)
        axs[1,1].plot(x_in_x_order, running_max, label=f"{method_to_plot} max", color=method_color)
        axs[1,1].grid(True, which='both')
        axs[1,1].set_xlabel(f'{distance} length')
        axs[1,1].set_ylabel(f'{metric} score')

        axs[1,0].hist(x, orientation='vertical', alpha=0.5, cumulative=False, density=True, bins=length_bins)
        axs[1,0].hist(x, orientation='vertical', alpha=0.5, cumulative=True, density=True, bins=length_bins)
        axs[1,0].grid(True, which='both')
        axs[1,0].set_xlabel(f'{distance} length')
        
        if log_length:
            axs[0,0].set_xscale('log')
            axs[1,0].set_xscale('log')
            axs[1,1].set_xscale('log')
        if log_score:
            axs[0,0].set_yscale('log')
            axs[0,1].set_yscale('log')
            axs[1,1].set_yscale('log')
        if log_bins_height:
            axs[0,1].set_xscale('log')
            axs[1,0].set_yscale('log')

        fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.01))
        plt.show()


# %%
plot_metric_by_length(metrics_per_method, 
                               ref_lengths, 
                               used_metrics,
                               distance='euclidean',
                               method_to_plot='Euclidean', 
                               log_length=True, 
                               log_score=True, 
                               log_bins_height=False,
                               alpha=0.1,
                               n_bins=30,
                               density_plot = True)

# %%
from matplotlib.colors import LogNorm

metric_clip = [0.1, 10]
ratio_clip = [1.0, 2.0]
alpha = 1.0
n_bins = 30
log_score = False
log_ratios = False
log_bins_height = True
density_plot = True

methods_to_plot = [
    "Euclidean", 
]

print(ref_lengths)
euclidean_lengths = ref_lengths['euclidean']
n_pixels_lengths = ref_lengths['n_pixels']
geodesic_lengths = ref_lengths['geodesic']
assert len(euclidean_lengths) == len(n_pixels_lengths) == len(geodesic_lengths)

ratios = {
    "geodesic_n_pixels_ratio": geodesic_lengths / n_pixels_lengths,
    "geodesic_euclidean_ratio": geodesic_lengths / euclidean_lengths,
    "n_pixels_euclidean_ratio": n_pixels_lengths / euclidean_lengths
}

ratio_name = "geodesic_euclidean_ratio"
ratio = ratios[ratio_name]

print(ratio)
assert np.all(ratio >= 1)

method_names = metrics_per_method.keys()
n_methods = len(method_names)

x = ratio
indices = np.argsort(x)
x_in_x_order = x[indices]
x = np.clip(x, a_min=ratio_clip[0], a_max=ratio_clip[1])
max_ratio_val = np.max(x)

for i, metric in enumerate(used_metrics):
    metric_values = [metrics_per_method[method_name][:, i] for method_name in method_names]
    for method_name, y in zip(method_names, metric_values):
        if (methods_to_plot is None) or (method_name in methods_to_plot):
            max_score_val = np.max(y)
            if metric_clip[1] is not None:
                max_score_val = metric_clip[1]
            y = np.clip(y, a_min=metric_clip[0], a_max=max_score_val)

            # Point cloud
            if log_score:
                score_bins = 10**(np.linspace(np.log10(metric_clip[0]), np.log10(max_score_val), n_bins))
            else:
                score_bins = n_bins
            if log_ratios:
                ratio_bins = 10**(np.linspace(np.log10(ratio_clip[0]), np.log10(max_ratio_val), n_bins))
            else:
                ratio_bins = n_bins
            bins = [ratio_bins, score_bins]
            plt.hist2d(x, y, bins=bins, norm=LogNorm())
            plt.colorbar()

    length_bins = n_bins
    if log_ratios:
        length_bins = 10**(np.linspace(0, np.log10(max_ratio_val), n_bins))
    
    if log_ratios:
        plt.xscale('log')
    if log_score:
        plt.yscale('log')

    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05))
    plt.show()

# %%
from matplotlib.colors import LogNorm

metric_clip = [0.1, 10]
ratio_clip = [1.0, 2.0]
alpha = 1.0
n_bins = 30
log_score = False
log_ratios = False
log_bins_height = True
density_plot = True

methods_to_plot = [
    "Euclidean", 
]

print(ref_lengths)
euclidean_lengths = ref_lengths['euclidean']
n_pixels_lengths = ref_lengths['n_pixels']
geodesic_lengths = ref_lengths['geodesic']

assert len(euclidean_lengths) == len(n_pixels_lengths) == len(geodesic_lengths)

ratios = {
    "geodesic_n_pixels_ratio": geodesic_lengths / n_pixels_lengths,
    "geodesic_euclidean_ratio": geodesic_lengths / euclidean_lengths,
    "n_pixels_euclidean_ratio": n_pixels_lengths / euclidean_lengths
}
ratio_name = "geodesic_euclidean_ratio"
y = ratios[ratio_name]
assert np.all(y >= 1)

x = euclidean_lengths
r = np.corrcoef(x, y)[0,1]
print("Correlation:", r)

from scipy.stats import spearmanr
r, _ = spearmanr(x, y)
print("Spearman correlation:", r)

a, b = np.polyfit(x, y, 1)  # degree 1 = line
print("Slope:", a, ", intercept:", b)
# Create line values
x_line = np.linspace(min(x), max(x), 100)
y_line = a * x_line + b
plt.plot(x_line, y_line)

plt.scatter(x, y)
plt.xlim(left=6)
plt.ylim((1,2))
plt.xscale('log')
plt.yscale('log')
plt.show()

for i in range(len(euclidean_lengths)):
    e_l = euclidean_lengths[i]
    n_p_l = n_pixels_lengths[i]
    geo_l = geodesic_lengths[i]
    r = y[i]
    if r > 1.5:
        print(e_l, n_p_l, geo_l, r)

# %%
reconstruction_methods: list[PathReconstructionMethodInstance] = [
    PathReconstructionMethodInstance(
        f"SquaredMinEnergyOnProbMap_f={f}", 
        SquaredMinEnergyPathReconstructionMethod(up_height_factor=f, down_height_factor=f), 
        lambda prob_map, img: prob_map
    ) for f in [0.01, 0.1, 1, 10, 100, 1000]
]

# %%
metrics_per_method, ref_lengths = process_case(1, reconstruction_methods)

# %%
plot_metric_by_metric_hparam(metrics_per_method, used_metrics, hparam_name="f", log=True)

# %%
plot_metric_by_geodesic_length(metrics_per_method, 
                               ref_lengths, 
                               used_metrics, 
                               methods_to_plot=None, 
                               log_length=True, 
                               log_score=True, 
                               log_bins_height=False,
                               plot_for_each_method=False, 
                               alpha=0.2, clip_at=0.1, n_bins=50)

# %%
reconstruction_methods: list[PathReconstructionMethodInstance] = [
    PathReconstructionMethodInstance(
        f"SquaredMinEnergyOnGreenChannel_f={f}", 
        SquaredMinEnergyPathReconstructionMethod(up_height_factor=f, down_height_factor=f), 
        lambda prob_map, img: img[1]
    ) for f in [0.01, 0.1, 1, 10, 100, 1000]
]

# %%
metrics_per_method = process_case(1, reconstruction_methods)

# %%
plot_metric_by_metric_hparam(metrics_per_method, used_metrics, hparam_name="f", log=True)

# %%
reconstruction_methods: list[PathReconstructionMethodInstance] = [
    PathReconstructionMethodInstance(
        f"BluredSquaredMinEnergyOnProb_s={s}", 
        SquaredMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img, sigma=s: torch.tensor(gaussian(prob_map.clone().detach(), sigma=sigma))
    ) for s in [0.0, 0.01, 0.1, 0.5, 1, 2, 5]
]

metrics_per_method = process_case(5, reconstruction_methods)

# %%
plot_metric_by_metric_hparam(metrics_per_method, used_metrics, hparam_name="s", log=True)

# %%
reconstruction_methods: list[PathReconstructionMethodInstance] = [
    PathReconstructionMethodInstance(
        f"BluredSquaredMinEnergyOnGreenChannel_s={s}", 
        SquaredMinEnergyPathReconstructionMethod(), 
        lambda prob_map, img, sigma=s: torch.tensor(gaussian(img[1].clone().detach(), sigma=sigma))
    ) for s in [0.0, 0.01, 0.1, 0.5, 1, 2, 5]
]

metrics_per_method = process_case(30, reconstruction_methods)

# %%
plot_metric_by_metric_hparam(metrics_per_method, used_metrics, hparam_name="s", log=True)

# %%
