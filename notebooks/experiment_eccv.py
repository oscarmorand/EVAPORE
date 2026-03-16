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
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

# %%
print("PyTorch version:", torch.__version__)
print("is cuda available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
dataset = "FIVES_clean"
data_dir = f"/home/morand/afs/EVAPORE/data/{dataset}"

# %%
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision

from path_neural_networks.models.path_encoders import PathEncoder
from path_neural_networks.models.path_samplers import PathSampler
from path_neural_networks.models.path_classifiers import PathClassifier
from path_neural_networks.models.features_generators import FeaturesGenerator
from path_neural_networks.models.losses import PathClassificationLoss
from path_neural_networks.models.metrics.binary_prauc import BinaryPRAUC
from path_neural_networks.utils.symmetry_enforcement import SymmetryEnforcementMode

METRIC_REGISTRY = {
    "accuracy": BinaryAccuracy,
    "auroc": BinaryAUROC,
    "recall": BinaryRecall,
    "precision": BinaryPrecision,
    "pr_auc": BinaryPRAUC
}
METRIC_THRESHOLDS_REGISTRY = {
    "accuracy": True,
    "auroc": False,
    "recall": True,
    "precision": True,
    "pr_auc": False
}

class ReducedPipelineModel(pl.LightningModule):
    def __init__(
        self,
        features_generator: FeaturesGenerator,
        path_sampler: PathSampler,
        path_encoder: PathEncoder,
        path_classifier: PathClassifier,
        edge_classification_loss_fn: PathClassificationLoss,
        cfg: dict,
        archi: str,
        lr: float = 1e-3,
        symmetry_enforcement_mode: SymmetryEnforcementMode = SymmetryEnforcementMode.NONE,
        inference_threshold: float = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["features_generator", "path_sampler", "path_encoder", "path_classifier", "edge_classification_loss_fn"])
        self.features_generator = features_generator
        self.path_sampler = path_sampler
        self.path_encoder = path_encoder
        self.path_classifier = path_classifier

        self.edge_classification_loss_fn = edge_classification_loss_fn

        # Metrics
        self.base_metrics = cfg.get("metrics", [])
        self.train_metrics = nn.ModuleDict({metric_name: METRIC_REGISTRY[metric_name]() for metric_name in self.base_metrics})
        self.val_metrics = nn.ModuleDict({metric_name: METRIC_REGISTRY[metric_name]() for metric_name in self.base_metrics})
        if inference_threshold is None:
            inference_threshold = 0.5
        self.set_inference_threshold(inference_threshold)
        

    def set_inference_threshold(self, inference_threshold: float):
        self.test_metrics = nn.ModuleDict(
            {metric_name: (METRIC_REGISTRY[metric_name](threshold=inference_threshold) if METRIC_THRESHOLDS_REGISTRY[metric_name] else METRIC_REGISTRY[metric_name]()) for metric_name in self.base_metrics}
        )

    def forward(self, 
                img: torch.Tensor,
                paths: list[torch.Tensor]
    ) -> torch.Tensor:
    
        # Compute the feature maps
        feature_maps = self.features_generator(img) # shape (1, channels, H, W)

        # Sample features along paths
        paths_logits = []
        for i, path in enumerate(paths):
            # Sample features along path
            path_features = self.path_sampler(feature_maps, path)  # shape (1, channels, path_length)
            # Encode path
            encoded_path = self.path_encoder(path_features) # shape (1, out_channels)
            # Classify path
            path_logits = self.path_classifier(encoded_path)

            # Strict Symmetry enforcement (if needed)
            if self.hparams.symmetry_enforcement_mode == SymmetryEnforcementMode.DOUBLE_PASS:
                inv_path_features = torch.flip(path_features, dims=[2])  # shape (1, channels, path_length)
                inv_encoded_path = self.path_encoder(inv_path_features) # shape (1, out_channels)
                inv_path_logits = self.path_classifier(inv_encoded_path)
                path_logits = (path_logits + inv_path_logits) / 2.0

            paths_logits.append(path_logits)
            
        paths_logits = torch.cat(paths_logits, dim=1).squeeze()  # shape (num_edges,)

        return paths_logits, feature_maps

    def _shared_step(self, 
                     batch: tuple,
                     step: str,
                     metrics: nn.ModuleDict
    ) -> torch.Tensor:
        img, (paths, edges_classes), _, _ = batch
        paths_logits, _ = self.forward(img, paths)
        paths_probs = torch.sigmoid(paths_logits)
        edges_classes = edges_classes.squeeze().type(torch.float)

        loss = self.edge_classification_loss_fn(paths_logits, edges_classes)

        self.log(f"{step}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for name, metric in (metrics.items()):
            metric(paths_probs, edges_classes.int())
            self.log(f"{step}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def _augment_batch_with_flipped_paths(self,
                                          batch: tuple
    ) -> tuple:
        img, (paths, edges_classes), _, _ = batch
        flipped_paths = []
        for path in paths:
            flipped_path = torch.flip(path, dims=[0])
            flipped_paths.append(flipped_path)
        augmented_paths = paths + flipped_paths
        augmented_edges_classes = torch.cat([edges_classes, edges_classes], dim=1)
        return img, (augmented_paths, augmented_edges_classes), None, None

    def training_step(self, batch, batch_idx):
        if self.hparams.symmetry_enforcement_mode == SymmetryEnforcementMode.DATA_AUGMENTATION:
            batch = self._augment_batch_with_flipped_paths(batch)
        loss = self._shared_step(batch, step="train", metrics=self.train_metrics)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, step="val", metrics=self.val_metrics)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, step="test", metrics=self.test_metrics)
        return loss

    def predict_step(self, batch, batch_idx):
        img, _, (edges, radius, paths, _), _ = batch
        paths_logits, _ = self.forward(img, paths)
        paths_probs = torch.sigmoid(paths_logits)
        return edges, radius, paths_probs

    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr'] 
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.hparams.get("lr", 3e-4),
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

# %%
from path_neural_networks.models.path_samplers.multi_scale_square_path_sampler import MultiScaleSquarePathSampling
from path_neural_networks.models.path_samplers.square_path_sampler import SquarePathSampler
from path_neural_networks.models.path_samplers.sampling_aggregation_method import *
from path_neural_networks.models.path_encoders import ConvMaxPoolingPathEncoder, ConvMultiStatsPoolingPathEncoder

centerline_dir = "euclidean_lt_100_clean_centerlines"

use_data_augmentation = True

lr = 3e-4

use_features_generator = True
features_generator_to_sampler = 32
features_generator_skip_connections = False
pretrained_features_generator = True
features_generator_ckpt = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_202/checkpoints/best-checkpoint-epoch=65-val_loss=0.0685.ckpt"
freeze_pretrained_features_generator = False

path_sampler_cls = MultiScaleSquarePathSampling
sampling_square_size = 3
sampling_square_sizes = [1, 3, 5]
sampling_aggregation_method = SamplingMaxAggregation()

path_encoder_cls = ConvMaxPoolingPathEncoder
conv_path_residual_blocks = False
conv_path_skip_connections = False
conv_path_layers = [None, None, 256]

path_classifier_n_hidden_layers = 2
path_classifier_dropout = 0

use_pos_weight_in_loss = True

symmetry_enforcement_mode = SymmetryEnforcementMode.NONE

# %%
from path_neural_networks.data.image_centerline_dataset import ImageCenterlineDataset
from path_neural_networks.data.image_centerline_datamodule import ImageCenterlineDatamodule
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.data_augmentation.add_gaussian_noise import AddGaussNoise

data_seed = 42
split_seed = 42

image_centerline_dataset = ImageCenterlineDataset(data_dir=data_dir, centerline_dir=centerline_dir)
mean = image_centerline_dataset.stats["mean"]
std = image_centerline_dataset.stats["std"]
classes_count = image_centerline_dataset.stats["classes_count"]
classes_ratio = image_centerline_dataset.stats["classes_ratio"]

if use_data_augmentation:
        train_transforms = A.Compose([
            A.RandomBrightnessContrast(
            brightness_limit=(-0.15, 0.15),
            contrast_limit=(-0.15, 0.15),
            p=0.5
        ),
        A.Lambda(image=AddGaussNoise(std=(0.005, 0.015)), p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        ToTensorV2()
    ], seed=data_seed)
else:
    train_transforms = A.Compose([
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        ToTensorV2()
    ], seed=data_seed)

val_transforms = A.Compose([
    A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
    ToTensorV2()
], seed=data_seed)


image_centerline_datamodule = ImageCenterlineDatamodule(dataset=image_centerline_dataset, 
                                                         val_split=0.2,
                                                         test_split=0.1,
                                                         train_transforms=train_transforms,
                                                         val_transforms=val_transforms, 
                                                         test_transforms=val_transforms,
                                                         num_workers=0,
                                                         seed=split_seed)

def serialize_data_augmentation(transforms):
    if transforms is None:
        return "None"
    if isinstance(transforms, A.Compose):
        list_of_transforms = []
        for t in transforms.transforms:
            list_of_transforms.append(serialize_data_augmentation(t))
        return {"A.Compose": list_of_transforms}
    if isinstance(transforms, A.RandomBrightnessContrast):
        return {
            "A.RandomBrightnessContrast": {
                "brightness_limit": list(transforms.brightness_limit),
                "contrast_limit": list(transforms.contrast_limit),
                "p": transforms.p
            }
        }
    if isinstance(transforms, A.Lambda) and transforms.custom_apply_fns["image"] == AddGaussNoise:
        transform = transforms.custom_apply_fns["image"]
        return {
            "AddGaussNoise": {
                "std": transform.std,
                "p": transforms.p
            }
        }
    if isinstance(transforms, A.Normalize):
        return {
            "A.Normalize": {
                "mean": list(transforms.mean),
                "std": list(transforms.std)
            }
        }
    if isinstance(transforms, ToTensorV2):
        return "ToTensorV2"
    return str(transforms)


# %%
from path_neural_networks.models.path_encoders import ConvMaxPoolingPathEncoder
from path_neural_networks.models.path_encoders.pooling_encoder import MaxPoolingPathEncoder, NonLearnedPoolingPathEncoder
from path_neural_networks.models.path_classifiers import FCNPathClassifier, PathClassifier
from path_neural_networks.models.features_generators import PretrainedUnetFeaturesGenerator, NoFeaturesGenerator, FeaturesGenerator
from path_neural_networks.models.losses import WeightedBCEWithLogitsLoss, BCEWithLogitsLoss, PathClassificationLoss

features_generator: FeaturesGenerator = None
if use_features_generator and pretrained_features_generator:
    features_generator = PretrainedUnetFeaturesGenerator(features_generator_ckpt, device=device, out_channels=features_generator_to_sampler, freeze_pretrained=freeze_pretrained_features_generator, skip_connection=features_generator_skip_connections)
else:
    features_generator = NoFeaturesGenerator()
    
path_sampler: PathSampler = path_sampler_cls(in_channels=features_generator.out_channels,
                                square_size=sampling_square_size,
                                aggregation=sampling_aggregation_method,
                                square_sizes=sampling_square_sizes
)

path_encoder: PathEncoder = path_encoder_cls(in_channels=path_sampler.out_channels, 
                                             hidden_layers=conv_path_layers, 
                                             skip_connection=conv_path_skip_connections, 
                                             residual_blocks=conv_path_residual_blocks)

path_classifier: PathClassifier = FCNPathClassifier(in_channels=path_encoder.out_channels, n_hidden_layers=path_classifier_n_hidden_layers, num_classes=1, dropout=path_classifier_dropout)

loss_fn : PathClassificationLoss = None
if use_pos_weight_in_loss:
    loss_fn = WeightedBCEWithLogitsLoss(classes_ratio)
else:
    loss_fn = BCEWithLogitsLoss()

cfg = {
    "centerline_dir": centerline_dir,
    "features_generator": features_generator.as_dict(),
    "path_sampler": path_sampler.as_dict(),
    "path_encoder": path_encoder.as_dict(),
    "path_classifier": path_classifier.as_dict(),
    "metrics": ["accuracy", "auroc", "recall", "precision", "pr_auc"],
    "train_data_augmentation": serialize_data_augmentation(train_transforms),
    "test_data_augmentation": serialize_data_augmentation(val_transforms),
    "classification_loss_fn": loss_fn.as_dict()
}
print(cfg)
archi = str(features_generator) + str(path_encoder) + str(path_classifier)
print(archi)

# %%
reduced_pipeline_model = ReducedPipelineModel(
    features_generator=features_generator,
    path_sampler=path_sampler,
    path_encoder=path_encoder,
    path_classifier=path_classifier,
    edge_classification_loss_fn=loss_fn,
    cfg=cfg,
    archi=archi,
    lr=lr,
    symmetry_enforcement_mode=symmetry_enforcement_mode
)

# %%
from pytorch_lightning import Trainer

ckpt_path = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_247_best_model_i_think/checkpoints/best-pr-auc-checkpoint-epoch=03-val_pr_auc=0.9134.ckpt"
best_threshold = 0.6870

reduced_pipeline_model = ReducedPipelineModel.load_from_checkpoint(
    ckpt_path,
    features_generator=features_generator,
    path_sampler=path_sampler,
    path_encoder=path_encoder,
    path_classifier=path_classifier,
    edge_classification_loss_fn=loss_fn,
)

reduced_pipeline_model.set_inference_threshold(best_threshold)

torch.set_float32_matmul_precision("medium")
trainer = Trainer(accelerator='gpu', devices="auto", max_epochs=10, precision='16-mixed')

# %%
trainer.test(reduced_pipeline_model, datamodule=image_centerline_datamodule)

# %%
dataloader = image_centerline_datamodule.test_dataloader()

# %%
print(mean)
print(std)

# %%
import numpy as np
from graph_neural_networks.data.utils.pred_state import EdgePredState
import networkx as nx
import torch
from skimage.measure import label
from skimage.morphology import dilation, disk

def happend_edge_index_undirected(index_list: list[list[int]],
                                u: int,
                                v: int) -> None:
    index_list.append([u, v])
    index_list.append([v, u])  # undirected graph

def get_edges_index(nx_graph: nx.Graph):
    message_passing_edges_index = []
    gt_edges_index = []
    virtual_edges_index = []
    in_pred_edges_index = []
    not_in_pred_edges_index = []
    visited_edges = set()
    for u, v in nx_graph.edges(data=False):
        if (u, v) in visited_edges:
                continue
        visited_edges.add((u, v))
        for e, data in nx_graph.get_edge_data(u, v).items():
            pred_state = data.get('edge_pred_state', None)
            virtual_edge = data.get('virtual_edge', False)
            if virtual_edge:
                happend_edge_index_undirected(message_passing_edges_index, u, v)
                happend_edge_index_undirected(virtual_edges_index, u, v)
            else:
                if pred_state is None or pred_state in [EdgePredState.IN_PREDICTION, EdgePredState.IN_PREDICTION.value]:
                    happend_edge_index_undirected(message_passing_edges_index, u, v)
                    happend_edge_index_undirected(in_pred_edges_index, u, v)
                elif pred_state in [EdgePredState.NOT_IN_PREDICTION, EdgePredState.NOT_IN_PREDICTION.value]:
                    happend_edge_index_undirected(not_in_pred_edges_index, u, v)
                happend_edge_index_undirected(gt_edges_index, u, v)

    message_passing_edges_index_tensor = torch.tensor(message_passing_edges_index, dtype=torch.long).t().contiguous()
    gt_edges_index_tensor = torch.tensor(gt_edges_index, dtype=torch.long).t().contiguous()
    virtual_edges_index_tensor = torch.tensor(virtual_edges_index, dtype=torch.long).t().contiguous()
    in_pred_edges_index_tensor = torch.tensor(in_pred_edges_index, dtype=torch.long).t().contiguous()
    not_in_pred_edges_index_tensor = torch.tensor(not_in_pred_edges_index, dtype=torch.long).t().contiguous()

    return message_passing_edges_index_tensor, gt_edges_index_tensor, virtual_edges_index_tensor, in_pred_edges_index_tensor, not_in_pred_edges_index_tensor

def cut_mask_from_negative_edges(centerline, mask):
    centerline_mask = np.zeros_like(mask, dtype=bool)
    for (x, y) in centerline:
        centerline_mask[x, y] = True

    combined_mask = np.logical_and(centerline_mask, np.logical_not(mask))
    labeled_mask, num_labels = label(combined_mask, return_num=True, connectivity=2)
    centerlines = [[] for _ in range(num_labels)]
    for (x, y) in centerline:
        label_id = labeled_mask[x, y]
        if label_id > 0:
            centerlines[label_id - 1].append([int(x), int(y)])
    
    return centerlines

def cut_mask_from_negative_edges_for_all(centerlines, mask, classes=None, edges=None, return_old_centerlines=False):
    new_centerlines = []
    new_classes = []
    new_edges = []
    old_centerlines = []
    for i, centerline in enumerate(centerlines):
        new_centerline = cut_mask_from_negative_edges(centerline, mask)
        n_new_centerline = len(new_centerline)
        if n_new_centerline > 0:
            new_centerlines.extend(new_centerline)
            if return_old_centerlines:
                old_centerlines.extend([centerline for _ in range(n_new_centerline)])
            if classes is not None:
                cls = classes[i]
                new_classes.extend([cls for _ in range(len(new_centerline))])
            if edges is not None:
                edge = edges[i]
                new_edges.extend([edge for _ in range(len(new_centerline))])

    res = {"new_centerlines": new_centerlines}
    if return_old_centerlines:
        res["old_centerlines"] = old_centerlines
    if classes is not None:
        res["classes"] = new_classes
    if edges is not None:
        res["edges"] = new_edges
    return res

def get_query_edges(graph: nx.Graph, n_closest: int = 1, max_dist: float = None) -> torch.Tensor:
    G = graph.copy()

    connected_components = list(nx.connected_components(graph))

    cc_extrimities = {}
    for cc_i, cc in enumerate(connected_components):
        extrimities = []
        for n in cc:
            if G.degree(n) == 1:
                extrimities.append(n)
        cc_extrimities[cc_i] = extrimities

    cc_n_count = [len(cc) for cc in connected_components]
    main_cc = np.argmax(cc_n_count)
    small_ccs = {i: cc for i, cc in enumerate(connected_components) if i != main_cc}

    virtual_edges_to_add = []
    for i, cc in small_ccs.items():
        extremities = cc_extrimities[i]
        other_ccs = {j: other_cc for j, other_cc in enumerate(connected_components) if j != i}
        other_ccs_all_nodes = []
        for other_cc in other_ccs.values():
            other_ccs_all_nodes.extend(other_cc)

        for extrimity in extremities:
            other_nodes_distances = []
            extrimity_pos = np.array(G.nodes[extrimity]['pos'])
            for other_cc_node in other_ccs_all_nodes:
                other_cc_node_pos = np.array(G.nodes[other_cc_node]['pos'])
                dist = np.linalg.norm(extrimity_pos - other_cc_node_pos)
                other_nodes_distances.append(dist)
            other_nodes_distances = np.array(other_nodes_distances)
            closest_indices = np.argsort(other_nodes_distances)[:n_closest]
            if max_dist is not None:
                closest_indices = closest_indices[other_nodes_distances[closest_indices] <= max_dist]
            closest_nodes = [other_ccs_all_nodes[idx] for idx in closest_indices]

            for extremity_closest_other_cc_node in closest_nodes:
                if (extrimity, extremity_closest_other_cc_node) not in virtual_edges_to_add and (extremity_closest_other_cc_node, extrimity) not in virtual_edges_to_add:
                    virtual_edges_to_add.append([extrimity, extremity_closest_other_cc_node])
    
    virtual_edges_index_tensor = torch.tensor(virtual_edges_to_add, dtype=torch.long).t().contiguous()
    return virtual_edges_index_tensor

def clean_paths_on_surface_of_mask(centerlines, mask, kernel_size=1, threshold=0.5, classes=None):
    new_centerlines = []
    if classes is not None:
        new_classes = []
        
    labeled_mask = label(mask)
    dilated_mask = dilation(labeled_mask, disk(kernel_size))
    
    for i, centerline in enumerate(centerlines):
        centerline_mask = np.zeros_like(mask, dtype=np.uint8)
        for (x, y) in centerline:
            centerline_mask[x, y] = 1

        combined_mask = centerline_mask * dilated_mask
        values_on_mask = np.count_nonzero(combined_mask)
        n_diff_cc = len(np.unique(combined_mask)) - (1 if 0 in combined_mask else 0)
        ratio_on_mask = values_on_mask / len(centerline)

        if ratio_on_mask <= threshold or n_diff_cc > 1:
            new_centerlines.append(centerline)
            if classes is not None:
                new_classes.append(classes[i])

    if classes is not None:
        return new_centerlines, new_classes
    return new_centerlines

def is_reconstructed_path_not_too_far(true_path_existing_centerline, true_path_reconstructed_centerline, distance_ratio_threshold):
    sum_min_distances = 0
    for (xr, yr) in true_path_reconstructed_centerline:
        r = np.array([xr, yr])
        min_dist = float('inf')
        for (xe, ye) in true_path_existing_centerline:
            e = np.array([xe, ye])
            dist = np.linalg.norm(r - e)
            if dist < min_dist:
                min_dist = dist
        sum_min_distances += min_dist
    sum_min_distances /= len(true_path_reconstructed_centerline)
    return sum_min_distances <= distance_ratio_threshold

def remove_too_far_reconstructed_paths_for_all(true_path_existing_centerlines, true_path_reconstructed_centerlines, distance_ratio_threshold):
    new_reconstructed_classes = []
    for i, true_path_reconstructed_centerline in enumerate(true_path_reconstructed_centerlines):
        true_path_existing_centerline = true_path_existing_centerlines[i]
        condition = is_reconstructed_path_not_too_far(true_path_existing_centerline, true_path_reconstructed_centerline, distance_ratio_threshold)
        new_reconstructed_classes.append(int(condition))
    return new_reconstructed_classes


# %%
import os
from graph_neural_networks.data.dataset.dynamic.graph_transforms.oversample_nodes import OversampleNodesTransform
from graph_neural_networks.data.dataset.dynamic.graph_transforms.compute_distance_matrix import ComputeDistanceMatrixTransform
import torch
from utils.reconstruction.path_reconstruction.euclidean_path_reconstruction import EuclideanPathReconstructionMethod

data_folder = "/home/morand/afs/EVAPORE/data/FIVES_clean/"
gt_folder = os.path.join(data_folder, "gt")
pred_folder = os.path.join(data_folder, "pred")
img_folder = os.path.join(data_folder, "img")

max_dist = 100

gt_path_list = os.listdir(gt_folder)
gt_path_list.sort()
print(len(gt_path_list))

path_reconstruction_method = EuclideanPathReconstructionMethod()
train_transforms = {
        "compute_distance_matrix_transform": ComputeDistanceMatrixTransform(skip_first_neighbor=True),
        "oversample_nodes_transform": OversampleNodesTransform(max_dist=50, remove_original_edges=True),
}
eval_transforms = {
    "oversample_nodes_transform": OversampleNodesTransform(max_dist=50, remove_original_edges=True),
}

# %%
import os
from PIL import Image
import numpy as np
from graph.graph_creation import img_to_graph
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
from graph_neural_networks.data.dataset.dynamic.apply_graph_transforms import apply_graph_transforms
import networkx as nx
import torch
import time
import json

def process_case(i):
    graph_creation_time = 0
    path_creation_time = 0

    filename = gt_path_list[i]
    gt_path = os.path.join(gt_folder, filename)
    pred_path = os.path.join(pred_folder, filename)

    gt_np = np.array(Image.open(gt_path).convert("L")) > 0
    segmentation_mask_np = np.array(Image.open(pred_path).convert("L")) > 0

    # ======== Eval data ========
    graph_creation_start_time = time.time()
    pred_graph: nx.Graph = img_to_graph(segmentation_mask_np, clean = True, closing_radius=1, return_pixel_graph=False)
    graph_wrapper = GraphWrapper(pred_graph)
    new_nx_graph: nx.Graph = apply_graph_transforms(graph_wrapper, eval_transforms).get_graph()
    graph_creation_time = time.time() - graph_creation_start_time

    nodes_radius_data = {}
    for id, n_data in new_nx_graph.nodes(data=True):
        nodes_radius_data[id] = float(n_data["radius"])

    path_creation_start_time = time.time()
    virtual_edges_index_tensor = get_query_edges(new_nx_graph, n_closest=5, max_dist=max_dist)
    eval_edges = virtual_edges_index_tensor.t().tolist()
    eval_path_centerlines = path_reconstruction_method.reconstruct(map=None, graph=new_nx_graph, new_edges=virtual_edges_index_tensor)
    eval_path_centerlines = [[list([int(x), int(y)]) for (x, y) in path] for path in eval_path_centerlines]
    _ = cut_mask_from_negative_edges_for_all(eval_path_centerlines, segmentation_mask_np, edges=eval_edges, return_old_centerlines=True)
    path_creation_time = time.time() - path_creation_start_time

    return graph_creation_time, path_creation_time


# %%
graph_creation_times = []
path_creation_times = []
total_times = []

image_centerline_datamodule.setup()
for idx in image_centerline_datamodule.test_indices:
    case_id = gt_path_list[idx].split(".")[0]
    print(f"Processing case {idx}/{len(gt_path_list)}: {case_id}")
    graph_creation_time, path_creation_time = process_case(idx)

    graph_creation_times.append(graph_creation_time)
    path_creation_times.append(path_creation_time)
    total_times.append(graph_creation_time + path_creation_time)

    print(f"Case {idx}: Graph creation time: {graph_creation_time:.2f}s, Path creation time: {path_creation_time:.2f}s")

results = {
    "graph_creation_times": graph_creation_times,
    "path_creation_times": path_creation_times,
    "total_times": total_times
}

experiment_dir = "/home/morand/afs/EVAPORE/experiments/"
with open(os.path.join(experiment_dir, "times.txt"), "w") as f:
    json.dump(results, f, indent=4)

# %%
image_centerline_datamodule.setup()

experiment_dir = "/home/morand/afs/EVAPORE/experiments/"
with open(os.path.join(experiment_dir, "times.txt"), "r") as f:
    times = json.load(f)

graph_creation_times = times["graph_creation_times"]
path_creation_times = times["path_creation_times"]
total_times = times["total_times"]

mean_graph_creation_time, std_graph_creation_time = np.mean(graph_creation_times), np.std(graph_creation_times)
mean_path_creation_time, std_path_creation_time = np.mean(path_creation_times), np.std(path_creation_times)
mean_total_time, std_total_time = np.mean(total_times), np.std(total_times)

print(f"Graph creation time: {mean_graph_creation_time:.2f}s ± {std_graph_creation_time:.2f}s")
print(f"Path creation time: {mean_path_creation_time:.2f}s ± {std_path_creation_time:.2f}s")
print(f"Total time: {mean_total_time:.2f}s ± {std_total_time:.2f}s")

# %%
import gc
import matplotlib.pyplot as plt

from utils.reconstruction.radius_reconstruction.smallest_radius_reconstruction import SmallestRadiusReconstructionMethod
from utils.reconstruction.reconstruction_method import ReconstructionMethod

radius_reconstruction_method = SmallestRadiusReconstructionMethod()

model_inference_times = []
edge_prediction_times = []

with torch.no_grad():
    for batch in dataloader:
        model_inference_time = 0
        edge_prediction_time = 0

        img, _, (predict_edges, predict_nodes_radius, predict_paths, _), (gt, pred) = batch

        predict_edges_np = predict_edges.squeeze().cpu().numpy()
        predict_nodes_radius_np = predict_nodes_radius.squeeze().cpu().numpy()
        predict_paths_np = [path.type(torch.long).squeeze(dim=0).cpu().numpy() for path in predict_paths]
        src_nodes = np.unique(predict_edges_np[:, 0])
        src_edges = [predict_edges_np[predict_edges_np[:, 0] == src_node] for src_node in src_nodes]
        src_paths = []
        src_radius = []
        for src_node in src_nodes:
            src_paths.append([predict_paths_np[i] for i in range(len(predict_edges_np)) if predict_edges_np[i, 0] == src_node])
            src_radius.append([predict_nodes_radius_np[i] for i in range(len(predict_edges_np)) if predict_edges_np[i, 0] == src_node])
        src_edges_nb = [len(edges) for edges in src_edges]

        model_inference_time_start = time.time()
        edges_scores, _ = reduced_pipeline_model.forward(img, predict_paths)
        edges_scores = torch.sigmoid(edges_scores).detach().cpu().numpy()
        model_inference_time = time.time() - model_inference_time_start
        print(f"Model inference time: {model_inference_time:.2f}s")
        model_inference_times.append(model_inference_time)

        pred = pred.squeeze().cpu()

        edge_prediction_time_start = time.time()
        src_edges_scores = [edges_scores[predict_edges_np[:, 0] == src_node] for src_node in src_nodes]
        src_max_edge = [np.argmax(scores) for scores in src_edges_scores]
        src_max_score = [scores[max_edge] for scores, max_edge in zip(src_edges_scores, src_max_edge)]
        src_above_threshold = [score > best_threshold for score in src_max_score]

        reconstruction_paths = []
        reconstruction_radiuses = []
        for src_node_i, src_node in enumerate(src_nodes):
            edges = src_edges[src_node_i]
            max_edge_i = src_max_edge[src_node_i]
            above_threshold = src_above_threshold[src_node_i]
            path = src_paths[src_node_i][max_edge_i]
            r1, r2 = src_radius[src_node_i][max_edge_i]
            if above_threshold:
                radius_path = radius_reconstruction_method.reconstruct_one(r1, r2, path)
                reconstruction_paths.append(path)
                reconstruction_radiuses.append(radius_path)

        color_mask, full_mask = ReconstructionMethod.draw_reconstruction(pred, reconstruction_paths, reconstruction_radiuses)

        edge_prediction_time = time.time() - edge_prediction_time_start
        print(f"Edge prediction and reconstruction time: {edge_prediction_time:.2f}s")
        edge_prediction_times.append(edge_prediction_time)

        del img, pred, full_mask, gt

with open(os.path.join(experiment_dir, "times.txt"), "r") as f:
    times = json.load(f)

times["model_inference_times"] = model_inference_times
times["edge_prediction_times"] = edge_prediction_times
times["total_times"] = [t + model_time + edge_time for t, model_time, edge_time in zip(times["total_times"], model_inference_times, edge_prediction_times)]

with open(os.path.join(experiment_dir, "times.txt"), "w") as f:
    json.dump(times, f, indent=4)

gc.collect()
torch.cuda.empty_cache()

# %%
experiment_dir = "/home/morand/afs/EVAPORE/experiments/"
with open(os.path.join(experiment_dir, "times.txt"), "r") as f:
    times = json.load(f)

graph_creation_times = times["graph_creation_times"]
path_creation_times = times["path_creation_times"]
model_inference_times = times["model_inference_times"]
edge_prediction_times = times["edge_prediction_times"]
total_times = times["total_times"]

mean_graph_creation_time, std_graph_creation_time = np.mean(graph_creation_times), np.std(graph_creation_times)
mean_path_creation_time, std_path_creation_time = np.mean(path_creation_times), np.std(path_creation_times)
mean_model_inference_time, std_model_inference_time = np.mean(model_inference_times), np.std(model_inference_times)
mean_edge_prediction_time, std_edge_prediction_time = np.mean(edge_prediction_times), np.std(edge_prediction_times)
mean_total_time, std_total_time = np.mean(total_times), np.std(total_times)

print(f"Graph creation time: {mean_graph_creation_time:.2f}s ± {std_graph_creation_time:.2f}s")
print(f"Path creation time: {mean_path_creation_time:.2f}s ± {std_path_creation_time:.2f}s")
print(f"Model inference time: {mean_model_inference_time:.2f}s ± {std_model_inference_time:.2f}s")
print(f"Edge prediction time: {mean_edge_prediction_time:.2f}s ± {std_edge_prediction_time:.2f}s")
print(f"Total time: {mean_total_time:.2f}s ± {std_total_time:.2f}s")

# %%
y = np.array([mean_graph_creation_time, mean_path_creation_time, mean_model_inference_time, mean_edge_prediction_time])
plt.pie(y, labels=["Graph Creation", "Edge query selection", "Model Inference\n(Features generation,\nPath sampling,\npath encoding,\npath classification)", "Final Path Selection"], autopct='%1.1f%%', startangle=140)
plt.savefig(os.path.join(experiment_dir, "time_breakdown_pie_chart.svg"), format="svg")
plt.savefig(os.path.join(experiment_dir, "time_breakdown_pie_chart.png"), format="png")

# %%
import time
import gc
import matplotlib.pyplot as plt

from utils.reconstruction.radius_reconstruction.linear_interpolation_radius_reconstruction import LinearInterpolationRadiusReconstructionMethod
from utils.reconstruction.radius_reconstruction.smallest_radius_reconstruction import SmallestRadiusReconstructionMethod
from utils.reconstruction.reconstruction_method import ReconstructionMethod
from graph_neural_networks.models.metrics import Betti_error_0_2D, Betti_error_1_2D, CCDice, ClDice, BinaryDice

radius_reconstruction_method = SmallestRadiusReconstructionMethod()

metrics = [ 
    BinaryDice(),
    ClDice(),
    CCDice(),
    Betti_error_0_2D(),
    Betti_error_1_2D() 
]

metrics_before = {metric.__class__.__name__: [] for metric in metrics}
metrics_after = {metric.__class__.__name__: [] for metric in metrics}

with torch.no_grad():
    for batch in dataloader:
        img, _, (predict_edges, predict_nodes_radius, predict_paths, _), (gt, pred) = batch

        predict_edges_np = predict_edges.squeeze().cpu().numpy()
        predict_nodes_radius_np = predict_nodes_radius.squeeze().cpu().numpy()
        predict_paths_np = [path.type(torch.long).squeeze(dim=0).cpu().numpy() for path in predict_paths]
        src_nodes = np.unique(predict_edges_np[:, 0])
        src_edges = [predict_edges_np[predict_edges_np[:, 0] == src_node] for src_node in src_nodes]
        src_paths = []
        src_radius = []
        for src_node in src_nodes:
            src_paths.append([predict_paths_np[i] for i in range(len(predict_edges_np)) if predict_edges_np[i, 0] == src_node])
            src_radius.append([predict_nodes_radius_np[i] for i in range(len(predict_edges_np)) if predict_edges_np[i, 0] == src_node])
        src_edges_nb = [len(edges) for edges in src_edges]

        edges_scores, _ = reduced_pipeline_model.forward(img, predict_paths)
        edges_scores = torch.sigmoid(edges_scores).detach().cpu().numpy()

        pred = pred.squeeze().cpu()

        edge_prediction_time_start = time.time()
        src_edges_scores = [edges_scores[predict_edges_np[:, 0] == src_node] for src_node in src_nodes]
        src_max_edge = [np.argmax(scores) for scores in src_edges_scores]
        src_max_score = [scores[max_edge] for scores, max_edge in zip(src_edges_scores, src_max_edge)]
        src_above_threshold = [score > best_threshold for score in src_max_score]

        reconstruction_paths = []
        reconstruction_radiuses = []
        for src_node_i, src_node in enumerate(src_nodes):
            edges = src_edges[src_node_i]
            max_edge_i = src_max_edge[src_node_i]
            above_threshold = src_above_threshold[src_node_i]
            path = src_paths[src_node_i][max_edge_i]
            r1, r2 = src_radius[src_node_i][max_edge_i]
            if above_threshold:
                radius_path = radius_reconstruction_method.reconstruct_one(r1, r2, path)
                reconstruction_paths.append(path)
                reconstruction_radiuses.append(radius_path)

        color_mask, full_mask = ReconstructionMethod.draw_reconstruction(pred, reconstruction_paths, reconstruction_radiuses)

        pred = (pred > 0).type(torch.float32)
        gt = (gt.cpu().squeeze() > 0).type(torch.float32)

        # Metrics before reconstruction
        for metric in metrics:
            metric(pred, gt)
            print(f"{metric} before reconstruction: {metric.compute()}")
            metrics_before[metric.__class__.__name__].append(metric.compute())

        # Metrics after reconstruction
        for metric in metrics:
            metric(full_mask, gt)
            print(f"{metric} after reconstruction: {metric.compute()}")
            metrics_after[metric.__class__.__name__].append(metric.compute())

        del img, pred, full_mask, gt

gc.collect()
torch.cuda.empty_cache()

# %%
import time
import gc
import matplotlib.pyplot as plt

from utils.reconstruction.radius_reconstruction.linear_interpolation_radius_reconstruction import LinearInterpolationRadiusReconstructionMethod
from utils.reconstruction.radius_reconstruction.smallest_radius_reconstruction import SmallestRadiusReconstructionMethod
from utils.reconstruction.reconstruction_method import ReconstructionMethod
from graph_neural_networks.models.metrics import Betti_error_0_2D, Betti_error_1_2D, CCDice, ClDice, BinaryDice

radius_reconstruction_method = SmallestRadiusReconstructionMethod()

metrics = [ 
    BinaryDice(),
    ClDice(),
    CCDice(),
    Betti_error_0_2D(),
    Betti_error_1_2D() 
]

metrics_before = {metric.__class__.__name__: [] for metric in metrics}
metrics_after = {metric.__class__.__name__: [] for metric in metrics}

with torch.no_grad():
    for batch in dataloader:
        img, (predict_paths, _), _, (gt, pred) = batch

        predict_paths_np = [path.type(torch.long).squeeze(dim=0).cpu().numpy() for path in predict_paths]

        edges_scores, _ = reduced_pipeline_model.forward(img, predict_paths)
        edges_scores = torch.sigmoid(edges_scores).detach().cpu().numpy()
        edges_preds = (edges_scores > best_threshold).astype(int)

        pred = pred.squeeze().cpu()

        full_mask = pred.clone()
        for path, pred_i in zip(predict_paths_np, edges_preds):
            if pred_i == 1:
                full_mask[path[:, 0], path[:, 1]] = 1
        full_mask = (full_mask > 0).type(torch.float32)

        pred = (pred > 0).type(torch.float32)
        gt = (gt.cpu().squeeze() > 0).type(torch.float32)

        # Metrics before reconstruction
        for metric in metrics:
            metric(pred, gt)
            print(f"{metric} before reconstruction: {metric.compute()}")
            metrics_before[metric.__class__.__name__].append(metric.compute())

        # Metrics after reconstruction
        for metric in metrics:
            metric(full_mask, gt)
            print(f"{metric} after reconstruction: {metric.compute()}")
            metrics_after[metric.__class__.__name__].append(metric.compute())

        del img, pred, full_mask, gt

gc.collect()
torch.cuda.empty_cache()

# %%
import os
import json

experiment_dir = "/home/morand/afs/EVAPORE/experiments/"

before_reconstruction = {metric_name: [m.item() for m in values] for metric_name, values in metrics_before.items()}
after_reconstruction = {metric_name: [m.item() for m in values] for metric_name, values in metrics_after.items()}

metrics = {
    "before_reconstruction": before_reconstruction,
    "after_reconstruction": after_reconstruction
}

print(metrics)

with open(os.path.join(experiment_dir, "metrics.txt"), "w") as f:
    json.dump(metrics, f, indent=4)

# %%
metrics = {}
with open(os.path.join(experiment_dir, "metrics.txt"), "r") as f:
    metrics = json.load(f)

before_reconstruction = metrics["before_reconstruction"]
after_reconstruction = metrics["after_reconstruction"]

for metric_name in before_reconstruction.keys():
    before_values = before_reconstruction[metric_name]
    after_values = after_reconstruction[metric_name]

    diff = np.array(after_values) - np.array(before_values)

    mean_before = np.mean(before_values)
    std_before = np.std(before_values)
    mean_after = np.mean(after_values)
    std_after = np.std(after_values)

    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    print(f"{metric_name} before reconstruction: {mean_before:.4f} ± {std_before:.4f}")
    print(f"{metric_name} after reconstruction: {mean_after:.4f} ± {std_after:.4f}")
    print(f"{metric_name} difference (after - before): {mean_diff:.4f} ± {std_diff:.4f}")

# %%
import gc
import matplotlib.pyplot as plt

from utils.reconstruction.radius_reconstruction.linear_interpolation_radius_reconstruction import LinearInterpolationRadiusReconstructionMethod
from utils.reconstruction.radius_reconstruction.smallest_radius_reconstruction import SmallestRadiusReconstructionMethod
from utils.reconstruction.reconstruction_method import ReconstructionMethod
from graph_neural_networks.models.metrics import Betti_error_0_2D, Betti_error_1_2D, CCDice, ClDice, BinaryDice

radius_reconstruction_method = SmallestRadiusReconstructionMethod()

metrics = [ 
    BinaryDice(),
    ClDice(),
    CCDice(),
    Betti_error_0_2D(),
    Betti_error_1_2D() 
]

with torch.no_grad():
    for batch in dataloader:
        img, _, (predict_edges, predict_nodes_radius, predict_paths, _), (gt, pred) = batch

        predict_edges_np = predict_edges.squeeze().cpu().numpy()
        predict_nodes_radius_np = predict_nodes_radius.squeeze().cpu().numpy()
        predict_paths_np = [path.type(torch.long).squeeze(dim=0).cpu().numpy() for path in predict_paths]
        src_nodes = np.unique(predict_edges_np[:, 0])
        src_edges = [predict_edges_np[predict_edges_np[:, 0] == src_node] for src_node in src_nodes]
        src_paths = []
        src_radius = []
        for src_node in src_nodes:
            src_paths.append([predict_paths_np[i] for i in range(len(predict_edges_np)) if predict_edges_np[i, 0] == src_node])
            src_radius.append([predict_nodes_radius_np[i] for i in range(len(predict_edges_np)) if predict_edges_np[i, 0] == src_node])
        src_edges_nb = [len(edges) for edges in src_edges]

        edges_scores, _ = reduced_pipeline_model.forward(img, predict_paths)
        edges_scores = torch.sigmoid(edges_scores).detach().cpu().numpy()

        src_edges_scores = [edges_scores[predict_edges_np[:, 0] == src_node] for src_node in src_nodes]
        src_max_edge = [np.argmax(scores) for scores in src_edges_scores]
        src_max_score = [scores[max_edge] for scores, max_edge in zip(src_edges_scores, src_max_edge)]
        src_above_threshold = [score > best_threshold for score in src_max_score]

        pred = pred.squeeze().cpu()
        reconstruction_paths = []
        reconstruction_radiuses = []
        for src_node_i, src_node in enumerate(src_nodes):
            edges = src_edges[src_node_i]
            max_edge_i = src_max_edge[src_node_i]
            above_threshold = src_above_threshold[src_node_i]
            path = src_paths[src_node_i][max_edge_i]
            r1, r2 = src_radius[src_node_i][max_edge_i]
            if above_threshold:
                radius_path = radius_reconstruction_method.reconstruct_one(r1, r2, path)
                reconstruction_paths.append(path)
                reconstruction_radiuses.append(radius_path)

        color_mask, full_mask = ReconstructionMethod.draw_reconstruction(pred, reconstruction_paths, reconstruction_radiuses)

        img = img.cpu().squeeze().permute(1, 2, 0)
        img = np.clip((img * np.array(std) + np.array(mean)).numpy(), 0, 1)

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0,0].imshow(img)
        axs[0,0].axis('off')
        axs[0,0].set_title("Reconstructed Image")
        axs[0,1].imshow(pred.cpu().squeeze())
        axs[0,1].axis('off')
        axs[0,1].set_title("Predicted Mask")
        axs[1,0].imshow(full_mask.cpu().squeeze())
        axs[1,0].axis('off')
        axs[1,0].set_title("Reconstructed Mask")
        axs[1,1].imshow(gt.cpu().squeeze())
        axs[1,1].axis('off')
        axs[1,1].set_title("Ground Truth Mask")
        plt.show()

        print(pred.shape, pred.min(), pred.max(), pred.mean(), pred.dtype)
        print(gt.shape, gt.min(), gt.max(), gt.mean(), gt.dtype)
        pred = (pred > 0).type(torch.float32)
        gt = (gt.cpu().squeeze() > 0).type(torch.float32)
        print(pred.shape, pred.min(), pred.max(), pred.mean(), pred.dtype)
        print(gt.shape, gt.min(), gt.max(), gt.mean(), gt.dtype)

        # Metrics before reconstruction
        for metric in metrics:
            metric(pred, gt)
            print(f"{metric} before reconstruction: {metric.compute()}")

        # Metrics after reconstruction
        for metric in metrics:
            metric(full_mask, gt)
            print(f"{metric} after reconstruction: {metric.compute()}")

        del img, pred, full_mask, gt

        break

gc.collect()
torch.cuda.empty_cache()
