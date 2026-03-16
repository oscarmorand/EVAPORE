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
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# %%
print("PyTorch version:", torch.__version__)
print("is cuda available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from graph_neural_networks.data.dataset.image_dataset import ImageDataset
from graph_neural_networks.data.datamodules.image_datamodule import ImageDatamodule
from graph_neural_networks.models.unet import UNet

# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2

# %%
dataset = "FIVES"
data_dir = f"/home/morand/afs/EVAPORE/data/{dataset}/"

if dataset == "FIVES":
    mean = torch.tensor([0.3320, 0.1500, 0.0595])
    std = torch.tensor([0.2471, 0.1254, 0.0573])

# %%
train_transforms = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.2,
        p=0.5
    ),
    A.GaussNoise(p=0.05),
    A.Normalize(mean=mean.tolist(), std=std.tolist()),
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

val_transforms = A.Compose([
    A.Normalize(mean=mean.tolist(), std=std.tolist()),
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

dataset = ImageDataset(data_dir)

datamodule = ImageDatamodule(dataset=dataset, 
                             val_split=0.2,
                             test_split=0.1,
                             train_transforms=train_transforms,
                             val_transforms=val_transforms,
                             test_transforms=val_transforms,
                             num_workers=7,
                             train_batch_size=1, 
                             val_batch_size=1,
                             seed=42)

# %%
from abc import ABC

class PathSampler(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                feature_maps: torch.Tensor,
                path: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
    
class SinglePointPathSampler(PathSampler):
    def __init__(self):
        super().__init__()

    def forward(self,
                feature_maps: torch.Tensor,
                path: torch.Tensor
    ) -> torch.Tensor:
        path = path.type(torch.long).squeeze(dim=0)  # shape (path_length, 2)
        path_features = feature_maps[:, :, path[:, 0], path[:, 1]]  # shape (1, channels, path_length)
        return path_features

class SquarePathSampler(PathSampler):
    def __init__(self, square_size: int = 3, aggregation: str = "max"):
        super().__init__()
        self.square_size = square_size
        self.aggregation = aggregation

        self.half_size = square_size // 2
        
        # Pre-compute relative offsets for the square
        # shape: (square_size * square_size, 2)
        offsets = torch.stack([
            torch.arange(-self.half_size, self.half_size + 1).repeat_interleave(square_size),
            torch.arange(-self.half_size, self.half_size + 1).repeat(square_size)
        ], dim=1)
        self.register_buffer('offsets', offsets)
    
    def forward(self,
                feature_maps: torch.Tensor,
                path: torch.Tensor
        ) -> torch.Tensor:
        """
        Args:
            feature_maps: (batch, channels, height, width)
            path: (1, path_length, 2) or (path_length, 2)
        
        Returns:
            path_features: (batch, channels, path_length)
        """
        path = path.type(torch.long).squeeze(dim=0)  # shape (path_length, 2)
        batch_size, channels, height, width = feature_maps.shape
        path_length = path.shape[0]
        
        # Expand path coordinates with offsets
        # path: (path_length, 2) -> (path_length, 1, 2)
        # offsets: (square_size^2, 2) -> (1, square_size^2, 2)
        path_expanded = path.unsqueeze(1)  # (path_length, 1, 2)
        offsets_expanded = self.offsets.unsqueeze(0)  # (1, square_size^2, 2)
        
        # Broadcast and add: (path_length, square_size^2, 2)
        sample_coords = path_expanded + offsets_expanded
        
        # Flatten to (path_length * square_size^2, 2)
        sample_coords = sample_coords.reshape(-1, 2)
        
        # Clamp coordinates to valid range
        sample_coords[:, 0] = torch.clamp(sample_coords[:, 0], 0, height - 1)
        sample_coords[:, 1] = torch.clamp(sample_coords[:, 1], 0, width - 1)
        
        # Sample features: (batch, channels, path_length * square_size^2)
        sampled = feature_maps[:, :, sample_coords[:, 0], sample_coords[:, 1]]
        
        # Reshape to (batch, channels, path_length, square_size^2)
        sampled = sampled.reshape(batch_size, channels, path_length, -1)
        
        # Aggregate over the square (options: max, mean, sum)
        if self.aggregation == "max":
            path_features = sampled.max(dim=-1).values  # (batch, channels, path_length)
        elif self.aggregation == "mean":
            path_features = sampled.mean(dim=-1)  # (batch, channels, path_length)
        elif self.aggregation == "sum":
            path_features = sampled.sum(dim=-1)  # (batch, channels, path_length)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return path_features


# %%
import pytorch_lightning as pl
import numpy as np
from graph_neural_networks.models.path_encoders.path_encoder import PathEncoder
from graph_neural_networks.reconstruction.reconstruction_method import PathReconstructionMethod
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
from graph_neural_networks.data.dataset.dynamic.graph_transforms.add_edge_closest_cc import AddEdgeClosestCCTransform
from graph_neural_networks.data.dataset.dynamic.graph_transforms.oversample_nodes import OversampleNodesTransform
from graph_neural_networks.data.dataset.dynamic.apply_graph_transforms import apply_graph_transforms
from graph_neural_networks.data.utils.pred_state import get_combined_graph
from graph_neural_networks.data.utils.pred_state import EdgePredState
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class FullPipelineModel(pl.LightningModule):
    def __init__(
        self,
        image_segmentator: nn.Module,
        features_generator: nn.Module,
        path_reconstruction_method: PathReconstructionMethod,
        path_encoder: PathEncoder,
        path_classifier: nn.Module,
        edge_classification_loss_fn: nn.Module,
        lr = 1e-3,
    ):
        super().__init__()
        self.image_segmentator = image_segmentator
        self.features_generator = features_generator
        self.path_reconstruction_method = path_reconstruction_method
        self.path_encoder = path_encoder
        self.path_classifier = path_classifier

        self.edge_classification_loss_fn = edge_classification_loss_fn

        self.lr = lr

        self.transforms = {
            "oversample_nodes_transform": OversampleNodesTransform(max_dist=100, remove_original_edges=True),
            "add_edge_cc_transform": AddEdgeClosestCCTransform(only_one_edge_per_cc=True)
        }
        
    def _shared_step(self, 
                     batch: tuple[torch.Tensor, torch.Tensor], 
                     step: str
    ):
        img, gt = batch
        device = img.device

        # Segment the image
        probability_map: torch.Tensor = self.image_segmentator(img) # shape (1, 1, H, W)
        logger.debug(f"Probability map: {probability_map.shape}")
        segmentation_mask: torch.Tensor = torch.sigmoid(probability_map) > 0.5 # shape (1, 1, H, W)
        logger.debug(f"Segmentation mask: {segmentation_mask.shape}")

        # Compute the feature maps
        feature_maps = self.features_generator(img) # shape (1, channels, H, W)
        logger.debug(f"Feature maps: {feature_maps.shape}")
        
        # Graph creation, transforms application and edge querying are done on CPU
        gt_np: np.ndarray = gt.squeeze().cpu().numpy().astype(np.bool)
        segmentation_mask_np: np.ndarray = segmentation_mask.squeeze().cpu().numpy().astype(np.bool)

        combined_graph: nx.Graph = get_combined_graph(gt_np, segmentation_mask_np)
        logger.debug(f"Combined graph: {combined_graph}")

        graph_wrapper = GraphWrapper(combined_graph)
        new_nx_graph: nx.Graph = apply_graph_transforms(graph_wrapper, self.transforms).get_graph()
        logger.debug(f"Combined graph after transforms: {new_nx_graph}")

        # Query edges
        edges = [] # TODO: get the edge split
        edges_classes = [] # TODO: get the ground truth edges (true classification)
        for u, v, d in new_nx_graph.edges(data=True):
            logger.debug(f"Edge data keys: {d.keys()}")
            edges.append((u, v))
            edges.append((v, u))
            edge_class = d.get("edge_pred_state", None)
            edges_classes.append(0)
            edges_classes.append(0)
            break
        edges = torch.tensor(edges, device="cpu", dtype=torch.long).t() # shape (2, num_edges)
        edges_classes = torch.tensor(edges_classes, device=device, dtype=torch.float) # shape (num_edges,)
        logger.debug(f"Edges to classify: {edges.shape}")
        logger.debug(f"Edges classes: {edges_classes.shape}")
    
        # Reconstruct paths centerlines from edges
        heightmap = probability_map.squeeze().detach().cpu()
        paths = self.path_reconstruction_method.reconstruct(map=heightmap, graph=new_nx_graph, new_edges=edges) # List of paths, each path is a list of (x,y) coordinates

        # Sample features along paths
        edges_pred = []
        for i, path in enumerate(paths):
            # Back to GPU (if GPU is used)
            path = torch.tensor(path, device=device, dtype=torch.long)  # shape (path_length, 2)
            logger.debug(f"Path {i}: {path.shape}")

            path_features = feature_maps[:, :, path[:, 0], path[:, 1]]  # shape (1, channels, path_length)
            logger.debug(f"Path {i} features: {path_features.shape}")
        
            # Encode path
            encoded_path = self.path_encoder(path_features) # shape (1, out_channels)
            logger.debug(f"Encoded path {i}: {encoded_path.shape}")
        
            # Classify path
            path_class = self.path_classifier(encoded_path)
            logger.debug(f"Path {i} class: {path_class.shape}")

            edges_pred.append(path_class)
        edges_pred = torch.cat(edges_pred, dim=1).squeeze()  # shape (num_edges,)
        logger.debug(f"Edges predictions: {edges_pred.shape}")

        edge_classification_loss = self.edge_classification_loss_fn(edges_pred, edges_classes)
        self.log(f"{step}_edge_classification_loss", edge_classification_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return edge_classification_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def evaluation_step(self, batch, batch_idx):
        return self._shared_step(batch, "eval")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


# %%
import pytorch_lightning as pl
import torch.nn as nn
import torch
from graph_neural_networks.models.path_encoders.path_encoder import PathEncoder
import logging
from enum import Enum
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class SymmetryEnforcementMode(Enum):
    DATA_AUGMENTATION = "data_augmentation"
    DOUBLE_PASS = "double_pass"
    SYMMETRIC_KERNELS = "symmetric_kernels"
    NONE = "none" 

class ReducedPipelineModel(pl.LightningModule):
    def __init__(
        self,
        features_generator: nn.Module,
        path_sampler: PathSampler,
        path_encoder: PathEncoder,
        path_classifier: nn.Module,
        edge_classification_loss_fn: nn.Module,
        lr = 1e-3,
        symmetry_enforcement_mode: SymmetryEnforcementMode = SymmetryEnforcementMode.NONE,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["features_generator", "path_sampler", "path_encoder", "path_classifier", "edge_classification_loss_fn"])
        self.features_generator = features_generator
        self.path_sampler = path_sampler
        self.path_encoder = path_encoder
        self.path_classifier = path_classifier

        self.edge_classification_loss_fn = edge_classification_loss_fn

        # Metrics
        self.train_metrics = nn.ModuleDict({
            "accuracy": BinaryAccuracy(),
            "auroc": BinaryAUROC(),
            "recall": BinaryRecall(),
            "precision": BinaryPrecision(),
        })
        self.val_metrics = nn.ModuleDict({
            "accuracy": BinaryAccuracy(),
            "auroc": BinaryAUROC(),
            "recall": BinaryRecall(),
            "precision": BinaryPrecision(),
        })
        self.test_metrics = nn.ModuleDict({
            "accuracy": BinaryAccuracy(),
            "auroc": BinaryAUROC(),
            "recall": BinaryRecall(),
            "precision": BinaryPrecision(),
        })

    def forward(self, 
                img: torch.Tensor,
                paths: list[torch.Tensor]
    ) -> torch.Tensor:
    
        # Compute the feature maps
        feature_maps = self.features_generator(img) # shape (1, channels, H, W)
        logger.debug(f"Feature maps: {feature_maps.shape}")

        # Sample features along paths
        edges_pred = []
        for i, path in enumerate(paths):
            # Sample features along path
            path_features = self.path_sampler(feature_maps, path)  # shape (1, channels, path_length)
            # Encode path
            encoded_path = self.path_encoder(path_features) # shape (1, out_channels)
            # Classify path
            path_class = self.path_classifier(encoded_path)

            # Strict Symmetry enforcement (if needed)
            if self.hparams.symmetry_enforcement_mode == SymmetryEnforcementMode.DOUBLE_PASS:
                inv_path_features = torch.flip(path_features, dims=[2])  # shape (1, channels, path_length)
                inv_encoded_path = self.path_encoder(inv_path_features) # shape (1, out_channels)
                inv_path_class = self.path_classifier(inv_encoded_path)
                path_class = (path_class + inv_path_class) / 2.0

            edges_pred.append(path_class)
            
        edges_pred = torch.cat(edges_pred, dim=1).squeeze()  # shape (num_edges,)
        logger.debug(f"Edges predictions: {edges_pred.shape}")

        return edges_pred, feature_maps

    def _shared_step(self, 
                     batch: tuple
    ) -> torch.Tensor:
        img, paths, edges_classes, _, _, _, _, _ = batch
        edges_pred, _ = self.forward(img, paths)
        edges_classes = edges_classes.squeeze().type(torch.float)

        loss = self.edge_classification_loss_fn(edges_pred, edges_classes)

        return loss, edges_pred, edges_classes

    def _augment_batch_with_flipped_paths(self,
                                          batch: tuple
    ) -> tuple:
        img, paths, edges_classes, _, _, _, _, _ = batch
        flipped_paths = []
        for path in paths:
            flipped_path = torch.flip(path, dims=[0])
            flipped_paths.append(flipped_path)
        augmented_paths = paths + flipped_paths
        augmented_edges_classes = torch.cat([edges_classes, edges_classes], dim=1)
        return img, augmented_paths, augmented_edges_classes, None, None, None, None

    def training_step(self, batch, batch_idx):
        if self.hparams.symmetry_enforcement_mode == SymmetryEnforcementMode.DATA_AUGMENTATION:
            batch = self._augment_batch_with_flipped_paths(batch)
        loss, edges_pred, edges_classes = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for name, metric in self.train_metrics.items():
            metric(edges_pred, edges_classes.int())
            self.log(f"train_{name}", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, edges_pred, edges_classes = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for name, metric in self.val_metrics.items():
            metric(edges_pred, edges_classes.int())
            self.log(f"val_{name}", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        loss, edges_pred, edges_classes = self._shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for name, metric in self.test_metrics.items():
            metric(edges_pred, edges_classes.int())
            self.log(f"test_{name}", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        img, _, _, edges, radius, paths, _, _ = batch
        edges_scores, _ = self.forward(img, paths)
        return edges, radius, edges_scores

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


# %%
from graph_neural_networks.models.unet import UNet
from graph_neural_networks.models.path_encoders.conv_max_path_encoder import ConvMaxPathEncoder
from graph_neural_networks.models.path_encoders.pooling_encoder import MaxPoolingEncoder, NonLearnedPoolingEncoder
from graph_neural_networks.models.path_classifiers.fcn_classifier import FCNClassifier
from graph_neural_networks.models.binary_segmentator import BinarySegmentator

features_generator_to_encoder = 32
encoder_to_classifier = 256
pretrained_features_generator = True
freeze_pretrained_features_generator = False
symmetry_enforcement_mode = SymmetryEnforcementMode.DATA_AUGMENTATION
use_symmetric_kernels = symmetry_enforcement_mode == SymmetryEnforcementMode.SYMMETRIC_KERNELS
sampling_square_size = 3
sampling_aggregation_method = "mean"

if pretrained_features_generator:
    features_generator_model = BinarySegmentator.load_from_checkpoint("/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_35_train_unet_seg/checkpoints/best-checkpoint-epoch=19-val_loss=0.1323.ckpt", map_location=device)
    features_generator = features_generator_model.net

    if freeze_pretrained_features_generator:
        for param in features_generator.parameters():
            param.requires_grad = False

    features_generator.layers[7] = nn.Conv2d(in_channels=32, out_channels=features_generator_to_encoder, kernel_size=1, stride=1)

    for i, layer in enumerate(features_generator.layers):
        trainable = any(param.requires_grad for param in layer.parameters())
        print(f"Layer {i}: {'Trainable' if trainable else 'Frozen'}")
else:
    features_generator = UNet(input_channels=3, num_classes=features_generator_to_encoder, num_layers=4, features_start=32, bilinear=False)
    

path_sampler = SquarePathSampler(square_size=sampling_square_size, aggregation=sampling_aggregation_method)

#path_encoder = ConvMaxPathEncoder(in_channels=features_generator_to_encoder, hidden_features=[64, 128], out_channels=encoder_to_classifier, symmetric=use_symmetric_kernels)
path_encoder = MaxPoolingEncoder()

if isinstance(path_encoder, NonLearnedPoolingEncoder):
    encoder_to_classifier = features_generator_to_encoder * path_encoder.out_channels_factor
path_classifier = FCNClassifier(in_features=encoder_to_classifier, hidden_features=[128, 64], num_classes=1)

classification_loss_fn = nn.BCEWithLogitsLoss()

reduced_pipeline_model = ReducedPipelineModel(
    features_generator=features_generator,
    path_sampler=path_sampler,
    path_encoder=path_encoder,
    path_classifier=path_classifier,
    edge_classification_loss_fn=classification_loss_fn,
    lr=1e-3,
    symmetry_enforcement_mode=symmetry_enforcement_mode,
)

# %%
from graph_neural_networks.data.dataset.image_centerline_dataset import ImageCenterlineDataset
from graph_neural_networks.data.datamodules.image_centerline_datamodule import ImageCenterlineDatamodule

image_centerline_dataset = ImageCenterlineDataset("/home/morand/afs/EVAPORE/data/FIVES/")
image_centerline_datamodule = ImageCenterlineDatamodule(dataset=image_centerline_dataset, 
                                                         val_split=0.2,
                                                         test_split=0.1,
                                                         num_workers=16,
                                                         seed=42)

# TODO: add callbacks (model checkpointing, early stopping, learning rate monitoring, etc.)
# TODO: add data augmentation (random flips, rotations, etc.)

# %%
from pytorch_lightning import Trainer

torch.set_float32_matmul_precision("medium")
trainer = Trainer(accelerator='gpu', devices="auto", max_epochs=100, precision='16-mixed')

trainer.fit(reduced_pipeline_model, datamodule=image_centerline_datamodule)

# %%
from graph_neural_networks.models.unet import UNet
from graph_neural_networks.models.path_encoders.conv_max_path_encoder import ConvMaxPathEncoder
from graph_neural_networks.models.path_classifiers.fcn_classifier import FCNClassifier

symmetry_enforcement_mode = SymmetryEnforcementMode.DATA_AUGMENTATION
use_symmetric_kernels = symmetry_enforcement_mode == SymmetryEnforcementMode.SYMMETRIC_KERNELS
features_generator_to_encoder = 32
encoder_to_classifier = 256
sampling_square_size = 3
sampling_aggregation_method = "mean"

features_generator = UNet(input_channels=3, num_classes=features_generator_to_encoder, num_layers=4, features_start=32, bilinear=False)
path_sampler = SquarePathSampler(square_size=sampling_square_size, aggregation=sampling_aggregation_method)
#path_sampler = SinglePointPathSampler()
path_encoder = ConvMaxPathEncoder(in_channels=features_generator_to_encoder, hidden_features=[64, 128], out_channels=encoder_to_classifier, symmetric=use_symmetric_kernels)
path_classifier = FCNClassifier(in_features=encoder_to_classifier, hidden_features=[128, 64], num_classes=1)
classification_loss_fn = nn.BCEWithLogitsLoss()

#ckpt_path = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_98_train_reduced_1/checkpoints/epoch=0-step=420.ckpt"
#ckpt_path = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_118_train_reduced_2/checkpoints/epoch=9-step=4200.ckpt"
#ckpt_path = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_122_train_reduced_3/checkpoints/epoch=3-step=1680.ckpt"
#ckpt_path = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_124_train_reduced_freezed/checkpoints/epoch=2-step=1260.ckpt"
#ckpt_path = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_127_train_reduced_pretrained_not_freezed/checkpoints/epoch=9-step=4200.ckpt"
#ckpt_path = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_167_train_reduced_pretrained_with_double_pass/checkpoints/epoch=5-step=2520.ckpt"
ckpt_path = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_173_train_reduced_pretrained_square_sampling/checkpoints/epoch=5-step=2520.ckpt"

reduced_pipeline_model = ReducedPipelineModel.load_from_checkpoint(
    ckpt_path,
    features_generator=features_generator,
    path_sampler=path_sampler,
    path_encoder=path_encoder,
    path_classifier=path_classifier,
    edge_classification_loss_fn=classification_loss_fn,
)

# %%
from pytorch_lightning import Trainer

torch.set_float32_matmul_precision("medium")
trainer = Trainer(accelerator='gpu', devices="auto", max_epochs=10, precision='16-mixed')

# %%
trainer.test(reduced_pipeline_model, datamodule=image_centerline_datamodule)

# %%
edges_preds_binarized = trainer.predict(reduced_pipeline_model, datamodule=image_centerline_datamodule)

# %%
dataloader = image_centerline_datamodule.predict_dataloader()
#dataloader = image_centerline_datamodule.train_dataloader()

# %%
import matplotlib.pyplot as plt
import gc

with torch.no_grad():
    for batch in dataloader:
        img, paths, edges_classes, _, _ ,_, gt, pred = batch
        
        edges_classes = edges_classes.squeeze().type(torch.float)
        pred_path_class, _ = reduced_pipeline_model.forward(img, paths)
        edge_pred_binarized = torch.sigmoid(pred_path_class) > 0.5

        background_img = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
        pred_np = pred.squeeze().cpu().numpy().astype(bool)
        background_img[pred_np] = np.array([255, 255, 255], dtype=np.uint8) 

        for path, path_pred_class, true_class in zip(paths, edge_pred_binarized, edges_classes):
            path = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            path_pred_class = path_pred_class.cpu().item()
            true_class = true_class.cpu().item()
            if path_pred_class:
                if true_class:
                    background_img[path[:, 0], path[:, 1]] = np.array([0, 255, 0], dtype=np.uint8)  # Green for predicted positive correct
                else:
                    background_img[path[:, 0], path[:, 1]] = np.array([255, 255, 0], dtype=np.uint8)  # Yellow for predicted positive wrong
            else:
                if true_class:
                    background_img[path[:, 0], path[:, 1]] = np.array([255, 0, 0], dtype=np.uint8)  # Red for predicted negative wrong
                else:
                    background_img[path[:, 0], path[:, 1]] = np.array([0, 0, 255], dtype=np.uint8)  # Blue for predicted negative correct

        plt.subplots(1, 2, figsize=(80, 40))
        plt.subplot(1, 2, 1)
        plt.imshow(background_img)
        plt.subplot(1, 2, 2)
        plt.imshow(gt.squeeze().cpu().numpy(), cmap='gray')
        plt.show()
        plt.close()

        del img, background_img, paths, pred, pred_np

        break

gc.collect()
torch.cuda.empty_cache()

# %%
import gc

image_centerline_datamodule.setup()
dataloader = image_centerline_datamodule.predict_dataloader()

with torch.no_grad():
    for batch in dataloader:
        img, paths, edges_classes, _, _, eval_path_centerlines, gt, pred = batch
        
        background_img = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
        pred_np = pred.squeeze().cpu().numpy().astype(bool)
        background_img[pred_np] = np.array([255, 255, 255], dtype=np.uint8) 

        for path in paths:
            colormap = plt.get_cmap('autumn')
            path = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            for i, (x, y) in enumerate(path):
                color = colormap(i / len(path))  # RGBA
                color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                background_img[x, y] = color_rgb

        for path in eval_path_centerlines:
            colormap = plt.get_cmap('winter')
            path = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            for i, (x, y) in enumerate(path):
                color = colormap(i / len(path))  # RGBA
                color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                background_img[x, y] = color_rgb
            
        plt.figure(figsize=(80, 80))
        plt.imshow(background_img)
        plt.show()
        plt.close()

        del background_img, img, gt, pred, pred_np

        break

torch.cuda.empty_cache()
gc.collect()

# %%
import gc
from PIL import Image

with torch.no_grad():
    for batch in dataloader:
        img, _, _, _, _, eval_path_centerlines, _, _ = batch
        _, feature_maps = reduced_pipeline_model.forward(img, eval_path_centerlines)

        for i in range(0,feature_maps.shape[1]):
            feature_map = feature_maps[0, i, :, :].cpu().detach().numpy()
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255.0
            feature_map_img = np.zeros((feature_map.shape[0], feature_map.shape[1], 3), dtype=np.uint8)
            feature_map_img[:, :, 0] = feature_map.astype(np.uint8)
            feature_map_img[:, :, 1] = feature_map.astype(np.uint8)
            feature_map_img[:, :, 2] = feature_map.astype(np.uint8)
            feature_map_img = Image.fromarray(feature_map.astype(np.uint8)).save(f"feature_map_{i}.png")
            break

            del feature_map
        del feature_maps
        del img

        break

gc.collect()
torch.cuda.empty_cache()

# %%
image_centerline_datamodule.setup()
dataloader = image_centerline_datamodule.predict_dataloader()

with torch.no_grad():
    for batch in dataloader:
        img, _, _, eval_edges, _, eval_path_centerlines, gt, pred = batch
        pred_path_class, _ = reduced_pipeline_model.forward(img, eval_path_centerlines)
        edge_pred_binarized = torch.sigmoid(pred_path_class) > 0.5

        print(pred_path_class.detach().cpu().numpy())
        print(edge_pred_binarized.detach().cpu().numpy())

        background_img = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
        pred_np = pred.squeeze().cpu().numpy().astype(bool)
        background_img[pred_np] = np.array([255, 255, 255], dtype=np.uint8) 

        for path, edge_score in zip(eval_path_centerlines, edge_pred_binarized):
            path = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            if edge_score.item() < 0.5:
                background_img[path[:, 0], path[:, 1]] = np.array([255, 0, 0], dtype=np.uint8)  # Red for predicted negative
        for path, edge_score in zip(eval_path_centerlines, edge_pred_binarized):
            path = path.type(torch.long).squeeze(dim=0).cpu().numpy()    
            if edge_score.item() >= 0.5:
                background_img[path[:, 0], path[:, 1]] = np.array([0, 255, 0], dtype=np.uint8)  # Green for predicted positive

        plt.figure(figsize=(40, 40))
        plt.imshow(background_img)
        plt.show()
        plt.close()

        del background_img
        del img
        del gt
        del pred
        del pred_np

        break

gc.collect()
torch.cuda.empty_cache()

# %%

with torch.no_grad():
    for batch in dataloader:
        img, _, _, predict_edges, _, predict_paths, gt, pred = batch

        predict_edges = predict_edges.squeeze()
        predict_edges = predict_edges.cpu().numpy()

        predict_paths_np = [path.type(torch.long).squeeze(dim=0).cpu().numpy() for path in predict_paths]

        src_nodes = np.unique(predict_edges[:, 0])
        print(src_nodes)
        src_edges = [predict_edges[predict_edges[:, 0] == src_node] for src_node in src_nodes]
        print(src_edges)
        src_paths = []
        for src_node in src_nodes:
            src_paths.append([predict_paths_np[i] for i in range(len(predict_edges)) if predict_edges[i, 0] == src_node])
        print(src_paths)
        src_edges_nb = [len(edges) for edges in src_edges]
        print(src_edges_nb)

        print(img.shape)
        print(predict_edges.shape)
        print(len(predict_paths), predict_paths[0].shape)
        print(gt.shape)
        print(pred.shape)

        edges_scores, _ = reduced_pipeline_model.forward(img, predict_paths)
        edges_scores = torch.sigmoid(edges_scores).detach().cpu().numpy()

        print("edges scores:", edges_scores)
        src_edges_scores = [edges_scores[predict_edges[:, 0] == src_node] for src_node in src_nodes]
        print("src edges scores:", src_edges_scores)
        src_max_edge = [np.argmax(scores) for scores in src_edges_scores]
        print("src max edge:", src_max_edge)
        src_max_score = [scores[max_edge] for scores, max_edge in zip(src_edges_scores, src_max_edge)]
        print("src max score:", src_max_score)
        src_above_threshold = [score > 0.5 for score in src_max_score]
        print("src above threshold:", src_above_threshold)

        background_img = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
        pred_np = pred.squeeze().cpu().numpy().astype(np.bool)
        background_img[pred_np] = np.array([255, 255, 255], dtype=np.uint8)

        for path in predict_paths_np:
            background_img[path[:, 0], path[:, 1]] = np.array([0, 0, 255], dtype=np.uint8)  # Blue for all predicted paths

        for src_node_i, src_node in enumerate(src_nodes):
            edges = src_edges[src_node_i]
            max_edge_i = src_max_edge[src_node_i]
            above_threshold = src_above_threshold[src_node_i]
            path = src_paths[src_node_i][max_edge_i]
            if above_threshold:
                background_img[path[:, 0], path[:, 1]] = np.array([0, 255, 0], dtype=np.uint8)  # Green for predicted positive
            else:
                background_img[path[:, 0], path[:, 1]] = np.array([255, 0, 0], dtype=np.uint8)  # Red for predicted negative

        plt.subplots(1, 2, figsize=(80, 40))
        plt.subplot(1, 2, 1)
        plt.imshow(background_img)
        plt.subplot(1, 2, 2)
        plt.imshow(gt.squeeze().cpu().numpy())
        plt.show()

        del background_img
        del img
        del pred
        del gt
        del pred_np

        break

gc.collect()
torch.cuda.empty_cache()

# %%
with torch.no_grad():
    for batch in dataloader:
        img, paths, edges_classes, _, _, _, gt, pred = batch
        
        edges_classes = edges_classes.squeeze().type(torch.float)
        pred_path_class, feature_maps = reduced_pipeline_model.forward(img, paths)
        edge_pred_binarized = torch.sigmoid(pred_path_class) > 0.5

        background_img = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
        pred_np = pred.squeeze().cpu().numpy().astype(np.bool)
        background_img[pred_np] = np.array([255, 255, 255], dtype=np.uint8)

        plt.figure(figsize=(80,80))
        for path_i, (path, path_pred_class, true_class) in enumerate(zip(paths, edge_pred_binarized, edges_classes)):
            path = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            middle_pixel = path[len(path)//2]

            path_pred_class = path_pred_class.cpu().item()
            true_class = true_class.cpu().item()
            color = None
            if path_pred_class:
                if true_class:
                    color = np.array([0, 255, 0], dtype=np.uint8)  # Green for predicted positive correct
                else:
                    color = np.array([255, 255, 0], dtype=np.uint8)  # Yellow for predicted positive wrong
            else:
                if true_class:
                    color = np.array([255, 0, 0], dtype=np.uint8)  # Red for predicted negative wrong
                else:
                    color = np.array([0, 0, 255], dtype=np.uint8)  # Blue for predicted negative correct
            background_img[path[:, 0], path[:, 1]] = color

            color = color.astype(np.float32) / 255.0
            plt.text(middle_pixel[1], middle_pixel[0], str(path_i), color=(float(color[0]), float(color[1]), float(color[2])))

        plt.imshow(background_img)
        plt.show()
        plt.close('all')

        break
        for path_i, (path, path_pred_class, true_class) in enumerate(zip(paths, edge_pred_binarized, edges_classes)):
            path = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            path_pred_class = path_pred_class.cpu().item()
            true_class = true_class.cpu().item()

            plt.figure()
            plt.title(f"Features along path {path_i}, pred class: {path_pred_class}, true class: {true_class}")
            for feature_i in range(feature_maps.shape[1]):
                feature_map = feature_maps[0, feature_i, :, :].cpu().detach().numpy()
                feature_path = feature_map[path[:, 0], path[:, 1]]
                plt.plot(feature_path, alpha=0.4)
            plt.show()
            plt.close('all')

        del img, paths, edges_classes, background_img, pred_np

        break

gc.collect()
torch.cuda.empty_cache()

# %%
interest_path_idx = [127, 162]

with torch.no_grad():
    for batch in dataloader:
        img, paths, edges_classes, _, _, _, gt, pred = batch
        
        edges_classes = edges_classes.squeeze().type(torch.float)
        pred_path_class, feature_maps = reduced_pipeline_model.forward(img, paths)
        edge_pred_binarized = torch.sigmoid(pred_path_class) > 0.5

        for path_i, (path, path_pred_class, true_class) in enumerate(zip(paths, edge_pred_binarized, edges_classes)):
            if path_i not in interest_path_idx:
                continue
            path = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            path_pred_class = path_pred_class.cpu().item()
            true_class = true_class.cpu().item()

            plt.figure()
            plt.title(f"Features along path {path_i}, pred class: {path_pred_class}, true class: {true_class}")
            for feature_i in range(feature_maps.shape[1]):
                feature_map = feature_maps[0, feature_i, :, :].cpu().detach().numpy()
                feature_path = feature_map[path[:, 0], path[:, 1]]
                plt.plot(feature_path, alpha=0.4)
            plt.show()
            plt.close('all')

        del img, paths, edges_classes

        break

gc.collect()
torch.cuda.empty_cache()

# %%
interest_path_idx = [127, 162]

with torch.no_grad():
    for batch in dataloader:
        img, paths, edges_classes, _, _, _, gt, pred = batch
        
        inverse_paths = []
        for path in paths:
            path_np = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            inverse_path_np = path_np[::-1].copy()
            inverse_path = torch.tensor(inverse_path_np, dtype=torch.long).unsqueeze(dim=0).to(path.device)
            inverse_paths.append(inverse_path)

        edges_classes = edges_classes.squeeze().type(torch.float)

        pred_path_class, feature_maps = reduced_pipeline_model.forward(img, paths)
        inv_pred_path_class, inv_feature_maps = reduced_pipeline_model.forward(img, inverse_paths)

        edge_pred_binarized = torch.sigmoid(pred_path_class) > 0.5
        inv_edge_pred_binarized = torch.sigmoid(inv_pred_path_class) > 0.5

        for path_i, (path, path_pred_class, inv_path_pred_class, true_class) in enumerate(zip(paths, edge_pred_binarized, inv_edge_pred_binarized, edges_classes)):
            if path_i not in interest_path_idx:
                continue
            path = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            path_pred_class = path_pred_class.cpu().item()
            inv_path_pred_class = inv_path_pred_class.cpu().item()
            true_class = true_class.cpu().item()
            print(f"Path {path_i}, pred class: {path_pred_class}, inverse pred class: {inv_path_pred_class}, true class: {true_class}")

        break

gc.collect()
torch.cuda.empty_cache()

# %%
with torch.no_grad():
    for batch in dataloader:
        img, paths, edges_classes, _, _, _, gt, pred = batch
        
        inverse_paths = []
        for path in paths:
            path_np = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            inverse_path_np = path_np[::-1].copy()
            inverse_path = torch.tensor(inverse_path_np, dtype=torch.long).unsqueeze(dim=0).to(path.device)
            inverse_paths.append(inverse_path)

        edges_classes = edges_classes.squeeze().type(torch.float)

        pred_path_class, feature_maps = reduced_pipeline_model.forward(img, paths)
        inv_pred_path_class, inv_feature_maps = reduced_pipeline_model.forward(img, inverse_paths)

        edge_pred_binarized = torch.sigmoid(pred_path_class) > 0.5
        inv_edge_pred_binarized = torch.sigmoid(inv_pred_path_class) > 0.5

        diff = np.logical_xor(edge_pred_binarized.cpu().numpy(), inv_edge_pred_binarized.cpu().numpy())
        diff_indices = np.where(diff)[0]
        print("Different predictions at indices:", diff_indices)
        print("Percentage of different predictions:", len(diff_indices) / len(paths) * 100)

        break

gc.collect()
torch.cuda.empty_cache()

# %%
with torch.no_grad():
    for batch in dataloader:
        img, paths, edges_classes, _, _, _, gt, pred = batch
        
        edges_classes = edges_classes.squeeze().type(torch.float)
        pred_path_class, feature_maps = reduced_pipeline_model.forward(img, paths)
        probabilities = torch.sigmoid(pred_path_class)
        edge_pred_binarized = probabilities > 0.5

        background_img = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
        pred_np = pred.squeeze().cpu().numpy().astype(np.bool)
        background_img[pred_np] = np.array([255, 255, 255], dtype=np.uint8)

        plt.figure(figsize=(80,80))
        for path_i, (path, path_pred_class, true_class, probability) in enumerate(zip(paths, edge_pred_binarized, edges_classes, probabilities)):
            path = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            middle_pixel = path[len(path)//2]
            probability = probability.squeeze().cpu().numpy()

            path_pred_class = path_pred_class.cpu().item()
            true_class = true_class.cpu().item()
            color = None
            if path_pred_class:
                if true_class:
                    color = np.array([0, 255, 0], dtype=np.uint8)  # Green for predicted positive correct
                else:
                    color = np.array([255, 255, 0], dtype=np.uint8)  # Yellow for predicted positive wrong
            else:
                if true_class:
                    color = np.array([255, 0, 0], dtype=np.uint8)  # Red for predicted negative wrong
                else:
                    color = np.array([0, 0, 255], dtype=np.uint8)  # Blue for predicted negative correct
            background_img[path[:, 0], path[:, 1]] = color

            color = color.astype(np.float32) / 255.0
            plt.text(middle_pixel[1], middle_pixel[0], str(probability), color=(float(color[0]), float(color[1]), float(color[2])))

        plt.imshow(background_img)
        plt.show()
        plt.close('all')

        del img, paths, edges_classes, background_img, pred_np

        break

gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# # Applying the inference results on the prediction

# %%
import gc
import matplotlib.pyplot as plt

from graph_neural_networks.reconstruction.radius_reconstruction.linear_interpolation_radius_reconstruction import LinearInterpolationRadiusReconstructionMethod
from graph_neural_networks.reconstruction.radius_reconstruction.smallest_radius_reconstruction import SmallestRadiusReconstructionMethod
from graph_neural_networks.reconstruction.reconstruction_method import ReconstructionMethod

radius_reconstruction_method = SmallestRadiusReconstructionMethod()

with torch.no_grad():
    for batch in dataloader:
        img, _, _, predict_edges, predict_nodes_radius, predict_paths, gt, pred = batch

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
        src_above_threshold = [score > 0.5 for score in src_max_score]

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

        img, full_mask = ReconstructionMethod.draw_reconstruction(pred, reconstruction_paths, reconstruction_radiuses)

        plt.figure(figsize=(80, 80))
        plt.imshow(img)
        plt.show()

        del img, pred, full_mask, gt

        break

gc.collect()
torch.cuda.empty_cache()

# %%
