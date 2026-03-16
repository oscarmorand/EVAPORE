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
            
        if len(paths_logits) == 0:
            paths_logits = None
        else:
            paths_logits = torch.cat(paths_logits, dim=1).squeeze()  # shape (num_edges,)

        return paths_logits, feature_maps

    def _shared_step(self, 
                     batch: tuple,
                     step: str,
                     metrics: nn.ModuleDict
    ) -> torch.Tensor:
        img, (paths, edges_classes), _, _ = batch
        paths_logits, _ = self.forward(img, paths)
        if paths_logits is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
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
        paths_probs = None
        if paths_logits is not None:
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

centerline_dir = "all_filtering_centerlines"

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

print(classes_ratio)
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

# %% [markdown]
# # TRAIN

# %%
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder

logger = CSVLogger(".")


callbacks = [
    ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}"
    ),
    ModelCheckpoint(
        monitor="val_pr_auc",
        mode="max", 
        save_top_k=1, 
        filename="best-pr-auc-checkpoint-{epoch:02d}-{val_pr_auc:.4f}"
    ),
    EarlyStopping(
        monitor="val_pr_auc",
        patience=20,
        min_delta=1e-3,
        verbose=True,
        mode="max"
    ),
    #LearningRateFinder()
]

torch.set_float32_matmul_precision("medium")
trainer = Trainer(accelerator='gpu', 
                  devices="auto",
                  num_nodes=1,
                  max_epochs=100,
                  precision="16-mixed",
                  detect_anomaly=False, 
                  callbacks=callbacks,
                  logger=logger,
                  gradient_clip_val=1.0,
                  gradient_clip_algorithm="norm",
                  val_check_interval=0.1
)
print(trainer)
print("Num GPUs:", trainer.num_devices)
print(trainer.max_epochs, trainer.max_steps, trainer.min_epochs)

# %%
trainer.fit(reduced_pipeline_model, datamodule=image_centerline_datamodule)

# %% [markdown]
# # TEST

# %%
from pytorch_lightning import Trainer

ckpt_path = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_261/checkpoints/best-pr-auc-checkpoint-epoch=01-val_pr_auc=0.8201.ckpt"

reduced_pipeline_model = ReducedPipelineModel.load_from_checkpoint(
    ckpt_path,
    features_generator=features_generator,
    path_sampler=path_sampler,
    path_encoder=path_encoder,
    path_classifier=path_classifier,
    edge_classification_loss_fn=loss_fn,
)

torch.set_float32_matmul_precision("medium")
trainer = Trainer(accelerator='gpu', devices="auto", max_epochs=10, precision='16-mixed')

# %%
trainer.test(reduced_pipeline_model, datamodule=image_centerline_datamodule)

# %%
all_preds = []
all_targets = []

reduced_pipeline_model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(image_centerline_datamodule.val_dataloader()):
        img, (paths, edges_classes), _, _ = batch
        path_logits, _ = reduced_pipeline_model(img, paths)
        if path_logits is None:
            continue
        path_probs = torch.sigmoid(path_logits)
        print(f"Batch {batch_idx}: path_probs shape: {path_probs.shape}")
        all_preds.append(path_probs.detach().cpu())
        # get true labels for each edge
        _, (paths, edges_classes), _, _ = batch
        print(f"Batch {batch_idx}: edges_classes shape: {edges_classes.shape}")
        assert path_probs.shape[0] == edges_classes.shape[1], f"Number of edge predictions ({path_probs.shape[0]}) does not match number of edge labels ({edges_classes.shape[1]})"
        all_targets.append(edges_classes.squeeze().detach().cpu())

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

probs = torch.sigmoid(all_preds)

# %%
from torchmetrics.classification import BinaryPrecisionRecallCurve

pr_curve = BinaryPrecisionRecallCurve()
precision, recall, thresholds = pr_curve(probs, all_targets)

max_f1_threshold = thresholds[torch.argmax(2 * (precision * recall) / (precision + recall + 1e-8))]

print(max_f1_threshold)

# %%
reduced_pipeline_model.set_inference_threshold(max_f1_threshold.item())

# %%
all_preds = []
all_targets = []

reduced_pipeline_model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(image_centerline_datamodule.test_dataloader()):
        img, (paths, edges_classes), _, _ = batch
        path_logits, _ = reduced_pipeline_model(img, paths)
        path_probs = torch.sigmoid(path_logits)
        print(f"Batch {batch_idx}: path_probs shape: {path_probs.shape}")
        all_preds.append(path_probs.detach().cpu())
        # get true labels for each edge
        _, (paths, edges_classes), _, _ = batch
        print(f"Batch {batch_idx}: edges_classes shape: {edges_classes.shape}")
        assert path_probs.shape[0] == edges_classes.shape[1], f"Number of edge predictions ({path_probs.shape[0]}) does not match number of edge labels ({edges_classes.shape[1]})"
        all_targets.append(edges_classes.squeeze().detach().cpu())

probs = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryROC
from torchmetrics.utilities.compute import auc
import matplotlib.pyplot as plt

# ROC Curve
roc_curve = BinaryROC()
fpr, tpr, roc_thresholds = roc_curve(probs, all_targets)
auroc = auc(fpr, tpr)

# Precision-Recall Curve
pr_curve = BinaryPrecisionRecallCurve()
precision, recall, thresholds = pr_curve(probs, all_targets)
pr_auc = auc(recall, precision)

# find the tpr and fpr at the best F1 threshold
best_tpr = tpr[torch.argmin(torch.abs(roc_thresholds - max_f1_threshold))].item()
best_fpr = fpr[torch.argmin(torch.abs(roc_thresholds - max_f1_threshold))].item()

# find the precision and recall at the best F1 threshold
best_precision = precision[torch.argmax(2 * (precision * recall) / (precision + recall + 1e-8))].item()
best_recall = recall[torch.argmax(2 * (precision * recall) / (precision + recall + 1e-8))].item()

# Confusion Matrix at the best F1 threshold
cm = confusion_matrix(all_targets.numpy(), (probs.numpy() > max_f1_threshold.item()).astype(int))
cm = cm / cm.sum(axis=1, keepdims=True)  # Normalize by true class counts
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

# Dice vs threshold curve and confusion proportions vs threshold
eps = 1e-8
thresholds_dense = torch.quantile(probs, torch.linspace(0, 1, steps=100))
thresholds_dense = torch.unique(thresholds_dense)

targets = all_targets.bool()
dice_scores = []
tp_list, fp_list, tn_list, fn_list = [], [], [], []
for t in thresholds_dense:
    preds = probs >= t

    tp = (preds & targets).sum().float()
    fp = (preds & ~targets).sum().float()
    tn = (~preds & ~targets).sum().float()
    fn = (~preds & targets).sum().float()

    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    dice_scores.append(dice)
    tp_list.append(tp)
    fp_list.append(fp)
    tn_list.append(tn)
    fn_list.append(fn)

dice_scores = torch.stack(dice_scores)
tp, fp, tn, fn = torch.stack(tp_list), torch.stack(fp_list), torch.stack(tn_list), torch.stack(fn_list)
true_class_0_total = tn + fp
true_class_1_total = tp + fn
tp_prop = tp / true_class_1_total
fp_prop = fp / true_class_0_total
tn_prop = tn / true_class_0_total
fn_prop = fn / true_class_1_total

# ============= Visualization =============

fig, axs = plt.subplots(4, 2, figsize=(16, 22))

axs[0, 0].plot(fpr.numpy(), tpr.numpy(), label=f"ROC curve (AUROC={auroc.item():.4f})")
axs[0, 0].plot([0, 1], [0, 1], 'k--', label="Random guess")
axs[0, 0].fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
axs[0, 0].fill_between(fpr.numpy(), tpr.numpy(), alpha=0.1, color='blue')
axs[0, 0].scatter(best_fpr, best_tpr, color='red', label=f"Best F1 point (threshold={max_f1_threshold.item():.4f})")
axs[0, 0].set_xlabel("False Positive Rate")
axs[0, 0].set_ylabel("True Positive Rate")
axs[0, 0].set_title("ROC Curve (Test set)")
axs[0, 0].grid(True)
axs[0, 0].legend()

axs[0, 1].plot(recall.numpy(), precision.numpy(), label=f"PR curve (AUC={pr_auc.item():.4f})")
axs[0, 1].fill_between(recall.numpy(), precision.numpy(), alpha=0.1, color='blue')
axs[0, 1].scatter(best_recall, best_precision, color='red', label=f"Best F1 point (threshold={max_f1_threshold.item():.4f})")
axs[0, 1].set_xlabel("Recall")
axs[0, 1].set_ylabel("Precision")
axs[0, 1].set_title("Precision-Recall Curve (Test set)")
axs[0, 1].grid(True)
axs[0, 1].legend()

axs[1, 0].hist(probs[all_targets == 1], bins=50, alpha=0.6, label="Positive")
axs[1, 0].hist(probs[all_targets == 0], bins=50, alpha=0.6, label="Negative")
axs[1, 0].axvline(x=max_f1_threshold.item(), color='red', linestyle='--', label=f"Best F1 threshold: {max_f1_threshold.item():.4f}")
axs[1, 0].set_title("Histogram of predicted probabilities")
axs[1, 0].legend()
axs[1, 0].set_xlabel("Predicted probability")

axs[1, 1].hist(probs[all_targets == 1], density=True, bins=50, alpha=0.6, label="Positive")
axs[1, 1].hist(probs[all_targets == 0], density=True, bins=50, alpha=0.6, label="Negative")
axs[1, 1].axvline(x=max_f1_threshold.item(), color='red', linestyle='--', label=f"Best F1 threshold: {max_f1_threshold.item():.4f}")
axs[1, 1].set_yscale("log")
axs[1, 1].set_title("Normalized histogram of predicted probabilities")
axs[1, 1].legend()
axs[1, 1].set_xlabel("Predicted probability")

axs[2, 0].plot(thresholds_dense.numpy(), fp_prop.numpy()) 
axs[2, 0].fill_between(thresholds_dense.numpy(), fp_prop.numpy(), y2=0.0, alpha=0.1, color='red', label="FP area")
axs[2, 0].fill_between(thresholds_dense.numpy(), fp_prop.numpy(), y2=1.0, alpha=0.1, color='green', label="TN area")
axs[2, 0].axvline(x=max_f1_threshold.item(), color='red', linestyle='--', label=f"Best F1 threshold: {max_f1_threshold.item():.4f}") 
axs[2, 0].set_xlabel("Threshold") 
axs[2, 0].set_ylabel("Proportion of total samples for true class 0") 
axs[2, 0].set_title("Confusion Matrix (True class 0) Proportions vs Threshold") 
axs[2, 0].legend()
axs[2, 0].grid(True)

axs[2, 1].plot(thresholds_dense.numpy(), fn_prop.numpy()) 
axs[2, 1].fill_between(thresholds_dense.numpy(), fn_prop.numpy(), y2=0.0, alpha=0.1, color='red', label="FN area")
axs[2, 1].fill_between(thresholds_dense.numpy(), fn_prop.numpy(), y2=1.0, alpha=0.1, color='green', label="TP area")
axs[2, 1].axvline(x=max_f1_threshold.item(), color='red', linestyle='--', label=f"Best F1 threshold: {max_f1_threshold.item():.4f}") 
axs[2, 1].set_xlabel("Threshold") 
axs[2, 1].set_ylabel("Proportion of total samples for true class 1") 
axs[2, 1].set_title("Confusion Matrix (True class 1) Proportions vs Threshold") 
axs[2, 1].legend()
axs[2, 1].grid(True)

axs[3, 0].plot(thresholds_dense.numpy(), dice_scores.numpy())
axs[3, 0].axvline(x=max_f1_threshold.item(), color='red', linestyle='--', label=f"Best F1 threshold: {max_f1_threshold.item():.4f}")
axs[3, 0].set_xlabel("Threshold")
axs[3, 0].set_ylabel("Dice Score")
axs[3, 0].set_title("Dice Score vs Threshold")
axs[3, 0].legend()
axs[3,0].grid(True)

disp.plot(ax=axs[3, 1], values_format='.2f', cmap='Blues')
axs[3, 1].set_title(f"Confusion Matrix (Test set, threshold={max_f1_threshold.item():.4f})")

plt.savefig("/home/morand/afs/EVAPORE/experiments/classification_metrics.svg", format="svg")
plt.show()

# %%
# compute metrics at the best F1 threshold

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

preds_binary = (probs.numpy() > max_f1_threshold.item()).astype(int)

accuracy = accuracy_score(all_targets.numpy(), preds_binary)
precision = precision_score(all_targets.numpy(), preds_binary)
recall = recall_score(all_targets.numpy(), preds_binary)
f1 = f1_score(all_targets.numpy(), preds_binary)

print(f"Test set metrics at best F1 threshold ({max_f1_threshold.item():.4f}):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUROC: {auroc.item():.4f}")
print(f"PR AUC: {pr_auc.item():.4f}")

# %%
dataloader = image_centerline_datamodule.predict_dataloader()
#dataloader = image_centerline_datamodule.train_dataloader()

# %%
import matplotlib.pyplot as plt
import gc

with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        img, _, _, _ = batch

        img = img.squeeze().cpu().numpy().transpose(1,2,0)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        plt.imshow(img)
        plt.show()
        plt.close()

        del img

        break

gc.collect()
torch.cuda.empty_cache()

# %%
print(image_centerline_datamodule.train_indices, image_centerline_datamodule.val_indices, image_centerline_datamodule.test_indices)

print(image_centerline_dataset.img_list[image_centerline_datamodule.test_indices.numpy()[1]])
print(image_centerline_dataset.img_list[image_centerline_datamodule.test_indices.numpy()[5]])

# %%
import matplotlib.pyplot as plt
import gc

with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        img, (paths, edges_classes), _, (gt, pred) = batch
        
        edges_classes = edges_classes.squeeze().type(torch.float)
        pred_path_class, _ = reduced_pipeline_model.forward(img, paths)
        edge_pred_binarized = torch.sigmoid(pred_path_class) > max_f1_threshold.item()

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

        if batch_idx >= 10:
            break

gc.collect()
torch.cuda.empty_cache()

# %%
import gc

image_centerline_datamodule.setup()
dataloader = image_centerline_datamodule.predict_dataloader()

with torch.no_grad():
    for batch in dataloader:
        img, (paths, edges_classes), (_, _, eval_path_centerlines, _), (gt, pred) = batch
        
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

rgb = False

with torch.no_grad():
    for batch in dataloader:
        img, _, (_, _, eval_path_centerlines, _), _, = batch
        _, feature_maps = reduced_pipeline_model.forward(img, eval_path_centerlines)

        if rgb:
            for i in range(0, feature_maps.shape[1] - 3, 3):
                feature_map = feature_maps[0, i:i+3, :, :].cpu().detach().numpy()
                feature_map = ((feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                plt.figure(figsize=(15,15))
                plt.imshow(feature_map)
                plt.axis('off')
                plt.title(f"Feature map {i} to {i + 2}")
                plt.show()
                plt.close()

        else:
            for i in range(0,feature_maps.shape[1]):
                feature_map = feature_maps[0, i, :, :].cpu().detach().numpy()
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255.0
                feature_map_img = np.zeros((feature_map.shape[0], feature_map.shape[1], 3), dtype=np.uint8)
                feature_map_img[:, :, 0] = feature_map.astype(np.uint8)
                feature_map_img[:, :, 1] = feature_map.astype(np.uint8)
                feature_map_img[:, :, 2] = feature_map.astype(np.uint8)
                #Image.fromarray(feature_map.astype(np.uint8)).save(f"feature_map_{i}.png")
                plt.figure(figsize=(15,15))
                plt.imshow(feature_map_img)
                plt.axis('off')
                plt.title(f"Feature map {i}")
                plt.show()
                plt.close()

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
        img, _, (eval_edges, _, eval_path_centerlines, _), (gt, pred) = batch
        pred_path_class, _ = reduced_pipeline_model.forward(img, eval_path_centerlines)
        edge_pred_binarized = torch.sigmoid(pred_path_class) > max_f1_threshold.item()

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
        img, _, (predict_edges, _, predict_paths, _), (gt, pred) = batch

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
        src_above_threshold = [score > max_f1_threshold.item() for score in src_max_score]
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
            if not above_threshold:
                background_img[path[:, 0], path[:, 1]] = np.array([255, 0, 0], dtype=np.uint8)  # Red for predicted negative
        for src_node_i, src_node in enumerate(src_nodes):
            edges = src_edges[src_node_i]
            max_edge_i = src_max_edge[src_node_i]
            above_threshold = src_above_threshold[src_node_i]
            path = src_paths[src_node_i][max_edge_i]
            if above_threshold:
                background_img[path[:, 0], path[:, 1]] = np.array([0, 255, 0], dtype=np.uint8)  # Green for predicted positive

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
        img, (paths, edges_classes), _, (gt, pred) = batch
        
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

gc.collect()
torch.cuda.empty_cache()

# %%
interest_path_idx = [157, 190]

with torch.no_grad():
    for batch in dataloader:
        img, (paths, edges_classes), _, (gt, pred) = batch
        
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
interest_path_idx = [157, 190]

with torch.no_grad():
    for batch in dataloader:
        img, (paths, edges_classes), _, (gt, pred) = batch
        
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
        img, (paths, edges_classes),  _, (gt, pred) = batch
        
        inverse_paths = []
        for path in paths:
            path_np = path.type(torch.long).squeeze(dim=0).cpu().numpy()
            inverse_path_np = path_np[::-1].copy()
            inverse_path = torch.tensor(inverse_path_np, dtype=torch.long).unsqueeze(dim=0).to(path.device)
            inverse_paths.append(inverse_path)

        edges_classes = edges_classes.squeeze().type(torch.float)

        pred_path_class, feature_maps = reduced_pipeline_model.forward(img, paths)
        inv_pred_path_class, inv_feature_maps = reduced_pipeline_model.forward(img, inverse_paths)

        edge_pred_binarized = torch.sigmoid(pred_path_class) > max_f1_threshold.item()
        inv_edge_pred_binarized = torch.sigmoid(inv_pred_path_class) > max_f1_threshold.item()

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
        img, (paths, edges_classes), _, (gt, pred) = batch
        
        edges_classes = edges_classes.squeeze().type(torch.float)
        pred_path_class, feature_maps = reduced_pipeline_model.forward(img, paths)
        probabilities = torch.sigmoid(pred_path_class)
        edge_pred_binarized = probabilities > max_f1_threshold.item()

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

from utils.reconstruction.radius_reconstruction.linear_interpolation_radius_reconstruction import LinearInterpolationRadiusReconstructionMethod
from utils.reconstruction.radius_reconstruction.smallest_radius_reconstruction import SmallestRadiusReconstructionMethod
from utils.reconstruction.reconstruction_method import ReconstructionMethod

print(reduced_pipeline_model)

radius_reconstruction_method = SmallestRadiusReconstructionMethod()

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
        src_above_threshold = [score > max_f1_threshold.item() for score in src_max_score]

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
