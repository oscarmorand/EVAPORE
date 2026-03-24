import pytorch_lightning as pl
import torch.nn as nn
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision
from torchmetrics import MeanMetric, Metric, MetricCollection, MetricTracker

from path_neural_networks.models.metrics.binary_prauc import BinaryPRAUC
from path_neural_networks.utils.symmetry_enforcement import SymmetryEnforcementMode
from path_neural_networks.models.features_generators import FeaturesGenerator
from path_neural_networks.models.path_samplers import PathSampler
from path_neural_networks.models.path_encoders import PathEncoder
from path_neural_networks.models.path_classifiers import PathClassifier
from path_neural_networks.models.losses import PathClassificationLoss

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

class ReducedPipelineLitModule(pl.LightningModule):
    def __init__(
        self,
        features_generator: FeaturesGenerator,
        path_sampler: PathSampler,
        path_encoder: PathEncoder,
        path_classifier: PathClassifier,
        loss_fn: PathClassificationLoss,
        metrics: MetricCollection,
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
        self.loss_fn = loss_fn

        # Metrics
        self.base_metrics = metrics
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
        feature_maps = self.features_generator(img) # shape (1, features_generator.out_channels, H, W)

        # Sample features along paths
        paths_logits = []
        for i, path in enumerate(paths):
            # Sample features along path
            path_features = self.path_sampler(feature_maps, path)  # shape (1, path_sampler.out_channels, path_length)
            # Encode path
            encoded_path = self.path_encoder(path_features) # shape (1, path_encoder.out_channels)
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

class ReducedPipelineLitModuleHydra(ReducedPipelineLitModule):
    def __init__(
        self,
        features_generator_cfg: DictConfig,
        path_sampler_cfg: DictConfig,
        path_encoder_cfg: DictConfig,
        path_classifier_cfg: DictConfig,
        loss_fn: nn.Module,
        metrics: MetricCollection,
        lr: float = 1e-3,
        symmetry_enforcement_mode: SymmetryEnforcementMode = SymmetryEnforcementMode.NONE,
        inference_threshold: float = None
    ):
        features_generator = instantiate(features_generator_cfg)
        path_sampler = instantiate(path_sampler_cfg, in_channels=features_generator.out_channels)
        path_encoder = instantiate(path_encoder_cfg, in_channels=path_sampler.out_channels)
        path_classifier = instantiate(path_classifier_cfg, in_channels=path_encoder.out_channels)

        super().__init__(
            features_generator=features_generator,
            path_sampler=path_sampler,
            path_encoder=path_encoder,
            path_classifier=path_classifier,
            loss_fn=loss_fn,
            metrics=metrics,
            lr=lr,
            symmetry_enforcement_mode=symmetry_enforcement_mode,
            inference_threshold=inference_threshold
        )