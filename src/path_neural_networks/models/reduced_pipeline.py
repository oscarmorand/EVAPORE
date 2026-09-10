import pytorch_lightning as pl
import torch.nn as nn
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from path_neural_networks.utils.symmetry_enforcement import SymmetryEnforcementMode
from path_neural_networks.models.features_generators import FeaturesGenerator
from path_neural_networks.models.path_samplers import PathSampler
from path_neural_networks.models.path_encoders import PathEncoder
from path_neural_networks.models.path_classifiers import PathClassifier
from path_neural_networks.models.losses import PathClassificationLoss
from path_neural_networks.utils.metric_registry import METRIC_REGISTRY, METRIC_THRESHOLDS_REGISTRY


class ReducedPipelineLitModule(pl.LightningModule):
    def __init__(
        self,
        features_generator: FeaturesGenerator,
        path_sampler: PathSampler,
        path_encoder: PathEncoder,
        path_classifier: PathClassifier,
        edge_classification_loss_fn: PathClassificationLoss,
        metrics: MetricCollection,
        lr: float = 1e-3,
        symmetry_enforcement_mode: SymmetryEnforcementMode = SymmetryEnforcementMode.NONE,
        inference_threshold: float = None,
        forward_full_volume: bool = True,
        bbox_margin_size: int = 10,
        path_chunk_size: int = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "features_generator",
                "path_sampler",
                "path_encoder",
                "path_classifier",
                "edge_classification_loss_fn",
            ]
        )

        self.features_generator = features_generator
        self.path_sampler = path_sampler
        self.path_encoder = path_encoder
        self.path_classifier = path_classifier
        self.edge_classification_loss_fn = edge_classification_loss_fn

        # Manual optimization is required to support accumulating gradients
        # across path chunks (multiple backward() calls before one step()).
        # When path_chunk_size is None, training_step behaves equivalently
        # to a single automatic-optimization step (one backward, one step).
        self.automatic_optimization = False

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
        self.inference_threshold = inference_threshold

    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "features_generator": self.features_generator.as_dict(),
            "path_sampler": self.path_sampler.as_dict(),
            "path_encoder": self.path_encoder.as_dict(),
            "path_classifier": self.path_classifier.as_dict(),
            "edge_classification_loss_fn": self.edge_classification_loss_fn.as_dict(),
            "metrics": self.base_metrics,
            "lr": self.hparams.lr,
            "symmetry_enforcement_mode": self.hparams.symmetry_enforcement_mode.value,
            "inference_threshold": self.inference_threshold,
            "forward_full_volume": self.hparams.forward_full_volume,
            "bbox_margin_size": self.hparams.bbox_margin_size,
            "path_chunk_size": self.hparams.path_chunk_size,
        }

    def forward(self,
                img: torch.Tensor,
                paths: list[torch.Tensor]
    ) -> torch.Tensor:
        if self.hparams.forward_full_volume:
            return self._forward_full_volume(img, paths)
        return self._forward_bbox(img, paths)

    def _bbox_sparse(
        self,
        path: torch.Tensor,
        margin_size: int,
        spatial_shape: torch.Size,
    ):
        """
        path: (1, path_length, ndim) -- batch dim assumed to be 1
        """
        assert path.shape[0] == 1, f"expected batch size 1, got {path.shape[0]}"
        path_2d = path.squeeze(0)  # (path_length, ndim)

        ndim = len(spatial_shape)
        assert path_2d.shape[1] == ndim, (
            f"path has {path_2d.shape[1]} coordinate columns but spatial_shape "
            f"has {ndim} dims — check path's layout (expected (length, ndim))"
        )

        spatial_shape_t = torch.as_tensor(
            spatial_shape, dtype=torch.long, device=path.device
        )

        path_long = path_2d.long()
        mins = path_long.min(dim=0).values - margin_size
        maxs = path_long.max(dim=0).values + margin_size

        mins = torch.clamp(mins, min=torch.zeros(ndim, dtype=torch.long, device=path.device))
        maxs = torch.clamp(maxs, max=spatial_shape_t - 1)

        crop_slices = tuple(
            slice(int(mins[d].item()), int(maxs[d].item()) + 1) for d in range(ndim)
        )
        local_path_2d = path_2d - mins.to(path_2d.dtype)
        local_path = local_path_2d.unsqueeze(0)  # remettre la dim de batch: (1, path_length, ndim)

        return crop_slices, local_path

    def _forward_full_volume(self,
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

    def _forward_bbox(
        self,
        img: torch.Tensor,
        paths: list[torch.Tensor],
    ):
        paths_logits = []
        spatial_shape = img.shape[2:]  # img: (1, C, *spatial_dims)

        for i, path in enumerate(paths):
            crop_slices, local_path = self._bbox_sparse(
                path, self.hparams.bbox_margin_size, spatial_shape
            )
            patch = img[(..., *crop_slices)]  # (1, C, *cropped_spatial_dims)

            patch_feature_maps = self.features_generator(patch)  # (1, out_channels, ...)

            # Sample features along path (using local, patch-relative coordinates)
            path_features = self.path_sampler(patch_feature_maps, local_path)
            # Encode path
            encoded_path = self.path_encoder(path_features)

            # Strict Symmetry enforcement (if needed) — mirrors _forward_full_volume
            if self.hparams.symmetry_enforcement_mode == SymmetryEnforcementMode.DOUBLE_PASS:
                inv_path_features = torch.flip(path_features, dims=[2])
                inv_encoded_path = self.path_encoder(inv_path_features)
                inv_path_logits = self.path_classifier(inv_encoded_path)
                path_logits = self.path_classifier(encoded_path)
                path_logits = (path_logits + inv_path_logits) / 2.0
            else:
                path_logits = self.path_classifier(encoded_path)

            paths_logits.append(path_logits)

        paths_logits = torch.cat(paths_logits, dim=1).squeeze()
        return paths_logits, None

    def _forward_chunk(self, img: torch.Tensor, paths_chunk: list[torch.Tensor]):
        """
        Runs self.forward's underlying dispatch (full-volume or bbox) but
        restricted to a subset of paths. For forward_full_volume=True, note
        this still recomputes features_generator(img) once per chunk — if
        you want the whole-volume features computed only once across all
        chunks, compute feature_maps outside and pass them in instead
        (see note in training_step).
        """
        if self.hparams.forward_full_volume:
            chunk_logits, _ = self._forward_full_volume(img, paths_chunk)
        else:
            chunk_logits, _ = self._forward_bbox(img, paths_chunk)
        return chunk_logits

    def _shared_step(self,
        batch: tuple,
        step: str,
        metrics: nn.ModuleDict
    ) -> torch.Tensor:
        img, (paths, edges_classes), _ = batch
        edges_classes = edges_classes.squeeze().type(torch.float)  # (num_edges,)

        N = len(paths)
        chunk_size = self.hparams.path_chunk_size or N

        all_logits = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            paths_chunk = paths[start:end]

            chunk_logits = self._forward_chunk(img, paths_chunk)
            if chunk_logits.dim() == 0:
                chunk_logits = chunk_logits.unsqueeze(0)  # keep 1D even for a 1-path chunk
            all_logits.append(chunk_logits)

        paths_logits = torch.cat(all_logits, dim=0)
        paths_probs = torch.sigmoid(paths_logits)

        loss = self.edge_classification_loss_fn(paths_logits, edges_classes)

        self.log(f"{step}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for name, metric in (metrics.items()):
            metric(paths_probs, edges_classes.int())
            self.log(f"{step}_{name}", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def _augment_batch_with_flipped_paths(self,
                                          batch: tuple
    ) -> tuple:
        img, (paths, edges_classes), _ = batch
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

        img, (paths, edges_classes), _ = batch
        edges_classes = edges_classes.squeeze().type(torch.float)  # (num_edges,)

        N = len(paths)
        chunk_size = self.hparams.path_chunk_size or N

        optimizer = self.optimizers()
        optimizer.zero_grad()

        all_logits = []  # detached, for metric logging over the full batch
        total_loss = 0.0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            paths_chunk = paths[start:end]
            n_chunk = end - start

            chunk_logits = self._forward_chunk(img, paths_chunk)
            if chunk_logits.dim() == 0:
                chunk_logits = chunk_logits.unsqueeze(0)  # keep 1D even for n_chunk == 1
            chunk_targets = edges_classes[start:end]

            chunk_loss = self.edge_classification_loss_fn(chunk_logits, chunk_targets)

            # Scale by this chunk's share of paths so that summing the
            # gradients of each scaled_chunk_loss.backward() call equals
            # the gradient of the mean loss over all N paths. If your loss
            # uses reduction='sum' instead of 'mean', drop this scaling.
            scaled_chunk_loss = chunk_loss * (n_chunk / N)

            self.manual_backward(scaled_chunk_loss)

            all_logits.append(chunk_logits.detach())
            total_loss = total_loss + scaled_chunk_loss.detach()

        optimizer.step()

        full_logits = torch.cat(all_logits, dim=0)
        full_probs = torch.sigmoid(full_logits)

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for name, metric in self.train_metrics.items():
            metric(full_probs, edges_classes.int())
            self.log(f"train_{name}", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self._shared_step(batch, step="val", metrics=self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self._shared_step(batch, step="test", metrics=self.test_metrics)
        return loss

    def predict_step(self, batch, batch_idx):
        img, _, (edges, radius, paths, _), _ = batch
        with torch.no_grad():
            paths_logits, _ = self.forward(img, paths)
        paths_probs = torch.sigmoid(paths_logits)
        return edges, radius, paths_probs

    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_train_epoch_end(self):
        # With automatic_optimization=False, Lightning does not step the
        # scheduler automatically — do it here to match the previous
        # per-epoch CosineAnnealingLR behavior.
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

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
        edge_classification_loss_fn: nn.Module,
        metrics: MetricCollection,
        lr: float = 1e-3,
        symmetry_enforcement_mode: SymmetryEnforcementMode = SymmetryEnforcementMode.NONE,
        inference_threshold: float = None,
        forward_full_volume: bool = True,
        bbox_margin_size: int = 10,
        path_chunk_size: int = None,
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
            edge_classification_loss_fn=edge_classification_loss_fn,
            metrics=metrics,
            lr=lr,
            symmetry_enforcement_mode=symmetry_enforcement_mode,
            inference_threshold=inference_threshold,
            forward_full_volume=forward_full_volume,
            bbox_margin_size=bbox_margin_size,
            path_chunk_size=path_chunk_size,
        )