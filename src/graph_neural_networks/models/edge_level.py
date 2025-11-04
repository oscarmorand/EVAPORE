from collections.abc import Callable
from typing import Literal

import torch
from torch import nn
from torch_geometric.data import Batch
from torchmetrics import Metric, MetricCollection, MetricTracker

from graph_neural_networks.models import GraphLitModule
from graph_neural_networks.utils.logging_utils import log_nonscalar_metrics, pad_keys
import wandb

class LinkPredictionLitModule(GraphLitModule):
    """A LightningModule for GNNs aimed at link prediction tasks."""

    task_level = "edge"

    def __init__(
        self,
        task: Literal["binary"],
        encoder: nn.Module,
        decoder: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        """Initializes a `LinkPredictionLitModule`.

        Args:
            encoder: The GNN model used to encode the graph.
            decoder: The decoder used to make predictions based on the edge representations.
            *args: Additional positional arguments to pass to the superclass.
            **kwargs: Additional keyword arguments to pass to the superclass.
        """

        super().__init__(*args, **kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data: Batch) -> torch.Tensor:
        """Perform a forward pass through the model on a batch of graphs.

        Args:
            data: A batch of graphs, represented as one big (disconnected) graph.

        Returns:
            The predicted logits for the input graphs in the batch.
        """
        x, batch, batch_size = data.x, data.batch, data.batch_size
        # Cast input features that must be floats to floats
        y = self.encoder(
            x.float(),
            data.edge_index,
            edge_weight=data.edge_weight.float() if data.edge_weight is not None else None,
            edge_attr=data.edge_attr.float() if data.edge_attr is not None else None,
            batch=batch,
            batch_size=batch_size,
        )
        return y

    def model_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch: A batch of data containing the input tensor of images and target labels.

        Returns:
            A pair of tensors containing the loss and the (unnormalized) predictions (i.e. logits), respectively.
        """
        logits = self.forward(batch)
        out = self.decoder(logits, batch.edge_label_index)
        loss = self.criterion(out, batch.edge_label.float())
        return loss, out
    
    def _shared_eval_step(
        self,
        batch: Batch,
        loss_metric: Metric | MetricTracker,
        scalar_metrics: MetricCollection | MetricTracker | None = None,
        nonscalar_metrics: MetricCollection | None = None,
    ) -> torch.Tensor:
        # Perform the forward pass on the model and compute the loss
        loss, out = self.model_step(batch)

        # Update the stateful loss and metrics
        loss_metric.update(loss)
        if scalar_metrics:
            scalar_metrics.update(out, batch.edge_label)
        if nonscalar_metrics:
            nonscalar_metrics.update(out, batch.edge_label)

        return loss
    
    def _shared_epoch_end(
        self,
        prefix: str,
        loss_metric: Metric | MetricTracker,
        scalar_metrics: MetricCollection | MetricTracker | None = None,
        nonscalar_metrics: MetricCollection | None = None,
    ) -> None:
        # Log the loss and metrics accumulated over the epoch, and the best values so far
        # Note: best values are logged at every epoch, instead of only once at the end of the loop, because limitations
        # in Lightning's logging mean metrics logged on loop ends are not available to callbacks
        # (see issue recommending to log on epoch end as a workaround: https://github.com/Lightning-AI/pytorch-lightning/issues/5285)
        self.log(f"{prefix}loss", loss_metric.compute(), prog_bar=True)
        if isinstance(loss_metric, MetricTracker):
            self.log(f"{prefix}loss/best", loss_metric.best_metric(), prog_bar=True)
        if scalar_metrics:
            # Call the `compute` method explicitly, instead of passing the metric object to `log_dict` and relying on
            # Lightning to aggregate the metrics. This avoids issues when relying on the `MetricCollection` to flatten
            # the output dictionary, which is not supported when called internally by Lightning.
            self.log_dict(scalar_metrics.compute(), prog_bar=True)
            if isinstance(scalar_metrics, MetricTracker):
                self.log_dict(pad_keys(scalar_metrics.best_metric(), postfix="/best"), prog_bar=True)
        if nonscalar_metrics:
            # Warning: temporary workaround to log confusion matrix to wandb
            for k in nonscalar_metrics.keys():
                if 'confusion_matrix' in k:
                    binary_confusion_matrix = nonscalar_metrics[k].compute().cpu().numpy()
                    y_sums = binary_confusion_matrix.sum(axis=1)
                    y_true = ([0] * y_sums[0]) + ([1] * y_sums[1])
                    y_pred = ([0] * binary_confusion_matrix[0,0].item()) + ([1] * binary_confusion_matrix[0,1].item()) + ([0] * binary_confusion_matrix[1,0].item()) + ([1] * binary_confusion_matrix[1,1].item())
                    wandb.log({f"{k}": wandb.plot.confusion_matrix(probs=None,
                                                                  y_true=y_true,
                                                                  preds=y_pred,
                                                                  class_names=["no_edge", "edge"])})
