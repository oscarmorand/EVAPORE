import torch
import torch.nn as nn
import pytorch_lightning as pl

from path_neural_networks.utils.metric_registry import METRIC_REGISTRY

class NaiveModelLitModule(pl.LightningModule):
    def __init__(
        self,
        max_dist: float = None,
        metrics: list[str] = None,
    ):
        super().__init__()
        self.max_dist = max_dist

        # Metrics
        self.base_metrics = metrics
        self.train_metrics = nn.ModuleDict({metric_name: METRIC_REGISTRY[metric_name]() for metric_name in self.base_metrics})
        self.val_metrics = nn.ModuleDict({metric_name: METRIC_REGISTRY[metric_name]() for metric_name in self.base_metrics})
        self.test_metrics = nn.ModuleDict({metric_name: METRIC_REGISTRY[metric_name]() for metric_name in self.base_metrics})

    def forward(self, 
                img: torch.Tensor,
                paths: list[torch.Tensor]
    ) -> torch.Tensor:
        paths_logits = []
        for path in paths:
            path = path.squeeze(0)  # shape (path_length, 2)
            dist = torch.linalg.norm(path[0] - path[-1])
            paths_logits.append(dist < self.max_dist)
            
        paths_logits = torch.stack(paths_logits).squeeze()  # shape (num_edges,)

        # Return path logits and None for the feature map (since this model does not use it)
        return paths_logits, None