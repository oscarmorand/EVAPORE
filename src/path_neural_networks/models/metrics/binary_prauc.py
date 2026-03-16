import torch
from torch import nn
from torchmetrics import Metric
from torchmetrics.functional import precision_recall_curve

class BinaryPRAUC(Metric):
    """
    Computes Precision-Recall AUC for binary classification,
    works like torchmetrics.BinaryAUROC
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # store predictions and targets
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: raw logits or probabilities (will convert to probs)
        targets: 0 or 1
        """
        preds = torch.sigmoid(preds) if preds.max() > 1 else preds
        self.preds.append(preds.detach().flatten())
        self.targets.append(targets.detach().flatten())

    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        precision, recall, _ = precision_recall_curve(preds, targets, task="binary")
        pr_auc = torch.trapz(precision.flip(0), recall.flip(0))
        return pr_auc