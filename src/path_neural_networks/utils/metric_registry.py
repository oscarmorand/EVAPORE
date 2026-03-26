from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision

from path_neural_networks.models.metrics.binary_prauc import BinaryPRAUC

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