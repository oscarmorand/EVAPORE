import torch
from abc import ABC

class GaussianNormalizedDistance(torch.nn.Module):
    def __init__(self, sigma: float = 100) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(- (distances ** 2) / (2 * self.sigma ** 2))

class DistanceRegressionLoss(torch.nn.Module, ABC):
    """Custom Loss for distance regression tasks."""

    def __init__(self, distance_to_y_fn, loss_fn) -> None:
        super().__init__()
        self.distance_to_y_fn = distance_to_y_fn
        self.loss_fn = loss_fn

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_distances_to_y = self.distance_to_y_fn(targets)
        return self.loss_fn(predictions, target_distances_to_y)
    

class GaussianNormalizedDistanceL1Loss(DistanceRegressionLoss):
    """L1 Loss for Gaussian Normalized Distance predictions."""

    def __init__(self, sigma: float = 100) -> None:
        distance_to_y_fn = GaussianNormalizedDistance(sigma)
        loss_fn = torch.nn.L1Loss()
        super().__init__(distance_to_y_fn, loss_fn)

class GaussianNormalizedDistanceBCELoss(DistanceRegressionLoss):
    """BCE Loss for Gaussian Normalized Distance predictions."""

    def __init__(self, sigma: float = 100) -> None:
        distance_to_y_fn = GaussianNormalizedDistance(sigma)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        super().__init__(distance_to_y_fn, loss_fn)