import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Works for any spatial dimensionality (2D: [B, 1, H, W], 3D: [B, 1, D, H, W], ...)
    since intersection/union are computed by flattening every dimension after batch.

    Args:
        smooth: Additive smoothing term in numerator and denominator. Prevents
            division by zero when both prediction and target are empty (all-background
            sample), and stabilizes gradients when overlap is small.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: predicted probabilities in [0, 1], shape [B, 1, *spatial_dims].
                Expected to already be post-sigmoid (or post-softmax for the
                foreground channel), NOT raw logits.
            targets: binary ground truth, same shape as probs, values in {0, 1}.

        Returns:
            Scalar loss: 1 - mean Dice coefficient over the batch.
        """
        if probs.shape != targets.shape:
            raise ValueError(
                f"probs and targets must have the same shape, got {probs.shape} and {targets.shape}"
            )

        batch_size = probs.size(0)
        probs_flat = probs.reshape(batch_size, -1)
        targets_flat = targets.reshape(batch_size, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_score.mean()