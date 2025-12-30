from torchmetrics import Metric
import torch

class BinaryDice(Metric):
    """
    Computes the Dice metric for evaluating binary segmentation quality.
    """
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    min_f: float = 1e-6

    def __init__(self) -> None:
        """
        Initializes the Dice metric.
        """
        super().__init__()

        self.accum: torch.Tensor
        self.total: torch.Tensor
        
        self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')

    def get_score(self,
                  pred: torch.Tensor,
                  target: torch.Tensor
    ) -> float:
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        clidce_score = (2 * intersection + self.min_f) / (union + self.min_f)
        return clidce_score

    def update(
        self,
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> None:
        clidce_score = self.get_score(pred, target)

        self.accum += clidce_score
        self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.accum)
        return self.accum / self.total

    def reset(self) -> None:
        r"""Resets metric state variables to their default value."""
        super().reset()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'