from skimage.measure import label
import torch
from torchmetrics import Metric
import warnings
import numpy as np

class Betti_error_1_2D(Metric):
    """
    Computes the 1st Betti number error for evaluating segmentation quality of 2D binary masks.
    1st Betti number corresponds to the number of holes in the segmentation.
    1st Betti number error is defined as: | Betti_1(pred) - Betti_1(gt) | / (Betti_1(gt) + min_f)
    """
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    min_f: float = 1e-6

    def __init__(self) -> None:
        super().__init__()

        self.accum: torch.Tensor
        self.total: torch.Tensor
        
        self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')

    def betti_1_2D(self, mask: torch.Tensor) -> int:
        '''
        Compute the 1st Betti number (number of holes) for a 2D binary mask.

        Parameters:
            mask (torch.Tensor): 2D binary mask where foreground pixels are True.

        Returns:
            int: The 1st Betti number (number of holes).
        '''
        if mask.ndim != 2:
            raise ValueError("Input mask must be a 2D array.")
        if mask.dtype != torch.bool:
            mask = (mask > 0)

        # Invert the mask to find holes
        inverse_mask = torch.logical_not(mask)
        # Label connected components in the inverted mask, using 4-connectivity
        _, num_cc = label(inverse_mask.cpu().numpy(), return_num=True, connectivity=1)
        # Subtract 1 to exclude the outer background component
        return num_cc - 1

    def get_score(self,
                  pred: torch.Tensor,
                  target: torch.Tensor
    ) -> float:
        betti_1_number_pred = self.betti_1_2D(pred)
        betti_1_number_gt = self.betti_1_2D(target)
        print("betti_1_number_pred:", betti_1_number_pred, ", betti_1_number_gt:", betti_1_number_gt)

        betti_error_1 = abs(betti_1_number_pred - betti_1_number_gt) / (betti_1_number_gt + self.min_f)
        print("betti_error_1:", betti_error_1)
        return betti_error_1

    def update(
        self,
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> None:
        betti_error_1 = self.get_score(pred, target)

        self.accum += betti_error_1
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