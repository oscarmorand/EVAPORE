from skimage.measure import label
import torch
from torchmetrics import Metric
import warnings
import numpy as np

class Betti_error_0_2D(Metric):
    """
    Computes the 0th Betti number error for evaluating segmentation quality of 2D binary masks.
    0th Betti number corresponds to the number of connected components in the segmentation.
    0th Betti number error is defined as: | Betti_0(pred) - Betti_0(gt) | / (Betti_0(gt) + min_f)
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

    def betti_0_2D(self, mask: torch.Tensor) -> int:
        '''
        Compute the 0th Betti number (number of connected components) for a 2D binary mask.

        Parameters:
            mask (torch.Tensor): 2D binary mask where foreground pixels are True.

        Returns:
            int: The 0th Betti number (number of connected components).
        '''
        if mask.ndim != 2:
            raise ValueError("Input mask must be a 2D array.")
        if mask.dtype != torch.bool:
            mask = (mask > 0)

        # Label connected components in the inverted mask, using 8-connectivity
        _, num_cc = label(mask.cpu().numpy(), return_num=True, connectivity=2)
        return num_cc

    def get_score(self,
                  pred: torch.Tensor,
                  target: torch.Tensor
    ) -> float:
        betti_0_number_pred = self.betti_0_2D(pred)
        betti_0_number_gt = self.betti_0_2D(target)
        print("betti_0_number_pred:", betti_0_number_pred, ", betti_0_number_gt:", betti_0_number_gt)

        betti_error_0 = abs(betti_0_number_pred - betti_0_number_gt) / (betti_0_number_gt + self.min_f)
        print("betti_error_0:", betti_error_0)
        return betti_error_0
    
    def update(
        self,
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> None:
        betti_error_0 = self.get_score(pred, target)

        self.accum += betti_error_0
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