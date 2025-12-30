from torchmetrics import Metric
import torch
from skimage.morphology import skeletonize

class ClDice(Metric):
    """
    Computes the Centerline Dice (clDice) metric for evaluating segmentation quality of tubular structures.
    """
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    min_f: float = 1e-6

    def __init__(self) -> None:
        """
        Initializes the ClDice metric.
        """
        super().__init__()

        self.accum: torch.Tensor
        self.total: torch.Tensor
        
        self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')

    def cl_score(self, 
                 mask: torch.Tensor,
                 skel: torch.Tensor, 
    ) -> float:
        return torch.sum(mask*skel) / (torch.sum(skel) + self.min_f)

    def get_score(self,
                  pred: torch.Tensor,
                  target: torch.Tensor
    ) -> float:
        skel_gt = torch.tensor(skeletonize(target.numpy().astype(bool)))
        skel_pred = torch.tensor(skeletonize(pred.numpy().astype(bool)))
        tprec = self.cl_score(target, skel_pred)
        tsens = self.cl_score(pred, skel_gt)
        clidce_score = (2 * tprec * tsens) / (tprec + tsens + self.min_f)
        return clidce_score

    def update(
        self,
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> None:
        cldice_score = self.get_score(pred, target)

        self.accum += cldice_score
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