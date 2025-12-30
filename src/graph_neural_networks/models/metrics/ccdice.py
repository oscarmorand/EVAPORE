from torchmetrics import Metric
import torch
import numpy as np
from skimage.measure import label

class CCDice(Metric):
    """
    Computes the Connected components Dice (CCDice)
    """
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    min_f: float = 1e-6

    def __init__(self, alpha=0.5) -> None:
        """
        Initializes the CCDice metric.
        """
        super().__init__()

        self.accum: torch.Tensor
        self.total: torch.Tensor
        
        self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')

        self.alpha = alpha

    def S(self, y1, y2):
        return np.sum(y1 * y2) / np.sum(y1)

    def get_score(self,
                  pred: torch.Tensor,
                  target: torch.Tensor
    ) -> float:
        y_pred = pred.detach().cpu().numpy().astype(np.bool)
        y_true = target.detach().cpu().numpy().astype(np.bool)
        y_pred_label, cc_pred = label(y_pred, return_num=True)
        y_true_label, cc_true = label(y_true, return_num=True)
        
        y_true_label[y_true_label != 0] = y_true_label[y_true_label != 0] + cc_pred

        list_s = []
        indices_cc = []
        for a in range(1, cc_pred + 1):
            for b in range(cc_pred + 1, cc_pred + cc_true + 1):
                
                y1 = np.zeros(y_pred_label.shape)
                y1[y_pred_label == a] = 1
                
                y2 = np.zeros(y_true_label.shape)
                y2[y_true_label == b] = 1
                
                s_ab = self.S(y1, y2)
                s_ba = self.S(y2, y1)
                
                list_s.append(s_ab)
                list_s.append(s_ba)
                
                indices_cc.append((a, b))
                indices_cc.append((b, a))
            
        if self.alpha <= 0.5:
            # Sort the list
            list_s = np.array(list_s)
            indices = np.argsort(-list_s)
            indices_cc = np.array(indices_cc)
            
            list_s = np.array(list_s)
            list_s = list_s[indices]
            indices_cc = indices_cc[indices]
        
        left_list = []
        right_list = []
        tp = 0
        i = 0
        s = list_s[0]
        coor = indices_cc[0]
        while s >= self.alpha and i < len(list_s):
            
            if (coor[0] not in left_list) and (coor[1] not in right_list):
            
                left_list.append(coor[0])
                right_list.append(coor[1])
                tp += 1
                
            i += 1
            if i < len(list_s):
                s = list_s[i]
                coor = indices_cc[i]
            
        ccdice = tp / (cc_pred + cc_true)
        
        return ccdice

    def update(
        self,
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> None:
        ccdice_score = self.get_score(pred, target)

        self.accum += ccdice_score
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