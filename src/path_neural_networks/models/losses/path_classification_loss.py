import torch
import torch.nn as nn

class PathClassificationLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def as_dict(self):
        raise NotImplementedError