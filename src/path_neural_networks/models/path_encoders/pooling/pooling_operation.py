import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class PoolingOperation(nn.Module, ABC):
    def __init__(self):
        super(PoolingOperation, self).__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("PoolingOperation is an abstract class and cannot be instantiated directly.")