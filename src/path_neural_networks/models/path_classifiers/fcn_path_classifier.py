import torch
import torch.nn as nn
from typing import Sequence

from path_neural_networks.models.path_classifiers import PathClassifier

class FCNPathClassifier(PathClassifier):
    def __init__(self,
                 in_channels: int,
                 n_hidden_layers: int,
                 num_classes: int,
                 dropout: Sequence | float | int = 0.0
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.n_hidden_layers = n_hidden_layers
        self.num_classes = num_classes

        if isinstance(dropout, (int, float)):
            dropout = [float(dropout)] * n_hidden_layers
        elif not isinstance(dropout, Sequence):
            raise TypeError("dropout must be float or sequence of floats")
        
        if len(dropout) != n_hidden_layers:
            raise ValueError(f"If enabled, Dropout must be applied to each of the {n_hidden_layers} hidden layers, got {len(dropout)} dropout layers")
        
        for d in dropout:
            if not 0.0 <= d < 1.0:
                raise ValueError(f"Dropout must be in [0, 1), dot dropout={d}")

        self.dropout = dropout

        layers = []
        prev_features = in_channels
        for d in dropout:
            next_features = max(prev_features // 2, 1)
            layers.append(nn.Linear(prev_features, next_features))
            layers.append(nn.LayerNorm(next_features))
            layers.append(nn.ReLU())

            if d > 0.0:
                layers.append(nn.Dropout(p=d))

            prev_features = next_features
            
        layers.append(nn.Linear(prev_features, num_classes))
        
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
    
    def as_dict(self):
        return {
            "cls": self.__class__.__name__,
            "in_channels": self.in_channels,
            "n_hidden_layers": self.n_hidden_layers,
            "dropout": self.dropout,
            "num_classes": self.num_classes
        }