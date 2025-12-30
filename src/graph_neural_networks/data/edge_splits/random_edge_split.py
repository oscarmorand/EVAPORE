import torch
import numpy as np
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data

from graph_neural_networks.data.edge_splits.edge_split import EdgeSplit
from graph_neural_networks.data.data_targets.data_target import DataTarget

class RandomEdgeSplit(EdgeSplit):
    def __init__(self, remove_ratio: float = 0.1, 
                 negative_sampling_ratio: float = 1.0
    ) -> None:
        super().__init__()
        self.remove_ratio = remove_ratio
        self.negative_sampling_ratio = negative_sampling_ratio

    def __call__(self, data):
        self.check_basic_data_attrs(data)

        transform = RandomLinkSplit(
            num_val=self.remove_ratio,
            num_test=0.0,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=self.negative_sampling_ratio,
        )
        
        _, graph, _ = transform(data)
        
        return graph