import torch
import numpy as np
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data

from graph_neural_networks.data.edge_splits.edge_split import EdgeSplit
from graph_neural_networks.data.data_targets.data_target import DataTarget

class NoEdgeSplit(EdgeSplit):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        self.check_basic_data_attrs(data)
        graph = Data(
            x = data.x,
            edge_index = data.edge_index,
            edge_label_index = data.in_pred_edge_index,
            pos = data.pos,
            graph_id = data.graph_id,
        )
        return graph