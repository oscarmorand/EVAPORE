from abc import ABC
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper

class GraphTransform(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, graph_wrapper: GraphWrapper) -> GraphWrapper:
        raise NotImplementedError

    def _build_config(self) -> dict:
        raise NotImplementedError