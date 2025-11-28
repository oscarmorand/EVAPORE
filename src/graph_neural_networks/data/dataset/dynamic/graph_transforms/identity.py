import networkx as nx
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper

class IdentityTransform:
    def __init__(self):
        pass

    def __call__(self, graph_wrapper: GraphWrapper) -> GraphWrapper:
        return graph_wrapper

    def _build_config(self) -> dict:
        return {
            "_target_": self.__class__.__name__,
        }