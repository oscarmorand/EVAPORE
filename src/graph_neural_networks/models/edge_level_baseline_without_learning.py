import torch

class LinkPredictionBaselineWithoutLearning:
    def __init__(self) -> None:
        pass
    
    def predict_step(self, batch):
        """Perform a prediction step on a batch of data.

        Returns:
            A tuple containing the predicted new edges and their scores.
        """
        all_edges_set = set([tuple(edge) for edge in batch.edge_index.t().tolist()])
        real_edges_set = set([tuple(edge) for edge in batch.edge_label_index.t().tolist()])
        new_edges = set()
        for u, v in all_edges_set:
            if not ((u, v) in real_edges_set or (v, u) in real_edges_set):
                new_edges.add((u, v))
        new_edges = torch.tensor(list(new_edges)).t()
        print(new_edges.shape)
        return new_edges, None