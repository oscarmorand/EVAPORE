import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from abc import ABC

class EdgeQuery(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError
        
class EdgeQueryAllNonExisting(EdgeQuery):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_index = batch.edge_index
        num_nodes = batch.num_nodes
        full_edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        existing_edges_set = set((u.item(), v.item()) for u, v in edge_index.t())
        non_existing_edges = [
            (u.item(), v.item()) for u, v in full_edge_index.t()
            if (u.item(), v.item()) not in existing_edges_set and (v.item(), u.item()) not in existing_edges_set
        ]
        if len(non_existing_edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(non_existing_edges, dtype=torch.long).t()
        return edge_index_query


class EdgeQueryMaxDistance(EdgeQuery):
    def __init__(self, max_dist: float) -> None:
        super().__init__()
        self.max_dist = max_dist

    def forward(self, batch: Batch) -> torch.Tensor:
        nodes = torch.arange(batch.num_nodes)
        edge_index = batch.edge_index
        existing_edges = set((u.item(), v.item()) for u, v in edge_index.t())
        edge_index_query = []
        for n1 in nodes:
            for n2 in nodes:
                if n1 >= n2:
                    continue
                if (n1.item(), n2.item()) in existing_edges or (n2.item(), n1.item()) in existing_edges:
                    continue
                pos1 = batch.pos[n1]
                pos2 = batch.pos[n2]
                distance = torch.norm(pos1 - pos2)
                if distance <= self.max_dist:
                    edge_index_query.append((n1.item(), n2.item()))

        if len(edge_index_query) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(edge_index_query, dtype=torch.long).t()
        return edge_index_query
    

class EdgeQueryExtremitiesMaxDistance(EdgeQuery):
    def __init__(self, max_dist: float) -> None:
        super().__init__()
        self.max_dist = max_dist

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_index = batch.edge_index
        nodes = torch.arange(batch.num_nodes)
        existing_edges = set((u.item(), v.item()) for u, v in edge_index.t())

        node_degrees = torch.zeros(batch.num_nodes, dtype=torch.long)
        for u, v in edge_index.t():
            node_degrees[u] += 1
            node_degrees[v] += 1
        node_degrees = node_degrees / 2 # undirected graph
        extremity_nodes = nodes[node_degrees <= 1]

        edge_index_query = []
        for extremity_node in extremity_nodes:
            for n in nodes:
                if extremity_node == n:
                    continue
                if (extremity_node.item(), n.item()) in existing_edges or (n.item(), extremity_node.item()) in existing_edges:
                    continue
                pos1 = batch.pos[extremity_node]
                pos2 = batch.pos[n]
                distance = torch.norm(pos1 - pos2)
                if distance <= self.max_dist:
                    edge_index_query.append((extremity_node.item(), n.item()))

        if len(edge_index_query) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(edge_index_query, dtype=torch.long).t()
        return edge_index_query
    

class EdgeQueryDiffCCMaxDistance(EdgeQuery):
    def __init__(self, max_dist: float) -> None:
        super().__init__()
        self.max_dist = max_dist

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_index = batch.edge_index
        nodes = torch.arange(batch.num_nodes)

        cc_labels = []
        visited = set()
        queue = []
        for node in nodes:
            if node.item() in visited:
                continue
            queue.append(node.item())
            current_cc = []
            while queue:
                current_node = queue.pop(0)
                if current_node in visited:
                    continue
                visited.add(current_node)
                current_cc.append(current_node)
                neighbors = edge_index[1][edge_index[0] == current_node].tolist() + edge_index[0][edge_index[1] == current_node].tolist()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
            cc_labels.append(current_cc)

        # Find pairs of nodes in different connected components
        edge_index_query = []
        for i, cc1 in enumerate(cc_labels):
            for j, cc2 in enumerate(cc_labels):
                if i >= j:
                    continue
                for n1 in cc1:
                    for n2 in cc2:
                        pos1 = batch.pos[n1]
                        pos2 = batch.pos[n2]
                        distance = torch.norm(pos1 - pos2)
                        if distance <= self.max_dist:
                            edge_index_query.append((n1, n2))

        if len(edge_index_query) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(edge_index_query, dtype=torch.long).t()
        return edge_index_query


class EdgeQueryExtremitiesDiffCCMaxDistance(EdgeQuery):
    def __init__(self, max_dist: float) -> None:
        super().__init__()
        self.max_dist = max_dist

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_index = batch.edge_index
        edge_label_index = batch.edge_label_index
        nodes = torch.arange(batch.num_nodes)

        # Reconstruct adjacency matrix
        adj_matrix = torch.zeros((batch.num_nodes, batch.num_nodes), dtype=torch.bool)
        for u, v in edge_label_index.t():
            adj_matrix[u, v] = True
            adj_matrix[v, u] = True  # undirected graph

        # Find connected components
        cc_labels = []
        visited = set()
        queue = []
        for node in nodes:
            if node.item() in visited:
                continue
            queue.append(node.item())
            current_cc = []
            while queue:
                current_node = queue.pop(0)
                if current_node in visited:
                    continue
                visited.add(current_node)
                current_cc.append(current_node)
                neighbors = adj_matrix[current_node].nonzero(as_tuple=False).view(-1).tolist()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
            cc_labels.append(current_cc)

        # Find extremity nodes
        node_degrees = torch.zeros(batch.num_nodes, dtype=torch.long)
        for u, v in edge_index.t():
            node_degrees[u] += 1
            node_degrees[v] += 1
        if (node_degrees == 1).sum() == 0:
            node_degrees = node_degrees / 2 # undirected graph
        extremity_nodes = nodes[node_degrees <= 1]

        # Find pairs nodes (extremity, other) in different connected components
        edge_index_query = set()
        for extremity_node in extremity_nodes:
            extremity_cc = None
            for i, cc in enumerate(cc_labels):
                if extremity_node.item() in cc:
                    extremity_cc = i
                    break
            for j, cc in enumerate(cc_labels):
                if j == extremity_cc:
                    continue
                for n in cc:
                    if extremity_node == n:
                        continue
                    pos1 = batch.pos[extremity_node]
                    pos2 = batch.pos[n]
                    distance = torch.norm(pos1 - pos2)
                    if distance <= self.max_dist:
                        edge_index_query.add((min(extremity_node.item(), n), max(extremity_node.item(), n)))

        if len(edge_index_query) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(list(edge_index_query), dtype=torch.long).t()
        return edge_index_query

class EdgeQueryVirtualEdges(EdgeQuery):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_index = batch.edge_index
        edge_label_index = batch.edge_label_index
        existing_edges_set = set((u.item(), v.item()) for u, v in edge_index.t())
        real_edges_set = set((u.item(), v.item()) for u, v in edge_label_index.t())
        virtual_edges = [
            (u, v) for u, v in existing_edges_set
            if (u, v) not in real_edges_set and (v, u) not in real_edges_set
        ]
        if len(virtual_edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_query = torch.tensor(virtual_edges, dtype=torch.long).t()
        return edge_index_query