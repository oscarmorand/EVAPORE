import heapq
import networkx as nx
import numpy as np
import warnings

from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper

class ComputeDistanceMatrixTransform:
    def __init__(self):
        super().__init__()

    def compute_distance_matrix_by_graph_breadth_first(self, graph: nx.Graph) -> np.ndarray:
        N = graph.number_of_nodes()
        distance_matrix = np.ones((N, N)) * -1
        nodes_to_index = {n: i for i, n in enumerate(graph.nodes())}

        # Use Dijkstra for each start node
        for start_node in graph.nodes():
            start_index = nodes_to_index[start_node]

            # Min heap for priority queue
            pq = [(0.0, start_node)]
            distances = {n: float('inf') for n in graph.nodes()}
            distances[start_node] = 0.0

            while pq:
                current_dist, current_node = heapq.heappop(pq)
                current_index = nodes_to_index[current_node]

                # If already found a shorter path skip
                if current_dist > distances[current_node]:
                    continue

                # Write symmetric distances
                distance_matrix[start_index, current_index] = current_dist
                distance_matrix[current_index, start_index] = current_dist

                for neighbor in graph.neighbors(current_node):
                    edge_data = graph.get_edge_data(current_node, neighbor)

                    # Find shortest edge between the two nodes
                    min_length = min(d['length'] for d in edge_data.values())
                    new_dist = current_dist + min_length

                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))

        return distance_matrix


    def __call__(self, graph_wrapper: GraphWrapper) -> GraphWrapper:
        graph = graph_wrapper.get_graph()

        if graph_wrapper.distance_matrix is None:
            if graph_wrapper.is_oversampled:
                raise RuntimeError("Distance matrix computation should be done before graph node oversampling.")
            distance_matrix = self.compute_distance_matrix_by_graph_breadth_first(graph)
            graph_wrapper.distance_matrix = distance_matrix
        else:
            warnings.warn("GraphWrapper already has a distance matrix. Skipping computation.")

        return graph_wrapper

    def _build_config(self) -> dict:
        return {
            "_target_": self.__class__.__name__,
        }