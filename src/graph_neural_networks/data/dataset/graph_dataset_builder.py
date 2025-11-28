from abc import ABC
import os
import torch
import copy
from torch_geometric.io import fs
from torch_geometric.data import Data
from tqdm import tqdm
import networkx as nx

from graph_neural_networks.data.dataset.graph_dataset import GraphDataset
from graph_neural_networks.data.utils.io import json_to_networkx
from graph_neural_networks.utils import RankedLogger
from graph_neural_networks.data.dataset.config_manager import ConfigManager
from graph_neural_networks.data.dataset.dynamic.aggregation import aggregate_node_features, aggregate_edge_features
from graph_neural_networks.data.dataset.dynamic.handcrafted import handcraft_node_feature
from graph_neural_networks.data.dataset.dynamic.sampling import sample_node_features, sample_edge_features
from graph_neural_networks.data.dataset.dynamic.graph_transforms.graph_transform import GraphTransform
from graph_neural_networks.data.dataset.dynamic.apply_graph_transforms import apply_graph_transforms
from graph_neural_networks.data.dataset.graph_wrapper import GraphWrapper
from graph_neural_networks.data.utils.networkx import networkx_to_pyg
from graph_neural_networks.data.utils.pred_state import EdgePredState

log = RankedLogger(__name__, rank_zero_only=True)


class GraphDatasetBuilder(ABC):
    def __init__(self, base_dataset: GraphDataset | None = None) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.base_image_shape = (base_dataset.height, base_dataset.width)

    def get_dataset(self) -> GraphDataset:
        raise NotImplementedError


class ClassicDataset(GraphDatasetBuilder):
    def __init__(self, 
                 base_dataset: GraphDataset,
    ) -> None:
        super().__init__(base_dataset)
    
    def get_dataset(self) -> GraphDataset:
        return self.base_dataset


class DynamicDataset(GraphDatasetBuilder):
    def __init__(self,
                 base_dataset: GraphDataset,
                 handcrafted: dict,
                 use_extracted: bool,
                 extracted: dict,
                 force_recompute: bool,
                 graph_transforms: dict[str, GraphTransform] = {}) -> None:
        super().__init__(base_dataset)

        self.handcrafted = handcrafted
        self.use_extracted = use_extracted
        self.extracted = extracted
        self.graph_transforms = graph_transforms
        self.force_recompute = force_recompute

        self.config_manager = ConfigManager(self.base_dataset.processed_dir)

    def build_config(self) -> dict:
        cfg = {
            "base_dataset": {
                "name":self.base_dataset.__class__.__name__,
                "raw_dir_name": self.base_dataset.raw_dir_name,
            },
            "handcrafted": {
                "node_features": list(self.handcrafted.get('node_features')),
                "edge_features": list(self.handcrafted.get('edge_features')),
            },
            "use_extracted": self.use_extracted,
            "extracted": dict(self.extracted),
            "graph_transforms": [transform._build_config() for key, transform in self.graph_transforms.items()],
        }
        return cfg

    def compute_graph_full_features(self, 
                                     graph_extracted_features: torch.Tensor | None,
                                     graph_handcrafted_features: torch.Tensor | None,
                                     use_handcrafted: bool):
        if self.use_extracted:
            if use_handcrafted:
                graph_full_features = torch.cat([graph_extracted_features, graph_handcrafted_features], dim=-1)
            else:
                graph_full_features = graph_extracted_features
        else:
            if use_handcrafted:
                graph_full_features = graph_handcrafted_features
            else:
                raise ValueError("No features selected. Please select at least one type of features (extracted or handcrafted).")
        return graph_full_features


    def save(self, 
             datalist: list[Data], 
             path: str, 
             cfg: dict
    ) -> None:
        log.info(f"Saving processed DynamicDataset at {path}...")
        os.makedirs(path, exist_ok=True)

        log.info("Saving single config...")
        self.config_manager.save_single_config(cfg, path)

        for i, data in enumerate(datalist):
            processed_name = (self.base_dataset.raw_file_names[i]).split(".")[0]
            processed_path = os.path.join(path, processed_name + ".pt")
            fs.torch_save(data, processed_path)
        log.info("Processed DynamicDataset saved.")


    def compute_edge_index(self, 
                           nx_graph: nx.Graph
    ) -> torch.Tensor:
        edge_index = []
        for u, v, data in nx_graph.edges(data=True):
            pred_state = data.get('edge_pred_state', None)
            if pred_state is None or pred_state in [EdgePredState.IN_PREDICTION, EdgePredState.IN_PREDICTION.value]:
                edge_index.append([u, v])
                edge_index.append([v, u])  # undirected graph
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index_tensor
    
    def compute_virtual_edge_index(self,
                                   nx_graph: nx.Graph
    ) -> torch.Tensor:
        virtual_edge_index = []
        for u, v, data in nx_graph.edges(data=True):
            if data.get('virtual_edge', False):
                virtual_edge_index.append([u, v])
                virtual_edge_index.append([v, u])  # undirected graph
        virtual_edge_index_tensor = torch.tensor(virtual_edge_index, dtype=torch.long).t().contiguous()
        return virtual_edge_index_tensor


    def compute_datalist(self) -> list[Data]:
        log.info("Computing DynamicDataset...")

        node_features, edge_features = self.handcrafted.get('node_features'), self.handcrafted.get('edge_features')
        self.use_handcrafted_node_features = node_features and len(node_features) > 0
        self.use_handcrafted_edge_features = edge_features and len(edge_features) > 0
        backbone, aggregation, sampling = self.extracted.get('backbone'), self.extracted.get('aggregation'), self.extracted.get('sampling')

        datalist = []

        if self.use_extracted:
            data_dir = self.base_dataset.root
            extracted_features_dir = os.path.join(data_dir, "feature_maps", backbone)
            extracted_features_names = os.listdir(extracted_features_dir)
            extracted_features_names.sort()
            feature_maps_path = [os.path.join(extracted_features_dir, f) for f in extracted_features_names if f.endswith('.pt')]
            assert len(feature_maps_path) == len(self.base_dataset), "Number of feature maps must be equal to number of graphs in base dataset"

        for graph_id, (raw_path) in enumerate(tqdm(self.base_dataset.raw_paths)):
            nx_graph = json_to_networkx(raw_path, False, {})
            graph_wrapper = GraphWrapper(nx_graph)

            # Apply graph transforms
            graph_wrapper = apply_graph_transforms(graph_wrapper, self.graph_transforms)
            nx_graph = graph_wrapper.get_graph()

            keys = None
            for u, v, data in nx_graph.edges(data=True):
                if keys is None:
                    keys = data.keys()
                for key in keys:
                   if key not in data:
                       print(f"Warning: Edge ({u}, {v}) is missing key '{key}'. Filling with default value 0.")
                for key in data.keys():
                    if key not in keys:
                        print(f"Warning: Edge ({u}, {v}) has an unexpected key '{key}'.")

            pyg_graph = networkx_to_pyg(nx_graph, None, None, group_node_attrs=[], group_edge_attrs=[])

            if self.use_extracted:
                feature_map_path = feature_maps_path[graph_id]
                feature_map = torch.load(feature_map_path, map_location='cpu', weights_only=False) # (e_C, H, W)
                if torch.isnan(feature_map).any():
                    raise ValueError(f"NaN values found in feature map for graph id {graph_id} at path {feature_map_path}.")

            # Initialize graph features tensors
            N = nx_graph.number_of_nodes()
            E = nx_graph.number_of_edges()
            e_C = feature_map.shape[0] if self.use_extracted else 0
            h_C = len(node_features) if self.use_handcrafted_node_features else 0
            graph_extracted_node_features = torch.zeros((N, e_C)) if self.use_extracted else None
            graph_handcrafted_node_features = torch.zeros((N, h_C)) if self.use_handcrafted_node_features else None
            graph_extracted_edge_features = torch.zeros((E, e_C)) if self.use_extracted else None
            graph_handcrafted_edge_features = torch.zeros((E, h_C)) if self.use_handcrafted_edge_features else None

            # Populate node features
            nodes_dict = nx_graph.nodes(data=True)
            for node_id, (node_idx, attrs) in enumerate(nodes_dict):
                yx_coords, radius = tuple(attrs.get('pos')), attrs.get('radius')

                # Extracted features
                if self.use_extracted:
                    sampled_feature_map = sample_node_features(sampling, feature_map, yx_coords, radius, self.base_image_shape)  # (e_C, h, w)
                    node_aggregated_features = aggregate_node_features(aggregation, sampled_feature_map) # (e_C,)
                    graph_extracted_node_features[node_id] = node_aggregated_features

                # Handcrafted node features
                if self.use_handcrafted_node_features:
                    for k, node_feature in enumerate(node_features):
                        node_extracted_feature = handcraft_node_feature(node_feature, nx_graph, node_idx, base_image_shape=self.base_image_shape) # scalar
                        graph_handcrafted_node_features[node_id, k] = node_extracted_feature

            # Populate edge features
            for src, dst, attrs in nx_graph.edges(data=True):
                pass # TODO: implement edge features computation

            # Combine extracted and handcrafted node features
            graph_node_features  = self.compute_graph_full_features(graph_extracted_node_features, graph_handcrafted_node_features, self.use_handcrafted_node_features)
            graph_edge_features = self.compute_graph_full_features(graph_extracted_edge_features, graph_handcrafted_edge_features, self.use_handcrafted_edge_features)
            
            new_graph = pyg_graph.clone()
            new_graph.x = graph_node_features
            #new_graph.edge_attr = graph_edge_features             

            edge_index = self.compute_edge_index(nx_graph)
            new_graph.edge_index = edge_index

            virtual_edge_index = self.compute_virtual_edge_index(nx_graph)
            new_graph.virtual_edge_index = virtual_edge_index

            print(new_graph)
            print(new_graph.x)
            print(new_graph.edge_index)
            print(new_graph.virtual_edge_index)

            datalist.append(new_graph)
            break

        log.info("DynamicDataset computed.")

        return datalist


    def get_dataset(self) -> GraphDataset:
        cfg = self.build_config()
        print(cfg)

        dataset = copy.copy(self.base_dataset)
        dataset.classic_dataset = False

        exists, existing_path = self.config_manager.already_exists(cfg)
        if exists and not self.force_recompute:
            log.info(f"DynamicDataset with the same configuration already exists at {existing_path}. Loading existing dataset...")
            dataset.dynamic_dir = existing_path
        else:
            datalist = self.compute_datalist()
            path = existing_path
            if exists and self.force_recompute:
                log.info(f"Force recompute is enabled. Recomputing the DynamicDataset even if it already exists at {existing_path}.")
            else:
                path = self.config_manager.add_config(cfg)
            self.save(datalist, path, cfg)

            dataset.dynamic_dir = path

        return dataset
