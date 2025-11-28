import torch

# ===============================================================
# AGGREGATION METHODS DICTIONARIES
# ===============================================================

node_aggregation_methods_dict = {
    "mean": torch.mean,
    "max": torch.amax,
    "min": torch.amin,
    "sum": torch.sum,
}

edge_aggregation_methods_dict = {
    "mean": torch.mean,
    "max": torch.amax,
    "min": torch.amin,
    "sum": torch.sum,
}

# ===============================================================
# AGGREGATION FUNCTIONS
# ===============================================================

def aggregate_node_features(aggregation_method: str,
                            sampled_features: torch.Tensor) -> torch.Tensor:
    if aggregation_method not in node_aggregation_methods_dict:
        raise ValueError(f"Aggregation method {aggregation_method} not recognized.")
    agg_func = node_aggregation_methods_dict[aggregation_method]
    return agg_func(sampled_features, dim=(-2, -1))  # aggregate over H and W

def aggregate_edge_features(aggregation_method: str,
                            sampled_features: torch.Tensor) -> torch.Tensor:
    if aggregation_method not in edge_aggregation_methods_dict:
        raise ValueError(f"Aggregation method {aggregation_method} not recognized.")
    agg_func = edge_aggregation_methods_dict[aggregation_method]
    return agg_func(sampled_features, dim=(-2, -1))  # aggregate over H and W
