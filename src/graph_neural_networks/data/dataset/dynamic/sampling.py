import torch
import math

# ===============================================================
# AUXILIARY FUNCTIONS
# ===============================================================

def get_radius_square(yx_coords: tuple[int, int], 
                      radius: int | float,
                      dim: tuple[int, int]) -> int:
    if isinstance(radius, float):
        radius = math.ceil(radius)
    h, w = dim
    y, x = yx_coords
    if x < 0 or x >= w or y < 0 or y >= h:
        raise ValueError(f"Coordinates {yx_coords} are out of bounds for feature map with dimensions {dim}.")
    x_min = max(x - radius, 0)
    x_max = min(x + radius + 1, w)
    y_min = max(y - radius, 0)
    y_max = min(y + radius + 1, h)
    return x_min, x_max, y_min, y_max


def coordinates_rescale(yx_coords: tuple[int, int],
                        original_shape: tuple[int, int],
                        target_shape: tuple[int, int]
) -> tuple[int, int]:
    orig_h, orig_w = original_shape
    target_h, target_w = target_shape
    y, x = yx_coords
    x_rescaled = int(x * target_w / orig_w)
    y_rescaled = int(y * target_h / orig_h)
    return (y_rescaled, x_rescaled)


def radius_rescale(radius: int | float,
                     original_shape: tuple[int, int],
                     target_shape: tuple[int, int]
) -> int | float:
    orig_h, orig_w = original_shape
    target_h, target_w = target_shape
    scale_factor = (target_h / orig_h + target_w / orig_w) / 2.0
    if isinstance(radius, float):
        return radius * scale_factor
    else:
        return int(radius * scale_factor)

# ===============================================================
# SAMPLING METHODS IMPLEMENTATIONS
# ===============================================================

# Node sampling methods

def node_single_pixel_sampling(feature_map: torch.Tensor,
                               yx_coords: tuple[int, int],
                               **kwargs
) -> torch.Tensor:
    y, x = yx_coords
    return feature_map[:, y, x]

def node_radius_square_sampling(feature_map: torch.Tensor, 
                           yx_coords: tuple[int, int], 
                           radius: int | float,
                           **kwargs
) -> torch.Tensor:
    x_min, x_max, y_min, y_max = get_radius_square(yx_coords, radius, (feature_map.shape[1], feature_map.shape[2]))
    return feature_map[:, y_min:y_max, x_min:x_max]

def node_radius_double_square_sampling(feature_map: torch.Tensor,
                                yx_coords: tuple[int, int],
                                radius: int | float,
                                **kwargs
) -> torch.Tensor:
    return node_radius_square_sampling(feature_map, yx_coords, radius * 2, **kwargs)


# Edge sampling methods

def edge_single_pixel_sampling(feature_map: torch.Tensor,
                               centerline: list[tuple[int, int]],
                               **kwargs
) -> torch.Tensor:
    if not centerline:
        return None
    centerline = torch.tensor(centerline).t()  # shape (2, N)
    sampled_x, sampled_y = centerline[0], centerline[1]
    return feature_map[:, sampled_y, sampled_x]

def edge_radius_square_sampling(feature_map: torch.Tensor,
                                centerline: list[tuple[int, int]],
                                radius: list[int | float],
                                **kwargs
) -> torch.Tensor:

    if radius is None or len(radius) == 0 or centerline is None or len(centerline) == 0:
        return None
    if len(centerline) != len(radius):
        raise ValueError("Length of centerline and radius must be the same.")
    sampled_coords = set()
    for (x, y), r in zip(centerline, radius):
        x_min, x_max, y_min, y_max = get_radius_square((x, y), r, (feature_map.shape[1], feature_map.shape[2]))
        for yi in range(y_min, y_max):
            for xi in range(x_min, x_max):
                sampled_coords.add((xi, yi))
    sampled_coords = torch.tensor(list(sampled_coords)).t()  # shape (2, N)
    sampled_x, sampled_y = sampled_coords[0], sampled_coords[1]
    return feature_map[:, sampled_y, sampled_x]


# ===============================================================
# SAMPLING METHODS DICTIONARIES
# ===============================================================

node_sampling_methods_dict = {
    "radius_square": node_radius_square_sampling,
    "double_radius_square": node_radius_double_square_sampling,
    "single_pixel": node_single_pixel_sampling,
}
edge_sampling_methods_dict = {
    "radius_square": edge_radius_square_sampling,
    "single_pixel": edge_single_pixel_sampling,
}


# ===============================================================
# SAMPLING FUNCTIONS
# ===============================================================

def sample_node_features(sampling_method: str,
                         feature_map: torch.Tensor,
                         yx_coords: tuple[int, int],
                         radius: int | float,
                         base_image_shape: tuple[int, int]) -> torch.Tensor:
    if sampling_method not in node_sampling_methods_dict:
        raise ValueError(f"Sampling method {sampling_method} not recognized.")
    
    features_shape = feature_map.shape[1:]  # (H, W)
    if features_shape != base_image_shape:
        yx_coords = coordinates_rescale(yx_coords, base_image_shape, features_shape)
        radius = radius_rescale(radius, base_image_shape, features_shape)

    sample_func = node_sampling_methods_dict[sampling_method]
    return sample_func(feature_map, yx_coords, radius=radius)

def sample_edge_features(sampling_method: str,
                         feature_map: torch.Tensor,
                         centerline: list[tuple[int, int]],
                         radius: list[int | float],
                         base_image_shape: tuple[int, int]) -> torch.Tensor:
    raise NotImplementedError("Edge feature sampling is not yet implemented.")