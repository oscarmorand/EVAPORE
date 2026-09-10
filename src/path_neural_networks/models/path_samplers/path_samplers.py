from path_neural_networks.models.path_samplers.path_sampler import PathSampler
from path_neural_networks.models.path_samplers.single_point_path_sampler import SinglePointPathSampler
from path_neural_networks.models.path_samplers.square_path_sampler import SquarePathSampler
from path_neural_networks.models.path_samplers.multi_scale_square_path_sampler import MultiScaleSquarePathSampler

available_path_samplers: dict = {
    "MultiScaleSquarePathSampler": MultiScaleSquarePathSampler,
    "SquarePathSampler": SquarePathSampler,
    "SinglePointPathSampler": SinglePointPathSampler
}