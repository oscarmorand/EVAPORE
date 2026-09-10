from path_neural_networks.models.path_samplers.aggregation_methods.pooling_aggregation import SamplingMaxAggregation, SamplingMinAggregation
from path_neural_networks.models.path_samplers.aggregation_methods.soft_aggregation import SamplingMeanAggregation, SamplingSoftMaxAggregation, SamplingSumAggregation

available_aggregation_methods = {
    "SamplingMaxAggregation": SamplingMaxAggregation,
    "SamplingMinAggregation": SamplingMinAggregation,
    "SamplingMeanAggregation": SamplingMeanAggregation,
    "SamplingSoftMaxAggregation": SamplingSoftMaxAggregation,
    "SamplingSumAggregation": SamplingSumAggregation
}