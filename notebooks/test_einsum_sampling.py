# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: graph-neural-networks
#     language: python
#     name: python3
# ---

# %%
import torch

# %%
feature_maps = torch.arange(50).reshape(2, 5, 5) # Example feature maps with shape (channels, height, width)

print(feature_maps)

# %%
sampling_squares = torch.zeros((4, 5, 5), dtype=torch.long) # Example sampling squares with shape (num_squares, height, width)
sampling_squares[0, 1:3, 1:3] = 1.0  # First square
sampling_squares[1, 0:2, 0:2] = 1.0 # Second square
sampling_squares[2, 3:5, 3:5] = 1.0  # Third square
sampling_squares[3, 2:4, 0:2] = 1.0  # Fourth square

print(sampling_squares)

# %%
features_sum = torch.einsum('chw,nhw->nc', feature_maps, sampling_squares)

print(features_sum.shape)
print(features_sum)  # Output shape will be (num_squares, channels)

# %%
sampling_area = torch.sum(sampling_squares, dim=(1, 2)).unsqueeze(0).T  # Shape (num_squares, 1, 1)
print(sampling_area)
features_mean = features_sum / sampling_area

print(features_mean.shape)
print(features_mean)  # Output shape will be (num_squares, channels)

# %%
