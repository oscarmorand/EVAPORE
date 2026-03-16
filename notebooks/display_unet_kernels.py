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
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# %%
from graph_neural_networks.models.binary_segmentator import BinarySegmentator

device = "cpu"

model = BinarySegmentator.load_from_checkpoint("/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_202/checkpoints/best-checkpoint-epoch=65-val_loss=0.0685.ckpt", map_location=device)

# %%
# go through the layers of the model and display the kernels of the first convolutional layer

for param in model.parameters():
     param.requires_grad = False

for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        print(f"Layer: {name}")
        kernels = layer.weight.data.cpu().numpy()
        num_kernels = kernels.shape[0]
        num_channels = kernels.shape[1]
        kernel_size = kernels.shape[2]
        
        if num_channels == 3:
            fig, axes = plt.subplots(num_kernels, 1, figsize=(num_channels*2, num_kernels*2))
            for i in range(num_kernels):
                rgb_kernel = kernels[i].transpose(1, 2, 0)
                rgb_kernel = (rgb_kernel - rgb_kernel.min()) / (rgb_kernel.max() - rgb_kernel.min())
                axes[i].imshow(rgb_kernel)
                axes[i].axis('off')
            plt.show()
        else:
        # display the kernels
            fig, axes = plt.subplots(num_kernels, num_channels, figsize=(num_channels*2, num_kernels*2))
            for i in range(num_kernels):
                for j in range(num_channels):
                    axes[i, j].imshow(kernels[i, j], cmap='gray')
                    axes[i, j].axis('off')
            plt.show()

# %%
