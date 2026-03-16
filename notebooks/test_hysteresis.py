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
from PIL import Image
import numpy as np
from skimage.filters import apply_hysteresis_threshold

# %%
prob_map = torch.load('/home/morand/afs/EVAPORE/data/FIVES/probability_maps/FIVES_001.pt')[1].numpy()

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(40,40))
plt.imshow(prob_map)
plt.colorbar()
plt.title('Probability Map')    
plt.show()


# %%
from skimage.filters import gaussian

prob_map_rescaled = np.log(prob_map - np.min(prob_map) + 1)
prob_map_rescaled_blured = gaussian(prob_map_rescaled, sigma=2)

plt.figure(figsize=(40,40))
plt.imshow(prob_map_rescaled_blured)
plt.colorbar()
plt.title('log of Probability Map')    
plt.show()

# %%
print(prob_map.shape, prob_map.dtype, prob_map.min(), prob_map.max(), prob_map.mean())

# %%
from torch.nn.functional import sigmoid

prob = sigmoid(torch.from_numpy(prob_map)).numpy()

plt.figure(figsize=(40,40))
plt.imshow(prob)
plt.colorbar()
plt.title('After Sigmoid')
plt.show()

pred = prob > 0.5

hystersis_pred = apply_hysteresis_threshold(prob, low=0.1, high=0.5)
plt.figure(figsize=(40,40))
plt.imshow(hystersis_pred, cmap='gray')
plt.show()

plt.figure(figsize=(40,40))
plt.imshow(pred, cmap='gray')
plt.show()

pred_low = prob > 0.1
plt.figure(figsize=(40,40))
plt.imshow(pred_low, cmap='gray')
plt.show()

# %%
