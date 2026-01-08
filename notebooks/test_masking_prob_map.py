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
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# %%
prob_map_path = "/home/morand/afs/EVAPORE/data/FIVES/probability_maps/FIVES_001.pt"
prob_map = torch.load(prob_map_path, map_location="cpu", weights_only=False)

print(prob_map.shape)

# %%
prob_map = prob_map[1].numpy()

plt.imshow(prob_map, cmap='gray')
plt.title("Original Probability Map")
plt.axis('off')
plt.show()

# %%
img_path = "/home/morand/afs/EVAPORE/data/FIVES/img/FIVES_001.png"

img = Image.open(img_path).convert("RGB")
img = np.array(img)
print(img.shape)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')
plt.show()

print(img[:10,:10])

# %%
img_gray = Image.open(img_path).convert("L")
img_gray = np.array(img_gray)

print(img_gray)

non_zero_mask = img_gray > 4

plt.imshow(non_zero_mask, cmap='gray')
plt.title("Non-zero Mask")
plt.axis('off')
plt.show()

# %%
gray_mask = (img[:,:,0] == img[:,:,1]) & (img[:,:,0] == img[:,:,2])

plt.imshow(gray_mask, cmap='gray')
plt.title("Gray Mask from RGB")
plt.axis('off')
plt.show()

# %%
from skimage.measure import label

cc, num_cc = label(non_zero_mask, return_num=True, connectivity=2)

cc_sizes = [np.sum(cc == i) for i in range(1, num_cc + 1)]
max_cc_index = np.argmax(cc_sizes) + 1
non_zero_mask = cc == max_cc_index

plt.imshow(non_zero_mask, cmap='gray')
plt.title("Non-zero Mask from Grayscale Image")
plt.axis('off')
plt.show()

# %%
prob_map_normalized = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())
prob_map_masked = prob_map_normalized * non_zero_mask.astype(prob_map.dtype)

plt.imshow(prob_map_masked, cmap='gray')
plt.title("Masked Probability Map")
plt.colorbar()
plt.axis('off')
plt.show()

# %%
min_val = prob_map[non_zero_mask].min()
prob_map_masked = prob_map.copy()
prob_map_masked[~non_zero_mask] = min_val

plt.imshow(prob_map_masked, cmap='gray')
plt.title("Masked Probability Map")
plt.colorbar()
plt.axis('off')
plt.show()

# %%
prob_dir = "/home/morand/afs/EVAPORE/data/FIVES/probability_maps/"
img_dir = "/home/morand/afs/EVAPORE/data/FIVES/img/"

prob_paths = os.listdir(prob_dir)
img_paths = [os.path.join(img_dir, fname.replace('.pt', '.png')) for fname in prob_paths]

threshold = 4
debug_i = 20

non_zero_masks = []
for i, (prob_map_name, img_path) in enumerate(zip(prob_paths, img_paths)):
    if i >= debug_i:
        break
    prob_map_path = os.path.join(prob_dir, prob_map_name)
    prob_map = torch.load(prob_map_path, map_location="cpu", weights_only=False)
    prob_map = prob_map[1].numpy()

    img_gray = Image.open(img_path).convert("L")
    img_gray = np.array(img_gray)
    non_zero_mask = img_gray > threshold

    cc, num_cc = label(non_zero_mask, return_num=True, connectivity=2)
    cc_sizes = [np.sum(cc == i) for i in range(1, num_cc + 1)]
    max_cc_index = np.argmax(cc_sizes) + 1
    non_zero_mask = cc == max_cc_index

    non_zero_masks.append(non_zero_mask)

common_mask = np.logical_and.reduce(non_zero_masks)

plt.imshow(common_mask, cmap='gray')
plt.title("Common Mask Between First Two Images")
plt.axis('off')
plt.show()

# %%
torch.save(torch.from_numpy(common_mask), "/home/morand/afs/EVAPORE/data/FIVES/foreground_masks/FIVES.pt")

# %%
