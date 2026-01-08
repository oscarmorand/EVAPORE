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
import logging
import numpy as np
import json
from PIL import Image
import warnings
import matplotlib.pyplot as plt

# %%
img = np.array(Image.open("/home/morand/afs/datasets/FIVES/train/Ground truth/1_A.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/New Piskel.png"))[:,:,0]

plt.imshow(img)
plt.show()

# %%
from skimage.measure import label

def betti_1_2D(mask: np.ndarray) -> int:
    '''
    Compute the 1st Betti number (number of holes) for a 2D binary mask.

    Parameters:
        mask (np.ndarray): 2D binary mask where foreground pixels are True.

    Returns:
        int: The 1st Betti number (number of holes).
    '''
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array.")
    if mask.dtype != bool:
        warnings.warn("Input mask is not boolean. Converting to boolean.")
        mask = (mask > 0)

    # Invert the mask to find holes
    inverse_mask = np.logical_not(mask)
    # Label connected components in the inverted mask, using 4-connectivity
    _, num_cc = label(inverse_mask, return_num=True, connectivity=1)
    # Subtract 1 to exclude the outer background component
    return max(num_cc - 1, 0)


# %%
patch_size = 50

height, width = img.shape

n_patches_y = height // patch_size
n_patches_x = width // patch_size

local_bettis = np.zeros((n_patches_y, n_patches_x), dtype=int)
for i in range(n_patches_y):
    for j in range(n_patches_x):
        y = i * patch_size
        x = j * patch_size
        patch = img[y:y+patch_size, x:x+patch_size]
        local_betti = betti_1_2D(patch > 0)
        local_bettis[i, j] = local_betti

# %%
plt.imshow(local_bettis, cmap='viridis')
plt.colorbar(label='Local Betti-0 Number')
plt.title('Local Betti-0 Numbers in Image Patches')
plt.show()

# %%

for patch_size in [20, 50, 100]:
    height, width = img.shape

    n_patches_y = height // patch_size
    n_patches_x = width // patch_size

    local_bettis = np.zeros((n_patches_y, n_patches_x), dtype=int)
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            y = i * patch_size
            x = j * patch_size
            patch = img[y:y+patch_size, x:x+patch_size]
            local_betti = betti_1_2D(patch > 0)
            local_bettis[i, j] = local_betti

    plt.imshow(local_bettis, cmap='viridis')
    plt.colorbar(label='Local Betti-0 Number')
    plt.title('Local Betti-0 Numbers in Image Patches')
    plt.show()


# %%
def betti_special_1_2D(mask: np.ndarray) -> int:
    '''
    Compute the 1st Betti number (number of holes) for a 2D binary mask.

    Parameters:
        mask (np.ndarray): 2D binary mask where foreground pixels are True.

    Returns:
        int: The 1st Betti number (number of holes).
    '''
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array.")
    if mask.dtype != bool:
        warnings.warn("Input mask is not boolean. Converting to boolean.")
        mask = (mask > 0)

    # Invert the mask to find holes
    inverse_mask = np.logical_not(mask)
    # Label connected components in the inverted mask, using 4-connectivity
    _, num_cc = label(inverse_mask, return_num=True, connectivity=1)

    return num_cc


# %%
from skimage.morphology import skeletonize

patch_size = 50
stride = 1
height, width = img.shape

n_patches_y = (height - patch_size) // stride + 1
n_patches_x = (width - patch_size) // stride + 1

img_p = np.pad(img, patch_size)

plt.imshow(img_p)
plt.show()

skel = skeletonize(img)

plt.imshow(skel)
plt.show()

local_bettis = np.zeros_like(img)

y, x = np.where(skel > 0)

for (j, i) in zip(x, y):
    k = patch_size // 2
    img_i = i - k + patch_size
    img_j = j - k + patch_size
    patch = img_p[img_i: img_i + patch_size, img_j: img_j + patch_size]
    local_betti = betti_special_1_2D(patch > 0)
    local_bettis[i, j] = local_betti

plt.figure(figsize=(15,15))
plt.imshow(local_bettis, cmap='nipy_spectral')
plt.colorbar(label='Local Betti-1 Number')
plt.show()

# %%
