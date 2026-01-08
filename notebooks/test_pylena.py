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
import pylena

# %%
import numpy as np

# %%
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("/home/morand/afs/EVAPORE/scripts/app/curved_up.png").convert("L")
img = np.array(img)

plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# %%
tree = pylena.morpho.tos(img)

print(tree.parent)
print(tree.nodemap)
print(tree.values)

print(tree.nodemap.shape)

# %%
plt.imshow(tree.nodemap, cmap='gray')
plt.title('Tree of Shapes Nodemap')
plt.axis('off')
plt.show()

# %%
y0, x0 = 65, 320
y1, x1 = 460, 350

n0 = tree.nodemap[y0, x0]
n1 = tree.nodemap[y1, x1]

print(n0, n1)
print(tree.values[n0], tree.values[n1])
print(tree.parent[n0], tree.parent[n1])

# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.graph import route_through_array
from numba import njit
from typing import Optional
from collections.abc import Sequence
from imageio.v3 import imread

import pylena as pln

@njit
def _dahu_tos_mask_indices(parent: np.ndarray, n1: int, n2: int, depth: np.ndarray) -> np.ndarray:
    M = np.zeros_like(parent)

    while depth[n1] > depth[n2]:
        M[n1] = 1
        n1 = parent[n1]
    while depth[n2] > depth[n1]:
        M[n2] = 1
        n2 = parent[n2]
    while n1 != n2:
        M[n1] = 1
        M[n2] = 1
        n1 = parent[n1]
        n2 = parent[n2]
    M[n1] = 1

    return M

def dahu_tos_mask(t: pln.morpho.ComponentTree, n1: int, n2: int, depth: Optional[np.ndarray] = None) -> np.ndarray:
    if depth is None:
        depth = t.compute_depth()
    mask_nodes = _dahu_tos_mask_indices(t.parent, n1, n2, depth).astype(bool)
    return t.reconstruct(mask_nodes).astype(bool)

def dahu_shortest_path(
    t: pln.morpho.ComponentTree,
    p1: Sequence[int],
    p2: Sequence[int],
    depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    assert len(p1) == len(p2) == t.nodemap.ndim
    if depth is None:
        depth = t.compute_depth()
    p1 = tuple([p1[i] * 2 - 1 for i in range(len(p1))])
    p2 = tuple([p2[i] * 2 - 1 for i in range(len(p2))])

    n1 = t.nodemap[tuple(p1)]
    n2 = t.nodemap[tuple(p2)]
    ROI = dahu_tos_mask(t, n1, n2, depth)
    w = np.ones_like(t.nodemap, dtype=np.uint8) * 255
    w[ROI] = 1
    path, _ = route_through_array(w, p1, p2, geometric=False, fully_connected=False)

    return np.asarray(path)


# %%
img = Image.open("/home/morand/afs/EVAPORE/data/FIVES/img/FIVES_001.png").convert("L")
img = np.array(img)

t = pln.morpho.tos(img, padding="median", subsampling="full")

y0, x0 = 350, 935
y1, x1 = 815, 1575
path = dahu_shortest_path(t, (y0, x0), (y1, x1))

# %%
mask1 = np.zeros((t.nodemap.shape[0], t.nodemap.shape[1], 3), dtype=np.uint8)
mask1[:, :, 0] = t.reconstruct()
mask1[:, :, 1] = mask1[:, :, 0]
mask1[:, :, 2] = mask1[:, :, 0]
mask1[path[:, 0], path[:, 1], :] = [255, 0, 0]

plt.figure(figsize=(10, 10))
plt.imshow(mask1)
plt.show()

# %%
