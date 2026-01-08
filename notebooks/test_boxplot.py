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
import matplotlib.pyplot as plt
import numpy as np

# %%
before_scores = np.random.normal(0.5, 0.1, 100)
after_scores = np.random.normal(0.6, 0.1, 100)
relative_differences = (after_scores - before_scores) / before_scores

# %%
y = ([before_scores, after_scores], [relative_differences])
labels = (["before", "after"], ["relative difference"])
fig, ax = plt.subplots(nrows=1, ncols=2)
ax: list[plt.Axes]
ax[0].boxplot(y[0], tick_labels=labels[0])
ax[0].set_ylabel('metric value')
ax[0].set_title('CCDice')
ax[1].boxplot(y[1], tick_labels=labels[1])
fig.canvas.draw()
img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
plt.close(fig)

# %%
plt.imshow(img)

# %%
from torch_geometric.io import fs

probability_map_path = "/home/morand/afs/EVAPORE/data/FIVES/probability_maps/FIVES_001.pt"

probability_map = fs.torch_load(probability_map_path)
print(probability_map.shape)
print(probability_map.dtype)

# %%
prob_img = probability_map[1]
prob_img = (prob_img - prob_img.min()) / (prob_img.max() - prob_img.min())
prob_img = (np.stack([prob_img.numpy()]*3, axis=-1) * 255.0).astype(np.uint8)

plt.imshow(prob_img)
plt.colorbar()
plt.axis('off')
plt.show()

# %%
