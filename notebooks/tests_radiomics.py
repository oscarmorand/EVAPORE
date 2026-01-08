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
import matplotlib.pyplot as plt

img = Image.open("/home/morand/afs/datasets/FIVES/train/Original/1_A.png")
img_array = np.array(img)

img_array = img_array[500:1000, 1300:1800]

plt.imshow(img_array)
plt.show()

print(img_array.shape)

# %%
gray_img = np.mean(img_array, axis=2).astype(np.uint8)

plt.imshow(gray_img, cmap='gray')
plt.show()

# %%
mask = (gray_img > 10).astype(np.uint8)  # Example mask where the pixel value is greater than 10

plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
plt.show()

mask = np.expand_dims(mask, axis=0)  # Expand dims to make it 3D
gray_img = np.expand_dims(gray_img, axis=0)  # Expand dims to make it 3D

print(gray_img.shape)
print(mask.shape)

# %%
import SimpleITK as sitk

sitk_img = sitk.GetImageFromArray(gray_img)
sitk_mask = sitk.GetImageFromArray(mask)

# %%
import radiomics

# --- Patch the broken dummy progress reporter ---
class FixedDummyProgressReporter:
    def __init__(self, *args, **kwargs):  # accept arbitrary args
        pass
    def __enter__(self):  # support 'with' statements
        return self
    def __exit__(self, *args):
        pass
    def update(self, *args, **kwargs):
        pass

radiomics._DummyProgressReporter = FixedDummyProgressReporter
radiomics.featureextractor._DummyProgressReporter = FixedDummyProgressReporter

# %%
from radiomics import featureextractor

extractor = featureextractor.RadiomicsFeatureExtractor(voxelBased=True)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')

features = extractor.execute(sitk_img, sitk_mask)

# %%
print(features)

for key, value in features.items():
    if isinstance(value, sitk.Image):
        arr = sitk.GetArrayFromImage(value)[0]
        plt.imshow(arr)
        plt.title(key)
        plt.show()

# %%
import torch

features_map_tensor = torch.zeros((len(features),) + gray_img.shape[1:], dtype=torch.float32)

print(features_map_tensor.shape)
for i, (key, value) in enumerate(features.items()):
    if isinstance(value, sitk.Image):
        arr = sitk.GetArrayFromImage(value)[0]
        features_map_tensor[i] = torch.from_numpy(arr)

# %%
from graph.graph_creation import img_to_graph
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

gt = np.array(Image.open("/home/morand/afs/datasets/FIVES/train/Ground truth/1_A.png"))[:,:,0]

plt.imshow(gt, cmap='gray')
plt.show()

graph = img_to_graph(gt, clean=True, closing_radius=1, return_pixel_graph=False)

positions = []
radiuses = []
for node, data in graph.nodes(data=True):
    positions.append(data['pos'])
    radiuses.append(data.get('radius', 1))

print(positions[:5])
print(radiuses[:5])

# %%
region_masks = []
for pos, radius in zip(positions, radiuses):
        y, x = pos
        y_min = np.floor(max(0, y - radius)).astype(int)
        y_max = np.ceil(min(gt.shape[0], y + radius + 1)).astype(int)
        x_min = np.floor(max(0, x - radius)).astype(int)
        x_max = np.ceil(min(gt.shape[1], x + radius + 1)).astype(int)

        #print(f"Position: {pos}, Radius: {radius}, Region: y[{y_min}:{y_max}], x[{x_min}:{x_max}]")

        region_mask = np.zeros(gt.shape[:2], dtype=bool)
        region_mask[y_min:y_max, x_min:x_max] = True
        region_masks.append(region_mask)

print(len(region_masks))
print(len(positions))

# %%
import SimpleITK as sitk
from radiomics import featureextractor

img = Image.open("/home/morand/afs/datasets/FIVES/train/Original/1_A.png")
img_array = np.array(img)
gray_img = np.mean(img_array, axis=2).astype(np.uint8)
sitk_img = sitk.GetImageFromArray(gray_img)

extractor = featureextractor.RadiomicsFeatureExtractor(voxelBased=False)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')

region_features = []
for region_mask in region_masks:
    sitk_region = sitk.GetImageFromArray(region_mask.astype(np.uint8))

    features = extractor.execute(sitk_img, sitk_region)

    region_features.append(features)

print(region_features[0])


# %%
real_features_keys = []
for key, value in region_features[0].items():
    if isinstance(value, np.ndarray):
        real_features_keys.append(key)

print(real_features_keys)

real_region_features = []
for region_features_dict in region_features:
    real_region_features.append([region_features_dict[key] for key in real_features_keys])

real_region_features = np.array(real_region_features)
print(real_region_features.shape)  # Should be (num_regions, num_features)

# %%
