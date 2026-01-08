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
import os
import torch
from PIL import Image
import numpy as np

n_features = 16

processed_dir = "~/afs/EVAPORE/data/FIVES/processed/"
data_dir = "~/afs/EVAPORE/data/FIVES/feature_maps/unet/"
gt_dir = "~/afs/EVAPORE/data/FIVES/gt/"

gt_files = os.listdir(os.path.expanduser(gt_dir))
gt_files.sort()

processed_files = os.listdir(os.path.expanduser(processed_dir))
processed_files.sort()

for i, (gt_file, processed_file) in enumerate(zip(gt_files, processed_files)):
    gt_path = os.path.join(os.path.expanduser(gt_dir), gt_file)
    gt = np.array(Image.open(gt_path))

    feature_map_path = os.path.join(os.path.expanduser(data_dir), processed_file)

    feature_map = torch.rand((n_features, gt.shape[0], gt.shape[1]))

    print(f"Saving feature map {i} to {feature_map_path} with shape {feature_map.shape}")
    torch.save(feature_map, feature_map_path)

# %%
import os
import torch
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
import gc

n_features = 64

input_dir = "/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds/"
data_dir = "~/afs/EVAPORE/data/FIVES/feature_maps/qtseg/"

input_files = [f for f in os.listdir(input_dir) if (f.endswith('.pt') and 'embeddings' in f)]
input_indexes = [int(f.split('_')[0]) for f in input_files]
arg_sorted_indexes = np.argsort(input_indexes)
print(len(arg_sorted_indexes))

print(input_files)

for index in tqdm(arg_sorted_indexes):
    input_file = input_files[index]
    i = input_indexes[index]
    new_name = "FIVES_" + str(i).zfill(3) + ".pt"
    input_path = os.path.join(input_dir, input_file)
    feature_map = torch.load(input_path, map_location='cpu', weights_only=False)
    output_path = os.path.join(os.path.expanduser(data_dir), new_name)
    torch.save(feature_map, output_path)

    del feature_map
    gc.collect()
'''
for i, (gt_file, processed_file) in enumerate(zip(gt_files, processed_files)):
    gt_path = os.path.join(os.path.expanduser(gt_dir), gt_file)
    gt = np.array(Image.open(gt_path))

    feature_map_path = os.path.join(os.path.expanduser(data_dir), processed_file)

    feature_map = torch.rand((n_features, gt.shape[0], gt.shape[1]))

    print(f"Saving feature map {i} to {feature_map_path} with shape {feature_map.shape}")
    torch.save(feature_map, feature_map_path)
'''

# %%
