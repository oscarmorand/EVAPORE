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
#     display_name: graph-neural-networks (3.12.11)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prepare data
# This notebook serve the purpose of preparing the data for the training of the model. It includes dataset formatting to the right format, splitting the dataset, cleaning it by removing the cases that we marked as too damaged, computing the common foreground mask, and statistics about the training data that will be used for the data augmentation and the training of the model.

# %% [markdown]
# The dataset should have already been downloaded and extracted in the data directory. If not, please follow the instructions in the README to do so.
# The FIVES dataset should be located in EVAPORE/data/FIVES, and should have the following structure:
# - FIVES/
#     - test/
#         - Ground truth/
#         - Original/
#     - train/
#         - Ground truth/
#         - Original/
#     - clean_files_idx.txt
#     - (optional) Quality Assessment.xlsx
#
# Note: The clean_files_idx.txt is provided in the git repository, and contains the list of indices of the clean files in the dataset.

# %%
import os

fives_dataset_dir = os.path.abspath('../data/FIVES')
print(f"FIVES dataset directory: {fives_dataset_dir}")

# %% [markdown]
# ## Format the dataset to the accepted format
# Here we keep the original data, but you can delete it if you want to save storage

# %%
from tqdm import tqdm
from shutil import copyfile
import json

train_img_dir = os.path.join(fives_dataset_dir, "train", "Original")
train_gt_dir = os.path.join(fives_dataset_dir, "train", "Ground truth")
train_filenames_list = [f for f in os.listdir(train_img_dir) if f.endswith(('.png'))]
print(f"Number of training images: {len(train_filenames_list)}")
print(f"First 5 training images: {train_filenames_list[:5]}")

test_img_dir = os.path.join(fives_dataset_dir, "test", "Original")
test_gt_dir = os.path.join(fives_dataset_dir, "test", "Ground truth")
test_filenames_list = [f for f in os.listdir(test_img_dir) if f.endswith(('.png'))]
print(f"Number of testing images: {len(test_filenames_list)}")
print(f"First 5 testing images: {test_filenames_list[:5]}")

dst_img_dir = os.path.join(fives_dataset_dir, "img")
dst_gt_dir = os.path.join(fives_dataset_dir, "gt")
os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_gt_dir, exist_ok=True)

splits = {}
train_idx = []
for filename in tqdm(train_filenames_list):
    i = int(filename.split('_')[0])
    id = f"FIVES_{i:03d}"
    train_idx.append(id)

    new_filename = f"{id}.png"
    src_img_path = os.path.join(train_img_dir, filename)
    dst_img_path = os.path.join(dst_img_dir, new_filename)
    copyfile(src_img_path, dst_img_path)

    src_gt_path = os.path.join(train_gt_dir, filename)
    dst_gt_path = os.path.join(dst_gt_dir, new_filename)
    copyfile(src_gt_path, dst_gt_path)

test_idx = []
for filename in tqdm(test_filenames_list):
    i = int(filename.split('_')[0]) + len(train_filenames_list)
    id = f"FIVES_{i:03d}"
    test_idx.append(id)

    new_filename = f"{id}.png"
    src_img_path = os.path.join(test_img_dir, filename)
    dst_img_path = os.path.join(dst_img_dir, new_filename)
    copyfile(src_img_path, dst_img_path)

    src_gt_path = os.path.join(test_gt_dir, filename)
    dst_gt_path = os.path.join(dst_gt_dir, new_filename)
    copyfile(src_gt_path, dst_gt_path)

splits['train'] = train_idx
splits['test'] = test_idx
splits_filepath = os.path.join(fives_dataset_dir, "splits.json")
with open(splits_filepath, 'w') as f:
    json.dump(splits, f, indent=4)

# %% [markdown]
# # Data cleaning
# We import the clean file indices that we manually identified and add them to the splits

# %%
clean_files_idx_filepath = os.path.join(fives_dataset_dir, "clean_files_idx.txt")
if not os.path.exists(clean_files_idx_filepath):
    raise ValueError(f"File {clean_files_idx_filepath} does not exist.")
with open(clean_files_idx_filepath, 'r') as f:
    clean_files_idx = [line.strip() for line in f.readlines()]

print(f"Number of clean files: {len(clean_files_idx)}")
print(f"First 5 clean file indices: {clean_files_idx[:5]}")

splits['train_clean'] = clean_files_idx

with open(splits_filepath, 'w') as f:
    json.dump(splits, f, indent=4)

# %% [markdown]
# ## Foreground mask computation
# Compute the common foreground mask across all images in the train set, which will be used to refine the predictions of the model by masking out irrelevant background areas.
# This will also be used in the metrics computation to focus only on the relevant areas of the images.

# %%
import numpy as np
from PIL import Image
from skimage.measure import label

img_dir = os.path.join(fives_dataset_dir, 'train/Original/')
img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.png')]

threshold = 4
debug_i = 20

non_zero_masks = []
for i, img_path in enumerate(img_paths):
    if i >= debug_i:
        break

    img_gray = Image.open(img_path).convert("L")
    img_gray = np.array(img_gray)
    non_zero_mask = img_gray > threshold

    cc, num_cc = label(non_zero_mask, return_num=True, connectivity=2)
    cc_sizes = [np.sum(cc == i) for i in range(1, num_cc + 1)]
    max_cc_index = np.argmax(cc_sizes) + 1
    non_zero_mask = cc == max_cc_index

    non_zero_masks.append(non_zero_mask)

common_mask = np.logical_and.reduce(non_zero_masks)

# %%
import matplotlib.pyplot as plt
import torch

plt.imshow(common_mask, cmap='gray')
plt.title(f"Common Mask Between First {debug_i} Images")
plt.axis('off')
plt.show()

foreground_mask_dir = os.path.join(fives_dataset_dir, "foreground_masks")
os.makedirs(foreground_mask_dir, exist_ok=True)

torch.save(torch.from_numpy(common_mask), os.path.join(foreground_mask_dir, "FIVES.pt"))

# %% [markdown]
# # Test and visualize the dataset

# %%
from image_segmentation.data import ImageDataset, ImageDatamodule

dataset = ImageDataset(data_dir=fives_dataset_dir, transforms=None, compute_stats_only_on_foreground=True)
datamodule = ImageDatamodule(dataset, 
                             split_file_path=splits_filepath,
                             train_split_name='train_clean',
                             val_split_ratio=0.2,
                             train_transforms=None,
                             val_transforms=None,
                             test_transforms=None,
                             num_workers=0,
                             train_batch_size=4,
                             val_batch_size=1,
                             seed=42,
                             shuffle_train=True)


# %%
