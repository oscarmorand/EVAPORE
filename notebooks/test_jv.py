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
import networkx as nx
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
    return num_cc - 1


# %%
import gudhi as gd

def betti_number_1(binary_image):
    """Compute Betti 1 (number of loops/holes) using persistent homology."""
    # Create a cubical complex from the binary image
    cubical_complex = gd.CubicalComplex(top_dimensional_cells=binary_image)
    # Compute persistence
    cubical_complex.compute_persistence()
    # Count the number of 1D holes (loops)
    persistence = cubical_complex.persistence_intervals_in_dimension(1)
    return len(persistence)


# %%
print(betti_1_2D(img))
print(betti_number_1(img))

# %%
from skimage.morphology import skeletonize

skel = skeletonize(img).astype(int)

print(betti_1_2D(skel))
print(betti_number_1(skel))

# %%
plt.imshow(skel, cmap='gray')
plt.show()

skel[24, 8] = 1

plt.imshow(skel, cmap='gray')
plt.show()

# %%
kernel_1 = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
kernel_2 = np.array([[64, 32, 16],
                     [128, 0, 8],
                     [1, 2, 4]])

from scipy.signal import convolve2d

convolved_1 = convolve2d(skel, kernel_1, mode='same', boundary='fill', fillvalue=0)
convolved_2 = convolve2d(skel, kernel_2, mode='same', boundary='fill', fillvalue=0)

n_neighbors_c8 = convolved_1 * skel
extremities = n_neighbors_c8 == 1

plt.imshow(extremities, cmap='gray')
plt.show()

plt.imshow(convolved_1 * skel, cmap='gray')
plt.show()

plt.imshow(convolved_2 * skel, cmap='gray')
plt.show()

A = convolved_2 * skel
B = ((A << 1) & A)
C = (B > 0) | (A == 0b10000001)

plt.imshow(C, cmap='gray')
plt.show()

final = (n_neighbors_c8 == 1) | ((n_neighbors_c8 == 2) & C)

plt.imshow(final, cmap='gray')
plt.show()

# %%
from scipy.signal import convolve2d

def detect_final_pixels(skel: np.ndarray) -> np.ndarray:
    kernel_1 = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
    kernel_2 = np.array([[64, 32, 16],
                        [128, 0, 8],
                        [1, 2, 4]])

    convolved_1 = convolve2d(skel, kernel_1, mode='same', boundary='fill', fillvalue=0)
    convolved_2 = convolve2d(skel, kernel_2, mode='same', boundary='fill', fillvalue=0)

    n_neighbors_c8 = convolved_1 * skel
    simple_final_pixels = n_neighbors_c8 == 1

    binary_neighbourhood = convolved_2 * skel
    consecutives_ones = ((binary_neighbourhood << 1) & binary_neighbourhood)
    complex_pixels = (consecutives_ones > 0) | (binary_neighbourhood == 0b10000001)
    complex_final_pixels = (n_neighbors_c8 == 2) & complex_pixels

    final = simple_final_pixels | complex_final_pixels
    return final


def detect_old_final_pixels(skel: np.ndarray) -> np.ndarray:
    kernel_1 = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])

    convolved_1 = convolve2d(skel, kernel_1, mode='same', boundary='fill', fillvalue=0)

    n_neighbors_c8 = convolved_1 * skel
    simple_final_pixels = n_neighbors_c8 == 1
    return simple_final_pixels


# %%
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

def detect_final_pixels_fast(skel: np.ndarray) -> np.ndarray:
    """Detect final pixels in a skeletonized image using a fast method.

    Args:
        skel (np.ndarray): The skeletonized image.

    Returns:
        np.ndarray: A binary mask of the final pixels.
    """

    # Define the kernels for convolution
    kernels = np.array([
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ],
        [
            [64, 32, 16],
            [128, 0, 8],
            [1, 2, 4]
        ]
    ])

    _, kH, kW = kernels.shape

    # Compute padding sizes
    pad_h = kH // 2
    pad_w = kW // 2

    # Pad the input image
    padded_image = np.pad(skel, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    patches = sliding_window_view(padded_image, (kH, kW))

    # Perform convolution using einsum
    convolved = np.einsum('ijkl,mkl->mij', patches, kernels)

    # Get "simple" final pixels, those with only one neighbor in connectivity 8
    n_neighbors_c8 = convolved[0] * skel
    simple_final_pixels = n_neighbors_c8 == 1

    # Get "complex" final pixels, those with two neighbors in connectivity 8 but with special patterns
    binary_neighbourhood = convolved[1] * skel
    consecutives_ones = ((binary_neighbourhood << 1) & binary_neighbourhood)
    complex_pixels = (consecutives_ones > 0) | (binary_neighbourhood == 0b10000001)
    complex_final_pixels = (n_neighbors_c8 == 2) & complex_pixels

    # Combine both types of final pixels
    final = simple_final_pixels | complex_final_pixels
    return final

def get_number_of_extremities(skel: np.ndarray) -> int:
    """
    Get the number of extremities (final pixels) in a skeletonized image.
    
    Args:
        skel (np.ndarray): The skeletonized image.
        
    Returns:
        int: The number of extremities.
    """
    final_pixels = detect_final_pixels_fast(skel)
    return np.sum(final_pixels)


# %%
import time

start_time = time.time()
final = detect_final_pixels(skel)
print("Standard method time: %s seconds" % (time.time() - start_time))

start_time = time.time()
final_fast = detect_final_pixels_fast(skel)
print("Fast method time: %s seconds" % (time.time() - start_time))


# %%
final = detect_final_pixels(skel)

plt.imshow(final, cmap='gray')
plt.show()


# %%
def plot_final_pixels(skel: np.ndarray) -> None:
    final_pixels = detect_final_pixels(skel)
    plt.imshow(skel, cmap='gray')
    y_coords, x_coords = np.where(final_pixels)
    plt.scatter(x_coords, y_coords, color='red', s=100)
    plt.show()


# %%
plot_final_pixels(skel)

# %%
skel_1 = np.array(Image.open("/home/morand/afs/tests/01R_2ndHO.png"))[:,:,0] > 0
skel_2 = np.array(Image.open("/home/morand/afs/tests/09L_2ndHO.png"))[:,:,0] > 0


# %%
plt.imshow(skel_1, cmap='gray')
plt.show()

plt.imshow(skel_2, cmap='gray')
plt.show()

# %%
print(np.sum(detect_final_pixels(skel_1)))
print(np.sum(detect_final_pixels(skel_2)))

# %%
plot_final_pixels(skel_1)

# %%
print(skel_1.sum(), skel_1.shape, skel_1.min(), skel_1.max())


# %%
def plot_old_final_pixels(skel: np.ndarray) -> None:
    final_pixels = detect_old_final_pixels(skel)
    plt.imshow(skel, cmap='gray')
    y_coords, x_coords = np.where(final_pixels)
    plt.scatter(x_coords, y_coords, color='red', s=100)
    plt.show()


# %%
plot_old_final_pixels(skel_1)

# %%
from utils.betti_numbers_2D import betti_0_2D

def betti_number_0(binary_image):
    """Compute Betti 0 (number of connected components)."""
    labeled_image = label(binary_image)
    return len(np.unique(labeled_image)) - 1  # Subtract 1 for background


print(betti_0_2D(img))
print(betti_number_0(img))


# %%
def count_neighbors(binary_image):
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_counts = convolve2d(binary_image, kernel, mode='same', boundary='fill', fillvalue=0)
    neighbor_counts = np.clip(neighbor_counts, 0, None)
    neighbor_counts = neighbor_counts * binary_image
    return neighbor_counts


def count_neighbors_less(binary_image):
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    kernel[0, 0] = 0
    kernel[0, 2] = 0
    kernel[2, 0] = 0
    kernel[2, 2] = 0
    neighbor_counts = convolve2d(binary_image, kernel, mode='same', boundary='fill', fillvalue=0)
    neighbor_counts = np.clip(neighbor_counts, 0, None)
    neighbor_counts = neighbor_counts * binary_image
    return neighbor_counts



# %%
def count_real_neighbors(skel: np.ndarray) -> int:
    real_neighbors = np.zeros_like(skel)

    height, width = skel.shape
    skel = np.pad(skel, pad_width=2, mode='constant', constant_values=0)

    for i in range(height):
        for j in range(width):
            if skel[i + 2, j + 2] == 0:
                continue

            patch = skel[i:i+5, j:j+5]

            cc = label(patch, connectivity=2)
            center_label = cc[2, 2]
            center_component = (cc == center_label)

            neighbor_5 = center_component.copy()
            neighbor_5[1:4, 1:4] = 0

            _, n_neighbors = label(neighbor_5, connectivity=2, return_num=True)

            real_neighbors[i, j] = n_neighbors

    return real_neighbors


# %%
def count_junctions(binary_image):
    neighbor_counts = count_real_neighbors(binary_image)

    # Count junctions
    junction_counts = {
        '1': np.sum(neighbor_counts == 1),
        '2': np.sum(neighbor_counts == 2),
        '3': np.sum(neighbor_counts == 3),
        '4': np.sum(neighbor_counts == 4),
        '5': np.sum(neighbor_counts == 5),
        '6': np.sum(neighbor_counts == 6),
        '7': np.sum(neighbor_counts == 7),
        '8': np.sum(neighbor_counts == 8),
    }
    return junction_counts


def plot_junctions(skel: np.ndarray) -> None:
    plt.figure(figsize=(12, 12))
    neighbor_counts = count_real_neighbors(skel)
    neighbor_range = np.array([i for i in np.unique(neighbor_counts) if i > 0])
    print(neighbor_range)
    plt.imshow(skel, cmap='gray')
    for n in neighbor_range:
        if n == 2:
            continue
        y_coords, x_coords = np.where(neighbor_counts == n)
        plt.scatter(x_coords, y_coords, s=50, label=f'Junctions with {n} neighbors')
    plt.legend()
    plt.show()

def process_batch(batch):
    """Process a batch of (image_array, metadata) and return results."""
    results = {}
    for image_array, metadata in batch:
        result_image = metadata['image_out_name']
        try:
            b0 = betti_number_0(image_array)
            b1 = betti_1_2D(image_array)
            junctions = count_junctions(image_array)
            results[result_image] = {
                'betti_0': b0,
                'betti_1': b1,
                'junctions': junctions,
                'white_pixels': np.count_nonzero(image_array),
                'final_extremities': get_number_of_extremities(image_array)
            }
        except Exception as e:
            print(f"Error processing {result_image}: {e}")
    return results



# %%
plot_junctions(skel)


# %%
def count_old_neighbors(binary_image):
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_counts = convolve2d(binary_image, kernel, mode='same', boundary='fill', fillvalue=0)
    neighbor_counts = np.clip(neighbor_counts, 0, None)
    neighbor_counts = neighbor_counts * binary_image
    return neighbor_counts


# %%
def plot_old_junctions(skel: np.ndarray) -> None:
    neighbor_counts = count_neighbors(skel)
    plt.figure(figsize=(12, 12))
    neighbor_range = np.array([i for i in np.unique(neighbor_counts) if i > 0])
    print(neighbor_range)
    plt.imshow(skel, cmap='gray')
    for n in neighbor_range:
        if n == 2:
            continue
        y_coords, x_coords = np.where(neighbor_counts == n)
        plt.scatter(x_coords, y_coords, s=50, label=f'Junctions with {n} neighbors')
    plt.legend()
    plt.show()


# %%
plot_old_junctions(skel[300:400, 350:450])

# %%
skel = (np.array(Image.open("/home/morand/afs/tests/01R_2ndHO.png"))[:,:,0] > 0).astype(int)

skel = skel[100:300, 100:300]

# %%
plot_old_junctions(skel)
plot_junctions(skel)

# %%
batch = [(skel, {'image_out_name': 'test_image'})]

results = process_batch(batch)
print(results)

# %%
skel = (np.array(Image.open("/home/morand/afs/tests/05.png"))[:,:,0] > 0).astype(int)

skel = skel[0:100, 0:100]

plt.imshow(skel, cmap='gray')
plt.show()

plot_old_junctions(skel)
plot_junctions(skel)

# %%
skel = (np.array(Image.open("/home/morand/afs/tests/clDice_4_gt_Image_08R_2ndHO.png")) > 0).astype(int)

plot_old_junctions(skel)
plot_junctions(skel)


# %%
def new_count_real_neighbors(skel: np.ndarray) -> int:
    """
    Detects the number of neighbors for each pixel in the skeletonized image
    """
    final_pixels = detect_final_pixels_fast(skel)
    
    real_neighbors = np.zeros_like(skel)

    height, width = skel.shape
    skel_p = np.pad(skel, pad_width=2, mode='constant', constant_values=0)

    for i in range(height):
        for j in range(width):
            if skel_p[i + 2, j + 2] == 0:
                continue

            patch = skel_p[i:i+5, j:j+5]

            cc = label(patch, connectivity=2)
            center_label = cc[2, 2]
            center_component = (cc == center_label)

            neighbor_5 = center_component.copy()
            neighbor_5[1:4, 1:4] = 0

            _, n_neighbors = label(neighbor_5, connectivity=2, return_num=True)

            real_neighbors[i, j] = n_neighbors

    junctions = real_neighbors > 2
    neighbors_cc = label(junctions, connectivity=2)

    simple_neighbors = np.zeros_like(real_neighbors)
    for region_label in np.unique(neighbors_cc):
        if region_label == 0:
            continue

        region_mask = (neighbors_cc == region_label)
        region_n_neighbors = real_neighbors * region_mask
        max_id = np.argmax(region_n_neighbors)
        max_coords = np.unravel_index(max_id, region_n_neighbors.shape)

        simple_neighbors[max_coords] = real_neighbors[max_coords]

    simple_neighbors = simple_neighbors + final_pixels

    skel = skel * 2
    skel[simple_neighbors > 0] = simple_neighbors[simple_neighbors > 0]

    return skel


# %%
def plot_new_junctions(skel: np.ndarray) -> None:
    neighbor_counts = new_count_real_neighbors(skel)
    plt.figure(figsize=(12, 12))
    plt.imshow(neighbor_counts, cmap='nipy_spectral')
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 12))
    neighbor_range = np.array([i for i in np.unique(neighbor_counts) if i > 0])
    print(neighbor_range)
    plt.imshow(skel, cmap='gray')
    for n in neighbor_range:
        if n == 2:
            continue
        y_coords, x_coords = np.where(neighbor_counts == n)
        plt.scatter(x_coords, y_coords, s=50, label=f'Junctions with {n} neighbors')
    plt.legend()
    plt.show()


# %%
skel = (np.array(Image.open("/home/morand/afs/tests/05.png"))[:,:,0] > 0).astype(int)

skel = skel[0:100, 0:100]

plt.imshow(skel, cmap='gray')
plt.show()

plot_old_junctions(skel)
plot_junctions(skel)
plot_new_junctions(skel)

# %%
real_skel = skeletonize(skel).astype(int)

plt.imshow(real_skel, cmap='gray')
plt.show()


# %%
plot_new_junctions(real_skel)

# %%
skel = (np.array(Image.open("/home/morand/afs/tests/01R_2ndHO.png"))[:,:,0] > 0).astype(int)

skel = skel[100:300, 100:300]

plot_old_junctions(skel)
plot_junctions(skel)
plot_new_junctions(skel)

# %%
'''
Author: Oscar Morand (LRE, CREATIS)
Date: October 2025
Description: Functions to compute extremities and junctions on 2d binary skeletons
'''

from skimage.measure import label
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import singledispatch

# ============================================
# Extremities detection
# ============================================

def detect_extremities(skel: np.ndarray) -> np.ndarray:
    """Detect extremities (final pixels) in a skeletonized image using a fast method.

    Args:
        skel (np.ndarray): The skeletonized image.

    Returns:
        np.ndarray: A binary mask of the extremities.
    """

    # Define the kernels for convolution
    kernels = np.array([
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ],
        [
            [64, 32, 16],
            [128, 0, 8],
            [1, 2, 4]
        ]
    ])

    _, kH, kW = kernels.shape

    # Compute padding sizes
    pad_h = kH // 2
    pad_w = kW // 2

    # Pad the input image
    padded_image = np.pad(skel, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    patches = sliding_window_view(padded_image, (kH, kW))

    # Perform convolution using einsum
    convolved = np.einsum('ijkl,mkl->mij', patches, kernels)

    # Get "simple" final pixels, those with only one neighbor in connectivity 8
    n_neighbors_c8 = convolved[0] * skel
    simple_final_pixels = n_neighbors_c8 == 1

    # Get "complex" final pixels, those with two neighbors in connectivity 8 but with special patterns
    binary_neighborhood = convolved[1] * skel
    consecutives_ones = ((binary_neighborhood << 1) & binary_neighborhood)
    complex_pixels = (consecutives_ones > 0) | (binary_neighborhood == 0b10000001)
    complex_final_pixels = (n_neighbors_c8 == 2) & complex_pixels

    # Combine both types of final pixels
    final = simple_final_pixels | complex_final_pixels
    return final

def get_number_of_extremities(skel: np.ndarray) -> int:
    """
    Get the number of extremities (final pixels) in a skeletonized image.
    
    Args:
        skel (np.ndarray): The skeletonized image.
        
    Returns:
        int: The number of extremities.
    """
    final_pixels = detect_extremities(skel)
    return np.sum(final_pixels)


# ============================================
# Junctions detection
# ============================================

def compute_neighbors_regions(skel: np.ndarray) -> np.ndarray:
    """Compute the number of neighboring regions for each pixel in the skeleton.

    Args:
        skel (np.ndarray): The skeletonized image.

    Returns:
        np.ndarray: The skeleton with each pixel value being the number of neighboring pixels.
    """
    neighbors_regions = np.zeros_like(skel)
    height, width = skel.shape
    skel_p = np.pad(skel, pad_width=2, mode='constant', constant_values=0)

    for i in range(height):
        for j in range(width):
            if skel_p[i + 2, j + 2] == 0:
                continue

            # Get a 5-pixels wide square patch of pixels
            patch = skel_p[i:i+5, j:j+5]

            # Only keep the pixels that are connected to the center pixel
            cc = label(patch, connectivity=2)
            center_label = cc[2, 2]
            center_component = (cc == center_label)

            # Get only the 1-pixel wide band of pixels that constitute the edge of the 5-pixels square patch
            neighbor_5 = center_component.copy()
            neighbor_5[1:4, 1:4] = 0

            # Get the number of connected components on this band, to get the number of independant branches
            _, n_neighbors = label(neighbor_5, connectivity=2, return_num=True)

            neighbors_regions[i, j] = n_neighbors

    return neighbors_regions


def _clean_neighbors_regions(real_neighbors: np.ndarray) -> np.ndarray:
    """
    Cleans the regions of more-than-2 neighbors pixels by keeping only the pixel with the maximum
    number of neighbors in each region.

    Args:
        real_neighbors (np.ndarray): The skeleton with each pixel value being the number of neighbors.

    Returns:
        np.ndarray: The cleaned skeleton with only one pixel per region of more-than-2 neighbors pixels.
    """

    junctions = real_neighbors > 2
    neighbors_cc = label(junctions, connectivity=2)

    simple_neighbors = np.zeros_like(real_neighbors)
    for region_label in np.unique(neighbors_cc):
        if region_label == 0:
            continue

        region_mask = (neighbors_cc == region_label)
        region_n_neighbors = real_neighbors * region_mask
        max_id = np.argmax(region_n_neighbors)
        max_coords = np.unravel_index(max_id, region_n_neighbors.shape)

        simple_neighbors[max_coords] = real_neighbors[max_coords]

    return simple_neighbors


def compute_neighbors_count(skel: np.ndarray) -> int:
    """
    Detects the number of neighbors for each pixel in the skeletonized image.
    WARNING: this function was not formally tested, and there are still some errors,
    so the result is an approximation of the real neighbor count of junctions pixels,
    especially with complex shaped skeletons like in ROSE dataset, or with skeletons
    that weren't simplified, e.g. with manual annotators.

    Args:
        skel (np.ndarray): The input binary skeleton

    Returns:
        np.ndarray: The skeleton with each pixel value being an approximation of its number of neighbors
    """

    if not isinstance(skel, np.ndarray):
        raise ValueError(f"Input skeleton must be a numpy array, got {type(skel)}")
    if skel.ndim != 2:
        raise ValueError(f"Input skeleton must be a 2D array, got {skel.ndim}D array of shape {skel.shape}")

    if skel.dtype == np.bool:
        warnings.warn("Input skeleton is of boolean type. It will be converted to integer type for processing.")
        skel = skel.astype(int)
    if set(np.unique(skel)) != {0, 1}:
        warnings.warn("Input skeleton is not binary. It will be thresholded at >0 for processing.")
        skel = (skel > 0).astype(int)
    if np.sum(skel) == 0:
        warnings.warn("Input skeleton is empty. Returning the original skeleton.")
        return skel

    # First, detect extremities to add them back later
    extremities = detect_extremities(skel)
    
    # Compute the number of neighbors for each pixel in the skeleton
    neighbors_regions = compute_neighbors_regions(skel)

    # Right now, there are some regions of more-than-2 neighbors pixels that are clamped together, we only need to keep
    # the pixel that has the maximum number of neighbors
    simple_neighbors = _clean_neighbors_regions(neighbors_regions)
    
    # Add back the extremities
    simple_neighbors = simple_neighbors + extremities

    # Add back the normal skeleton pixels (2 neighbors pixels)
    final_neighbor_count = skel * 2
    final_neighbor_count[simple_neighbors > 0] = simple_neighbors[simple_neighbors > 0]

    # return the final skeleton with each pixel value being an approximation of the number of neighbors
    return final_neighbor_count



def plot_junctions(skel: np.ndarray, neighbor_counts: dict = None) -> None:
    """
    Plots the junctions detected in the skeleton.

    Args:
        neighbor_counts (dict): A dictionary with the count of junctions for each number of neighbors
    """

    if neighbor_counts is None:
        neighbor_counts = compute_neighbors_count(skel)

    neighbor_range = np.array([i for i in np.unique(neighbor_counts) if i > 0])

    plt.figure(figsize=(12, 12))
    plt.imshow(skel, cmap='gray')
    plt.axis('off')
    for n in neighbor_range:
        if n == 2:
            continue
        y_coords, x_coords = np.where(neighbor_counts == n)
        plt.scatter(x_coords, y_coords, s=20, label=f'Junctions with {n} neighbors')
    plt.legend()
    plt.show()



def count_junctions(skel: np.ndarray, max_n_neighbors: int = 8) -> dict:
    """
    Counts the number of junctions in the skeleton.

    Args:
        skel (np.ndarray): The input binary skeleton.

    Returns:
        dict: A dictionary with the count of junctions for each number of neighbors.
    """

    neighbor_counts = compute_neighbors_count(skel)

    junction_counts = {str(k): np.sum(neighbor_counts == k) for k in range(1, max_n_neighbors + 1)}
    
    return junction_counts



@singledispatch
def plot_junctions_distribution(arg) -> None:
    raise TypeError(f"Unsupported type: {type(arg)}")

@plot_junctions_distribution.register(dict)
def _(junctions: dict) -> None:
    """
    Plots the distribution of junctions in the skeleton.

    Args:
        junctions (dict): A dictionary with the count of junctions for each number of neighbors.
    """

    plt.figure(figsize=(12, 12))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    junctions = {k: v for k, v in junctions.items() if k != "2"}
    x = np.array([i for i in junctions.keys()])
    y = np.array([j for j in junctions.values()])
    plt.bar(x, y, color=colors[:len(x)])
    plt.xlabel('Junction degree')
    plt.ylabel('Number of junctions')
    plt.title('Junctions in Skeleton')
    plt.show()

@plot_junctions_distribution.register(np.ndarray)
def _(skel: np.ndarray) -> None:
    junctions = count_junctions(skel)
    plot_junctions_distribution(junctions)


# %%
from PIL import Image

skel = (np.array(Image.open("/home/morand/afs/tests/01R_2ndHO.png"))[:,:,0] > 0).astype(int)

plot_junctions(skel)

junctions = count_junctions(skel)

print(junctions)

all_j = np.array([j for j in junctions.values()]).sum()
print(all_j)
print(skel.sum())

plot_junctions_distribution(skel)


# %%

def full_plot(skel: np.ndarray) -> None:
    """
    Plots the skeleton with junctions and the distribution of junctions.

    Args:
        skel (np.ndarray): The input binary skeleton.
    """

    fig, axes = plt.subplots(1, 2, figsize=(11, 9), gridspec_kw={'width_ratios': [1, 0.2]})

    neighbor_counts = compute_neighbors_count(skel)
    neighbor_range = np.array([i for i in np.unique(neighbor_counts) if i > 0])

    axes[0].imshow(skel, cmap='gray')
    axes[0].axis('off')
    for n in neighbor_range:
        if n == 2:
            continue
        y_coords, x_coords = np.where(neighbor_counts == n)
        if n == 1:
            axes[0].scatter(x_coords, y_coords, s=20, label=f'Extremities (1 neighbor)')
        else:
            axes[0].scatter(x_coords, y_coords, s=20, label=f'Junctions with {n} neighbors')
    axes[0].legend()

    junctions = count_junctions(skel)
    
    axes[1].set_title('Junctions Distribution')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    junctions = {k: v for k, v in junctions.items() if k != "2"}
    x = np.array([i for i in junctions.keys()])
    y = np.array([j for j in junctions.values()])
    last_y_non_zero = np.max(np.where(y != 0)[0])
    x = x[:last_y_non_zero + 1]
    y = y[:last_y_non_zero + 1]
    axes[1].bar(x, y, color=colors[:len(x)])
    axes[1].set_xlabel('Junction degree')
    axes[1].set_ylabel('Number of junctions')

    plt.tight_layout()
    plt.show()



# %%
full_plot(skel)

# %%
skel = (np.array(Image.open("/home/morand/afs/tests/skeletonBinMasked.png")) > 0).astype(int)

plot_junctions(skel)

junctions = count_junctions(skel)

print(junctions)

all_j = np.array([j for j in junctions.values()]).sum()
print(all_j)
print(skel.sum())

plot_junctions_distribution(skel)

# %%
from skimage.morphology import dilation, square

def compute_neighbors_regions(skel: np.ndarray) -> np.ndarray:
    """Compute the number of neighboring regions for each pixel in the skeleton.

    Args:
        skel (np.ndarray): The skeletonized image.

    Returns:
        np.ndarray: The skeleton with each pixel value being the number of neighboring pixels.
    """
    neighbors_regions = np.zeros_like(skel)
    height, width = skel.shape
    skel_p = np.pad(skel, pad_width=2, mode='constant', constant_values=0)

    for i in range(height):
        for j in range(width):
            if skel_p[i + 2, j + 2] == 0:
                continue

            # Get a 5-pixels wide square patch of pixels
            patch = skel_p[i:i+5, j:j+5]

            # Only keep the pixels that are connected to the center pixel
            cc = label(patch, connectivity=2)
            center_label = cc[2, 2]
            center_component = (cc == center_label)

            # Get only the 1-pixel wide band of pixels that constitute the edge of the 5-pixels square patch
            neighbor_5 = center_component.copy()
            neighbor_5[1:4, 1:4] = 0

            # Get the number of connected components on this band, to get the number of independant branches
            _, n_neighbors = label(neighbor_5, connectivity=2, return_num=True)

            neighbors_regions[i, j] = n_neighbors

    return neighbors_regions


def _clean_neighbors_regions(real_neighbors: np.ndarray) -> np.ndarray:
    """
    Cleans the regions of more-than-2 neighbors pixels by keeping only the pixel with the maximum
    number of neighbors in each region.

    Args:
        real_neighbors (np.ndarray): The skeleton with each pixel value being the number of neighbors.

    Returns:
        np.ndarray: The cleaned skeleton with only one pixel per region of more-than-2 neighbors pixels.
    """

    # Indentify regions of more-than-2 neighbors pixels
    junctions = real_neighbors > 2
    neighbors_cc = label(junctions, connectivity=2)

    simple_neighbors = np.zeros_like(real_neighbors)
    for region_label in np.unique(neighbors_cc):
        if region_label == 0:
            continue

        # Get the mask of the current region
        region_mask = (neighbors_cc == region_label)

        # Get the outer edge of the region
        region_mask_dilated = dilation(region_mask, square(3))
        region_mask_edge = region_mask_dilated & (~region_mask)

        # Compute the number of neighboring edges for the region
        region_vessel_neighbors = region_mask_edge * real_neighbors
        _, n_neighbors = label(region_vessel_neighbors, connectivity=2, return_num=True)

        # Set the pixel at the mean position of the region to the number of neighbors
        x, y = np.where(region_mask)
        x_mean, y_mean = np.mean(x).astype(int), np.mean(y).astype(int)
        simple_neighbors[(x_mean, y_mean)] = n_neighbors

    return simple_neighbors


def compute_new_neighbors_count(skel: np.ndarray) -> int:
    """
    Detects the number of neighbors for each pixel in the skeletonized image.
    WARNING: this function was not formally tested, and there are still some errors,
    so the result is an approximation of the real neighbor count of junctions pixels,
    especially with complex shaped skeletons like in ROSE dataset, or with skeletons
    that weren't simplified, e.g. with manual annotators.

    Args:
        skel (np.ndarray): The input binary skeleton

    Returns:
        np.ndarray: The skeleton with each pixel value being an approximation of its number of neighbors
    """

    if not isinstance(skel, np.ndarray):
        raise ValueError(f"Input skeleton must be a numpy array, got {type(skel)}")
    if skel.ndim != 2:
        raise ValueError(f"Input skeleton must be a 2D array, got {skel.ndim}D array of shape {skel.shape}")

    if skel.dtype == np.bool:
        warnings.warn("Input skeleton is of boolean type. It will be converted to integer type for processing.")
        skel = skel.astype(int)
    if set(np.unique(skel)) != {0, 1}:
        warnings.warn("Input skeleton is not binary. It will be thresholded at >0 for processing.")
        skel = (skel > 0).astype(int)
    if np.sum(skel) == 0:
        warnings.warn("Input skeleton is empty. Returning the original skeleton.")
        return skel

    # First, detect extremities to add them back later
    extremities = detect_extremities(skel)
    
    # Compute the number of neighbors for each pixel in the skeleton
    neighbors_regions = compute_neighbors_regions(skel)

    # Right now, there are some regions of more-than-2 neighbors pixels that are clamped together, we only need to keep
    # the pixel that has the maximum number of neighbors
    simple_neighbors = _clean_neighbors_regions(neighbors_regions)
    
    # Add back the extremities
    simple_neighbors = simple_neighbors + extremities

    # Add back the normal skeleton pixels (2 neighbors pixels)
    final_neighbor_count = skel * 2
    final_neighbor_count[simple_neighbors > 0] = simple_neighbors[simple_neighbors > 0]

    # return the final skeleton with each pixel value being an approximation of the number of neighbors
    return final_neighbor_count

def plot_new_junctions(skel: np.ndarray, neighbor_counts: dict = None) -> None:
    """
    Plots the junctions detected in the skeleton.

    Args:
        neighbor_counts (dict): A dictionary with the count of junctions for each number of neighbors
    """

    if neighbor_counts is None:
        neighbor_counts = compute_new_neighbors_count(skel)

    neighbor_range = np.array([i for i in np.unique(neighbor_counts) if i > 0])

    plt.figure(figsize=(12, 12))
    plt.imshow(skel, cmap='gray')
    plt.axis('off')
    for n in neighbor_range:
        if n == 2:
            continue
        y_coords, x_coords = np.where(neighbor_counts == n)
        plt.scatter(x_coords, y_coords, s=10, label=f'Junctions with {n} neighbors')
    plt.legend()
    plt.show()


# %%
from PIL import Image

skel = (np.array(Image.open("/home/morand/afs/tests/01R_2ndHO.png"))[:,:,0] > 0).astype(int)

plot_new_junctions(skel)

# %%
skel = (np.array(Image.open("/home/morand/afs/tests/skeletonBinMasked.png")) > 0).astype(int)

plt.figure(figsize=(12,12))
plt.imshow(skel)
plt.show()

plot_new_junctions(skel)

# %%
