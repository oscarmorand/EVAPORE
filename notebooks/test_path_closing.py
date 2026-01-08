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
import numpy as np
from PIL import Image

# %%
img = Image.open('/home/morand/afs/EVAPORE/data/FIVES/img/FIVES_001.png')
img_array = np.array(img)

print("Image shape:", img_array.shape)

# %%
pred = Image.open('/home/morand/afs/EVAPORE/data/FIVES/pred/FIVES_001.png').convert('L')
pred_array = np.array(pred)

print("Prediction shape:", pred_array.shape)

# %%
import matplotlib.pyplot as plt

plt.imshow(pred_array, cmap='gray')
plt.title('Prediction Mask')
plt.colorbar()
plt.show()

test = pred_array[1350:1500, 1250:1400]

plt.imshow(test, cmap='gray')
plt.title('Cropped Prediction Mask')
plt.colorbar()
plt.show()


# %%
def path_closing(mask, iterations=1, connectivity=8):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if connectivity == 8:
        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    closed = np.zeros_like(mask)
    for direction in directions:
        dy, dx = direction
        directed_dilated_mask = mask.copy()
        plt.imshow(directed_dilated_mask, cmap='gray')
        plt.title(f'Directed Dilation: direction {direction}, before first iteration')
        plt.colorbar()
        plt.show()
        for i in range(iterations):
            temp = np.zeros_like(mask)
            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                        temp[y, x] = max(directed_dilated_mask[y, x], directed_dilated_mask[ny, nx])
                    else:
                        temp[y, x] = directed_dilated_mask[y, x]
            directed_dilated_mask = temp
        plt.imshow(directed_dilated_mask, cmap='gray')
        plt.title(f'Directed Dilation: direction {direction}, final iteration')
        plt.colorbar()
        plt.show()

        for i in range(iterations):
            temp = np.zeros_like(mask)
            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    ny, nx = y - dy, x - dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                        temp[y, x] = min(directed_dilated_mask[y, x], directed_dilated_mask[ny, nx])
                    else:
                        temp[y, x] = directed_dilated_mask[y, x]
            directed_dilated_mask = temp
        plt.imshow(directed_dilated_mask, cmap='gray')
        plt.title(f'Directed Erosion: direction {direction}, final iteration')
        plt.colorbar()
        plt.show()

        closed = np.maximum(closed, directed_dilated_mask)

    return closed


# %%
closed = path_closing(test, iterations=10, connectivity=8)

# %%
plt.imshow(closed, cmap='gray')
plt.title('Closed Mask')
plt.colorbar()
plt.show()

# %%
closed = path_closing(pred_array, iterations=10, connectivity=8)


# %%
def get_directions_params(mask, direction):
    dy, dx = direction
    min_y, min_x, max_y, max_x = 0, 0, mask.shape[0], mask.shape[1]
    if dy == 1:
        max_y -= 1
    elif dy == -1:
        min_y += 1
    
    if dx == 1:
        max_x -= 1
    elif dx == -1:
        min_x += 1

    return dy, dx, min_y, min_x, max_y, max_x

def directed_erosion(mask, direction):
    dy, dx, min_y, min_x, max_y, max_x = get_directions_params(mask, direction)

    eroded_mask = np.zeros_like(mask)
    eroded_mask[min_y:max_y, min_x:max_x] = np.minimum(
        mask[min_y:max_y, min_x:max_x],
        mask[min_y + dy:max_y + dy, min_x + dx:max_x + dx]
    )
    eroded_mask[0:min_y, :] = mask[0:min_y, :]
    eroded_mask[max_y:, :] = mask[max_y:, :]
    eroded_mask[:, 0:min_x] = mask[:, 0:min_x]
    eroded_mask[:, max_x:] = mask[:, max_x:]
    return eroded_mask

def directed_dilation(mask, direction):
    dy, dx, min_y, min_x, max_y, max_x = get_directions_params(mask, direction)
    print(f"Direction: {direction}, min_y: {min_y}, min_x: {min_x}, max_y: {max_y}, max_x: {max_x}")

    dilated_mask = np.zeros_like(mask)
    dilated_mask[min_y:max_y, min_x:max_x] = np.maximum(
        mask[min_y:max_y, min_x:max_x],
        mask[min_y + dy:max_y + dy, min_x + dx:max_x + dx]
    )
    dilated_mask[0:min_y, :] = mask[0:min_y, :]
    dilated_mask[max_y:, :] = mask[max_y:, :]
    dilated_mask[:, 0:min_x] = mask[:, 0:min_x]
    dilated_mask[:, max_x:] = mask[:, max_x:]
    return dilated_mask

def path_closing(mask, iterations=1, connectivity=8):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if connectivity == 8:
        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    closed = np.zeros_like(mask)
    for direction in directions:
        directed_dilated_mask = mask.copy()

        plt.imshow(directed_dilated_mask, cmap='gray')
        plt.title(f'Directed Dilation: direction {direction}, before first iteration')
        plt.colorbar()
        plt.show()
        
        for i in range(iterations):
            directed_dilated_mask = directed_dilation(directed_dilated_mask, direction)

        opposite_direction = (-direction[0], -direction[1])
        directed_eroded_mask = directed_dilated_mask.copy()
        for i in range(iterations):
            directed_eroded_mask = directed_erosion(directed_eroded_mask, opposite_direction)

        plt.imshow(directed_eroded_mask, cmap='gray')
        plt.title(f'Directed Erosion: direction {direction}, final iteration')
        plt.colorbar()
        plt.show()

        closed = np.maximum(closed, directed_eroded_mask)

    return closed


# %%
closed = path_closing(test, iterations=10, connectivity=8)
plt.imshow(closed, cmap='gray')
plt.title('Closed Mask')
plt.colorbar()
plt.show()

# %%
closed = path_closing(pred_array, iterations=10, connectivity=8)
plt.figure(figsize=(40,40))
plt.imshow(closed, cmap='gray')
plt.title('Closed Full Prediction Mask')
plt.colorbar()
plt.show()

# %%
diff = closed - pred_array

plt.figure(figsize=(40,40))
plt.imshow(diff, cmap='gray')
plt.title('Difference between Closed Mask and Original Prediction')
plt.colorbar()
plt.show()

# %%
from skimage.measure import label

labeled_diff, n_label = label(diff, return_num=True)
print(f'Number of connected components in the difference: {n_label}')

# %%
plt.imshow(labeled_diff, cmap='nipy_spectral')
plt.title('Labeled Connected Components in Difference')
plt.colorbar()
plt.show()

# %%
from skimage.morphology import remove_small_objects

true_components = remove_small_objects(labeled_diff, min_size=10, connectivity=1)

# %%
plt.figure(figsize=(40,40))
plt.imshow(true_components, cmap='nipy_spectral')
plt.title('True Connected Components in Difference (size > 10)')
plt.colorbar()
plt.show()


# %%
def betti_0_2D(mask: np.ndarray) -> int:
        '''
        Compute the 0th Betti number (number of connected components) for a 2D binary mask.

        Parameters:
            mask (np.ndarray): 2D binary mask where foreground pixels are True.

        Returns:
            int: The 0th Betti number (number of connected components).
        '''
        if mask.ndim != 2:
            raise ValueError("Input mask must be a 2D array.")
        if mask.dtype != np.bool:
            mask = (mask > 0)

        # Label connected components in the inverted mask, using 8-connectivity
        _, num_cc = label(mask, return_num=True, connectivity=2)
        return num_cc

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
        if mask.dtype != np.bool:
            mask = (mask > 0)

        # Invert the mask to find holes
        inverse_mask = np.logical_not(mask)
        # Label connected components in the inverted mask, using 4-connectivity
        _, num_cc = label(inverse_mask, return_num=True, connectivity=1)
        # Subtract 1 to exclude the outer background component
        return num_cc - 1


# %%
base_betti_0 = betti_0_2D(pred_array)
print(f'Base Betti-0 (number of connected components) of original prediction: {base_betti_0}')
base_betti_1 = betti_1_2D(pred_array)
print(f'Base Betti-1 (number of holes) of original prediction: {base_betti_1}')

nothing_components = np.zeros_like(true_components)
closing_components = np.zeros_like(true_components)
closing_cycles_components = np.zeros_like(true_components)
filling_components = np.zeros_like(true_components)

new_components = np.unique_values(true_components)
for component in new_components:
    if component == 0:
        continue
    component_mask = (true_components == component)
    array_with_component = pred_array + component_mask.astype(np.uint8) * 255
    component_betti_0 = betti_0_2D(array_with_component)
    component_betti_1 = betti_1_2D(array_with_component)
    diff_betti_0 = component_betti_0 - base_betti_0
    diff_betti_1 = component_betti_1 - base_betti_1
    if diff_betti_0 == 0 and diff_betti_1 == 0:
        nothing_components += component_mask.astype(np.uint8) * 255
    if diff_betti_0 < 0:
        closing_components += component_mask.astype(np.uint8) * 255
    if diff_betti_1 > 0:
        closing_cycles_components += component_mask.astype(np.uint8) * 255
    if diff_betti_1 < 0:
        filling_components += component_mask.astype(np.uint8) * 255

    print(f'Component {component}: ΔBetti-0 = {diff_betti_0}, ΔBetti-1 = {diff_betti_1}')

img = np.zeros((pred_array.shape[0], pred_array.shape[1], 3), dtype=np.uint8)
img[..., 0] = pred_array + closing_components
img[..., 1] = pred_array + closing_cycles_components + nothing_components
img[..., 2] = pred_array + filling_components + nothing_components

plt.figure(figsize=(40,40))
plt.imshow(img)
plt.title('Original Prediction (Green), Closed Components (Red)')
plt.show()

# %%
from skimage.morphology import binary_closing, disk

def basic_closing(mask, iterations=1):
    mask = (mask > 0)
    closed_mask = binary_closing(mask, footprint=disk(iterations))
    return closed_mask.astype(np.uint8) * 255


# %%
basic_closed = basic_closing(pred_array, iterations=10)

# %%
plt.figure(figsize=(40,40))
plt.imshow(basic_closed, cmap='gray')
plt.title('Basic Closed Full Prediction Mask')
plt.colorbar()
plt.show()

# %%
diff_basic = basic_closed - pred_array

plt.figure(figsize=(40,40))
plt.imshow(diff_basic, cmap='gray')
plt.title('Difference between Basic Closed Mask and Original Prediction')
plt.colorbar()
plt.show()

# %%
labeled_diff = label(diff_basic)
true_components = remove_small_objects(labeled_diff, min_size=10, connectivity=1)

plt.figure(figsize=(40,40))
plt.imshow(true_components, cmap='nipy_spectral')
plt.title('True Connected Components in Basic Closing Difference (size > 10)')
plt.colorbar()
plt.show()

# %%
nothing_components = np.zeros_like(true_components)
closing_components = np.zeros_like(true_components)
closing_cycles_components = np.zeros_like(true_components)
filling_components = np.zeros_like(true_components)

new_components = np.unique_values(true_components)
for component in new_components:
    if component == 0:
        continue
    component_mask = (true_components == component)
    array_with_component = pred_array + component_mask.astype(np.uint8) * 255
    component_betti_0 = betti_0_2D(array_with_component)
    component_betti_1 = betti_1_2D(array_with_component)
    diff_betti_0 = component_betti_0 - base_betti_0
    diff_betti_1 = component_betti_1 - base_betti_1
    if diff_betti_0 == 0 and diff_betti_1 == 0:
        nothing_components += component_mask.astype(np.uint8) * 255
    if diff_betti_0 < 0:
        closing_components += component_mask.astype(np.uint8) * 255
    if diff_betti_1 > 0:
        closing_cycles_components += component_mask.astype(np.uint8) * 255
    if diff_betti_1 < 0:
        filling_components += component_mask.astype(np.uint8) * 255

    print(f'Component {component}: ΔBetti-0 = {diff_betti_0}, ΔBetti-1 = {diff_betti_1}')

img = np.zeros((pred_array.shape[0], pred_array.shape[1], 3), dtype=np.uint8)
img[..., 0] = pred_array + closing_components
img[..., 1] = pred_array + closing_cycles_components + nothing_components
img[..., 2] = pred_array + filling_components + nothing_components

# %%
plt.figure(figsize=(40,40))
plt.imshow(img)
plt.title('Original Prediction (Green), Basic Closed Components (Red)')
plt.show()


# %%
def min_betti_0_closing(mask: np.ndarray, iterations=1, connectivity=1, min_size=10) -> int:
    base_betti_0 = betti_0_2D(mask)

    bin_mask = (mask > 0)
    closed_mask = binary_closing(bin_mask, footprint=disk(iterations))
    closed_mask = closed_mask.astype(np.uint8) * 255

    diff = closed_mask - mask
    labeled_diff = label(diff)
    true_components = remove_small_objects(labeled_diff, min_size=min_size, connectivity=connectivity)

    final_mask = mask.copy()
    for component in np.unique(true_components):
        if component == 0:
            continue
        component_mask = (true_components == component)
        array_with_component = final_mask + component_mask.astype(np.uint8) * 255

        component_betti_0 = betti_0_2D(array_with_component)
        diff_betti_0 = component_betti_0 - base_betti_0

        if diff_betti_0 < 0:
            final_mask += component_mask.astype(np.uint8) * 255
            base_betti_0 = component_betti_0

    return final_mask


# %%
plt.figure(figsize=(40,40))
plt.imshow(pred_array, cmap='gray')
plt.title('Original Prediction Mask')
plt.colorbar()
plt.show()

min_betti_0_closed = min_betti_0_closing(pred_array, iterations=10, connectivity=1, min_size=10)

plt.figure(figsize=(40,40))
plt.imshow(min_betti_0_closed, cmap='gray')
plt.title('Min Betti-0 Closed Full Prediction Mask')
plt.colorbar()
plt.show()

# %%
diff = min_betti_0_closed - pred_array

img = np.zeros((pred_array.shape[0], pred_array.shape[1], 3), dtype=np.uint8)
img[..., 0] = pred_array + diff
img[..., 1] = pred_array
img[..., 2] = pred_array

plt.figure(figsize=(40,40))
plt.imshow(img)
plt.title('Difference between Min Betti-0 Closed Mask and Original Prediction')
plt.colorbar()
plt.show()

# %%
