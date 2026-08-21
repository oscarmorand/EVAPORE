import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.measure import label

def compute_foreground_mask(img_paths: list[str],
                            threshold: int = 4,
                            debug_i: int = 20
) -> np.ndarray:

    n_elements = len(img_paths)
    if debug_i > 0:
        n_elements = debug_i

    non_zero_masks = []
    for i in tqdm(range(n_elements)):
        img_path = img_paths[i]

        img_gray = Image.open(img_path).convert("L")
        img_gray = np.array(img_gray)
        non_zero_mask = img_gray > threshold

        cc, num_cc = label(non_zero_mask, return_num=True, connectivity=2)
        cc_sizes = [np.sum(cc == i) for i in range(1, num_cc + 1)]
        max_cc_index = np.argmax(cc_sizes) + 1
        non_zero_mask = cc == max_cc_index

        non_zero_masks.append(non_zero_mask)

    common_mask = np.logical_and.reduce(non_zero_masks)
    return common_mask