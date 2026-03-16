import torch
import numpy as np
from skimage.measure import label
from skimage.morphology import closing, disk, remove_small_objects
    
class ImageLevelReconnectionWithClosing:
    def __init__(self) -> None:
        pass
    
    def betti_0_2D(self, mask: np.ndarray) -> int:
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

    def min_betti_0_closing(self, mask: np.ndarray, iterations=1, connectivity=1, min_size=10) -> int:
        base_betti_0 = self.betti_0_2D(mask)

        bin_mask = (mask > 0)
        closed_mask = closing(bin_mask, footprint=disk(iterations))
        closed_mask = closed_mask.astype(np.uint8) * 255

        diff = closed_mask - mask
        labeled_diff = label(diff)
        true_components = remove_small_objects(labeled_diff, max_size=min_size, connectivity=connectivity)

        final_mask = (mask > 0).astype(np.bool)

        final_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        final_image[:, :, 0] = final_mask.astype(np.uint8) * 255
        final_image[:, :, 1] = final_image[:, :, 0]
        final_image[:, :, 2] = final_image[:, :, 0]

        for component in np.unique(true_components):
            if component == 0:
                continue
            component_mask = (true_components == component)
            array_with_component = final_mask + component_mask.astype(np.uint8) * 255

            component_betti_0 = self.betti_0_2D(array_with_component)
            diff_betti_0 = component_betti_0 - base_betti_0

            if diff_betti_0 < 0:
                final_mask = np.logical_or(final_mask, component_mask)
                final_image[:, :, 0][component_mask] = 255  # Red
                base_betti_0 = component_betti_0

        return final_image, final_mask

    def predict_step(self, 
                     batch: list[tuple[torch.Tensor, torch.Tensor]]):
        """Perform a prediction step on a batch of data.

        Returns:
            A tensor containing the predicted masks after applying Betti-0 minimizing closing.
        """
        idx, data = batch
        batch_size = data.shape[0]
        if batch_size != 1:
            raise NotImplementedError("Batch size greater than 1 is not supported.")
        mask = data[0].numpy()
        closed_image, closed_mask = self.min_betti_0_closing(mask, iterations=10, connectivity=1, min_size=10)
        closed_image = torch.tensor(closed_image)
        closed_mask = torch.tensor(closed_mask)
        return closed_image, closed_mask