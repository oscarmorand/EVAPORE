import torch
import matplotlib.pyplot as plt


def plot_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    plot_3d_mode: str = "mid_slice",
) -> None:
    """
    Display a batch of 2D or 3D images together with their ground truth.

    Supported shapes:
        2D:
            img: (B, H, W)
            gt:  (B, H, W)

        2D with channels:
            img: (B, C, H, W)
            gt:  (B, H, W) or (B, 1, H, W)

        3D:
            img: (B, H, W, D)
            gt:  (B, H, W, D)

        3D with channels:
            img: (B, C, H, W, D)
            gt:  (B, H, W, D) or (B, 1, H, W, D)

    Args:
        batch: Tuple containing (image, ground truth).
        plot_3d_mode:
            "mid_slice"  -> display the middle slice of each 3D volume.
            "all_slices" -> display every slice.
    """

    img, gt = batch[0], batch[1]

    if not isinstance(img, torch.Tensor) or not isinstance(gt, torch.Tensor):
        raise TypeError("img and gt must be torch.Tensor.")

    if img.ndim not in (3, 4, 5):
        raise ValueError(
            f"Unsupported image shape {tuple(img.shape)}. "
            "Expected 3D, 4D or 5D tensor."
        )

    if plot_3d_mode not in ("mid_slice", "all_slices"):
        raise ValueError(
            f"Unknown plot_3d_mode: {plot_3d_mode}. "
            'Expected "mid_slice" or "all_slices".'
        )

    # ------------------------------------------------------------------
    # Normalize shapes to:
    #   img -> (B, H, W)       for 2D
    #   img -> (B, H, W, D)    for 3D
    # ------------------------------------------------------------------

    # 2D batch: (B, H, W)
    if img.ndim == 3:
        is_3d = False

    # Could be:
    #   (B, C, H, W)  -> 2D
    #   (B, H, W, D)  -> 3D
    elif img.ndim == 4:
        # Assume a channel dimension if the second dimension is small.
        if img.shape[1] <= 4:
            img = img[:, 0]
            is_3d = False
        else:
            is_3d = True

    # (B, C, H, W, D)
    elif img.ndim == 5:
        img = img[:, 0]
        is_3d = True

    # ------------------------------------------------------------------
    # Normalize GT
    # ------------------------------------------------------------------

    if gt.ndim == img.ndim + 1 and gt.shape[1] == 1:
        gt = gt[:, 0]

    if gt.shape != img.shape:
        raise ValueError(
            f"Image and GT shapes do not match after normalization: "
            f"{tuple(img.shape)} vs {tuple(gt.shape)}"
        )

    batch_size = img.shape[0]

    # ------------------------------------------------------------------
    # 2D
    # ------------------------------------------------------------------

    if not is_3d:
        fig, axes = plt.subplots(
            batch_size,
            2,
            figsize=(8, 4 * batch_size),
            squeeze=False,
        )

        for i in range(batch_size):
            axes[i, 0].imshow(
                img[i].detach().cpu().numpy(),
                cmap="gray",
            )
            axes[i, 0].set_title(f"Image {i}")

            axes[i, 1].imshow(
                gt[i].detach().cpu().numpy(),
                cmap="gray",
            )
            axes[i, 1].set_title(f"Ground truth {i}")

            axes[i, 0].axis("off")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()
        return

    # ------------------------------------------------------------------
    # 3D
    # ------------------------------------------------------------------

    depth = img.shape[-1]

    if plot_3d_mode == "mid_slice":
        slice_indices = [depth // 2]
    else:
        slice_indices = range(depth)

    for z in slice_indices:
        fig, axes = plt.subplots(
            batch_size,
            2,
            figsize=(8, 4 * batch_size),
            squeeze=False,
        )

        for i in range(batch_size):
            axes[i, 0].imshow(
                img[i, :, :, z].detach().cpu().numpy(),
                cmap="gray",
            )
            axes[i, 0].set_title(f"Image {i} — slice {z}/{depth}")

            axes[i, 1].imshow(
                gt[i, :, :, z].detach().cpu().numpy(),
                cmap="gray",
            )
            axes[i, 1].set_title(f"Ground truth {i} — slice {z}/{depth}")

            axes[i, 0].axis("off")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()
