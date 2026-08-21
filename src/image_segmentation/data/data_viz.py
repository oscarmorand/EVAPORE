import matplotlib.pyplot as plt
import torch


def plot_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    ndim: int,
    pred: torch.Tensor | None = None,
    plot_3d_mode: str = "mid_slice",
) -> None:
    """
    Display a batch of 2D or 3D images together with their ground truth,
    and optionally the model's predicted (binary) mask.

    Supported shapes:
        2D:
            img:  (B, H, W) or (B, 1, H, W)          -> grayscale
                  (B, 3, H, W)                        -> RGB
            gt:   (B, H, W) or (B, 1, H, W)
            pred: (B, H, W) or (B, 1, H, W)

        3D:
            img:  (B, H, W, D) or (B, C, H, W, D)
            gt:   (B, H, W, D) or (B, 1, H, W, D)
            pred: (B, H, W, D) or (B, 1, H, W, D)

    Args:
        batch: Tuple containing (image, ground truth).
        ndim: Spatial dimensionality of the data — 2 or 3.
        pred: Optional model output, treated as a binary mask. Same
            accepted shapes as gt. When provided, an extra column is
            added to the figure showing the predicted mask.
        plot_3d_mode:
            "mid_slice"  -> display the middle slice of each 3D volume.
            "all_slices" -> display every slice.
    """

    img, gt = batch[0], batch[1]

    if not isinstance(img, torch.Tensor) or not isinstance(gt, torch.Tensor):
        raise TypeError("img and gt must be torch.Tensor.")

    if pred is not None and not isinstance(pred, torch.Tensor):
        raise TypeError("pred must be a torch.Tensor.")

    if ndim not in (2, 3):
        raise ValueError(f"Unsupported ndim: {ndim}. Expected 2 or 3.")

    if plot_3d_mode not in ("mid_slice", "all_slices"):
        raise ValueError(
            f"Unknown plot_3d_mode: {plot_3d_mode}. "
            'Expected "mid_slice" or "all_slices".'
        )

    is_3d = ndim == 3
    is_rgb = False

    # ------------------------------------------------------------------
    # Normalize image to:
    #   img -> (B, H, W)       for 2D grayscale
    #   img -> (B, H, W, 3)    for 2D RGB
    #   img -> (B, H, W, D)    for 3D
    # ------------------------------------------------------------------

    if ndim == 2:
        if img.ndim == 3:  # (B, H, W)
            pass
        elif img.ndim == 4:  # (B, C, H, W)
            n_channels = img.shape[1]
            if n_channels == 1:
                img = img[:, 0]
            elif n_channels == 3:
                is_rgb = True
                img = img.permute(0, 2, 3, 1)  # (B, H, W, 3)
            else:
                raise ValueError(
                    f"Unsupported channel count {n_channels} for ndim=2. "
                    "Expected 1 (grayscale) or 3 (RGB)."
                )
        else:
            raise ValueError(
                f"Unsupported image shape {tuple(img.shape)} for ndim=2. "
                "Expected (B, H, W) or (B, C, H, W) with C in {1, 3}."
            )
    else:  # ndim == 3
        if img.ndim == 4:  # (B, H, W, D)
            pass
        elif img.ndim == 5:  # (B, C, H, W, D)
            img = img[:, 0]
        else:
            raise ValueError(
                f"Unsupported image shape {tuple(img.shape)} for ndim=3. "
                "Expected (B, H, W, D) or (B, C, H, W, D)."
            )

    # ------------------------------------------------------------------
    # Normalize a mask-like tensor (GT or pred) to:
    #   (B, H, W)       for 2D
    #   (B, H, W, D)    for 3D
    # (independent of whether img was permuted for RGB)
    # ------------------------------------------------------------------

    def _normalize_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
        if ndim == 2:
            if mask.ndim == 3:  # (B, H, W)
                return mask
            if mask.ndim == 4 and mask.shape[1] == 1:  # (B, 1, H, W)
                return mask[:, 0]
            raise ValueError(
                f"Unsupported {name} shape {tuple(mask.shape)} for ndim=2. "
                "Expected (B, H, W) or (B, 1, H, W)."
            )
        else:  # ndim == 3
            if mask.ndim == 4:  # (B, H, W, D)
                return mask
            if mask.ndim == 5 and mask.shape[1] == 1:  # (B, 1, H, W, D)
                return mask[:, 0]
            raise ValueError(
                f"Unsupported {name} shape {tuple(mask.shape)} for ndim=3. "
                "Expected (B, H, W, D) or (B, 1, H, W, D)."
            )

    gt = _normalize_mask(gt, "GT")
    if pred is not None:
        pred = _normalize_mask(pred, "pred")

    expected_shape = img.shape[:-1] if is_rgb else img.shape
    if gt.shape != expected_shape:
        raise ValueError(
            f"Image and GT shapes do not match after normalization: "
            f"{tuple(img.shape)} vs {tuple(gt.shape)}"
        )
    if pred is not None and pred.shape != expected_shape:
        raise ValueError(
            f"Image and pred shapes do not match after normalization: "
            f"{tuple(img.shape)} vs {tuple(pred.shape)}"
        )

    batch_size = img.shape[0]
    n_cols = 3 if pred is not None else 2

    # ------------------------------------------------------------------
    # 2D
    # ------------------------------------------------------------------

    if not is_3d:
        fig, axes = plt.subplots(
            batch_size,
            n_cols,
            figsize=(4 * n_cols, 4 * batch_size),
            squeeze=False,
        )

        for i in range(batch_size):
            axes[i, 0].imshow(
                img[i].detach().cpu().numpy(),
                cmap=None if is_rgb else "gray",
            )
            axes[i, 0].set_title(f"Image {i}")

            axes[i, 1].imshow(
                gt[i].detach().cpu().numpy(),
                cmap="gray",
            )
            axes[i, 1].set_title(f"Ground truth {i}")

            axes[i, 0].axis("off")
            axes[i, 1].axis("off")

            if pred is not None:
                axes[i, 2].imshow(
                    pred[i].detach().cpu().numpy(),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axes[i, 2].set_title(f"Prediction {i}")
                axes[i, 2].axis("off")

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