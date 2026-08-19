import torch
import matplotlib.pyplot as plt


def plot_batch(
    batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """
    Display a batch of 2D images together with their ground truth.

    Supported shapes:
        2D:
            img: (B, H, W)
            gt:  (B, H, W)

        2D with channels:
            img: (B, C, H, W)
            gt:  (B, H, W) or (B, 1, H, W)

    Args:
        batch: Tuple containing (image, ground truth).
    """

    img, gt = batch[0], batch[1]

    if not isinstance(img, torch.Tensor) or not isinstance(gt, torch.Tensor):
        raise TypeError("img and gt must be torch.Tensor.")

    if img.ndim not in (3, 4):
        raise ValueError(
            f"Unsupported image shape {tuple(img.shape)}. "
            "Expected 3D, or 4D"
        )

    if gt.shape[1] == 1:
        gt = gt[:, 0]

    batch_size = img.shape[0]

    fig, axes = plt.subplots(
        batch_size,
        2,
        figsize=(8, 4 * batch_size),
        squeeze=False,
    )

    for i in range(batch_size):
        axes[i, 0].imshow(
            img[i].detach().cpu().numpy().transpose(1, 2, 0),
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