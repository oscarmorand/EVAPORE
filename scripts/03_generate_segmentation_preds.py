"""
CLI entry point for running inference with a trained UNet segmentation model.

Usage:
    # Predict with the single provided pretrained checkpoint
    python predict.py --pretrained

    # Predict with your own most recently trained run
    python predict.py

    # Predict with a specific named training run
    python predict.py --run-name my_run

    # Debug mode: save predictions inside the checkpoint's own folder instead
    # of next to the dataset (<data_dir>/pred)
    python predict.py --run-name my_run --debug

    # Use a different dataset (must be a key in available_datasets)
    python predict.py --dataset SOME_OTHER_DATASET
"""
import argparse
import gc
import logging
import os

import torch
from tqdm import tqdm

from image_segmentation.data.augmentations import build_val_transform
from image_segmentation.data.image_datamodule import ImageDatamodule
from image_segmentation.data.io_utils import save_array
from image_segmentation.models import BinarySegmentator
from utils.cli_common import (
    TRAINING_BASE_DIR,
    find_checkpoint_in_dir,
    resolve_dataset,
    resolve_run_dir,
    setup_logging,
)
from utils.device import get_device

logger = logging.getLogger(__name__)

# Directory holding the single, externally-provided pretrained checkpoint
# (as opposed to TRAINING_BASE_DIR, which holds your own training runs).
PRETRAINED_DIR = "unet_pretrained"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_inference_dataset(dataset_choice):
    """
    Normalization stats are computed from the training split (matching how
    the model was trained), then applied as val_transforms over the dataset
    used for prediction.
    """
    train_split = dataset_choice.preferred_train_split
    data_dir = dataset_choice.data_dir
    splits_filepath = dataset_choice.splits_filepath
    ndim = dataset_choice.ndim
    input_channels = dataset_choice.input_channels

    datamodule = ImageDatamodule(
        data_dir=data_dir,
        split_file_path=splits_filepath,
        train_split_name=train_split,
        num_workers=0,
        val_batch_size=1,
        shuffle_train=False,
    )
    datamodule.setup()

    stats = datamodule.dataset.get_dataset_stats(
        ndim=ndim,
        input_channels=input_channels,
        split_name=train_split,
        split_indices=datamodule.train_indices,
    )
    masked = "foreground" if "foreground" in stats else "full_image"
    mean, std = stats[masked]["mean"], stats[masked]["std"]
    logger.info(f"Dataset stats ({masked}): mean={mean}, std={std}")

    val_transforms = build_val_transform(ndim, mean, std)
    datamodule.test_transforms = val_transforms
    datamodule.setup()

    dataset = datamodule.dataset
    dataset.transforms = val_transforms
    return dataset


# --------------------------------------------------------------------------- #
# Checkpoint / output resolution
# --------------------------------------------------------------------------- #
def resolve_checkpoint_and_source_dir(dataset_choice, use_pretrained, run_name):
    """
    Returns (ckpt_path, source_dir).
    - use_pretrained=True: the single checkpoint under PRETRAINED_DIR.
    - use_pretrained=False: the checkpoint of one of your own training runs
      under TRAINING_BASE_DIR -- either --run-name, or the most recent run.
    source_dir is returned too since --debug saves predictions next to it.
    """
    if use_pretrained:
        if not os.path.isdir(PRETRAINED_DIR):
            raise FileNotFoundError(f"Pretrained model directory not found: {PRETRAINED_DIR}")
        return find_checkpoint_in_dir(PRETRAINED_DIR), PRETRAINED_DIR

    run_dir = resolve_run_dir(dataset_choice, TRAINING_BASE_DIR, run_name)
    return find_checkpoint_in_dir(run_dir), run_dir


def resolve_pred_dir(data_dir, source_dir, debug):
    """
    Non-debug (default): predictions are the "final" output, saved next to
    the dataset itself at <data_dir>/pred.
    Debug: predictions are saved inside the checkpoint's own folder at
    <source_dir>/pred, keeping exploratory runs separate from the dataset.
    """
    if debug:
        return os.path.join(source_dir, "pred")
    return os.path.join(data_dir, "pred")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained UNet segmentation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", type=str, default="PERSEVERE",
        help="Name of the dataset to use, must be a key of available_datasets (default: PERSEVERE)",
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--pretrained", action="store_true",
        help=f"Use the single provided pretrained checkpoint under '{PRETRAINED_DIR}/' "
             f"instead of one of your own training runs",
    )
    source_group.add_argument(
        "--run-name", type=str, default=None,
        help="Name of your own trained run to use (default: most recent run)",
    )

    parser.add_argument(
        "--debug", action="store_true",
        help="Save predictions inside the checkpoint's own folder (<source>/pred) instead of "
             "next to the dataset (<data_dir>/pred)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_choice = resolve_dataset(args.dataset)
    device = get_device()

    ckpt_path, source_dir = resolve_checkpoint_and_source_dir(
        dataset_choice, args.pretrained, args.run_name
    )

    pred_dir = resolve_pred_dir(dataset_choice.data_dir, source_dir, args.debug)
    os.makedirs(pred_dir, exist_ok=True)

    setup_logging(pred_dir, log_filename="predict.log")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Predictions directory: {pred_dir}")

    dataset = build_inference_dataset(dataset_choice)

    model = BinarySegmentator.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()

    with torch.no_grad():
        for i, img_path in enumerate(tqdm(dataset.img_paths)):
            img, _ = dataset[i]
            img = img.unsqueeze(0).float().to(device)

            logits = model._predict(img)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            pred = (probs > 0.5).astype("uint8") * 255

            base_name = os.path.basename(img_path)
            pred_path = os.path.join(pred_dir, base_name)
            save_array(pred, pred_path)

            del logits, probs, pred, img

    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Predictions saved to {pred_dir}, total {len(os.listdir(pred_dir))} images.")


if __name__ == "__main__":
    main()