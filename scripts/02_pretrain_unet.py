"""
CLI entry point for training/resuming/testing the UNet segmentation model.

Usage:
    # Train from scratch, auto-named run dir (timestamp)
    python train.py train

    # Train from scratch with a custom run name, then test at the end
    python train.py train --run-name my_first_run --test

    # Resume the most recent run into a new, auto-named run directory
    python train.py resume

    # Resume a specific run by name, into a new named run directory, then test
    python train.py resume --from-run my_first_run --run-name my_first_run_continued --test

    # Test a specific previous run
    python train.py test --run-name my_first_run

    # Test the most recent run
    python train.py test

    # Use a different dataset (must be a key in available_datasets)
    python train.py train --dataset SOME_OTHER_DATASET
"""
import argparse
import csv
import glob
import logging
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from image_segmentation.data.augmentations import (
    build_empty_transform,
    build_train_transform,
    build_val_transform,
)
from image_segmentation.data.image_datamodule import ImageDatamodule
from image_segmentation.models import BinarySegmentator
from image_segmentation.models.callbacks import PlotMetricsCallback, SaveConfigCallback
from utils.cli_common import (
    TRAINING_BASE_DIR as BASE_DIR,
    find_checkpoint_in_dir,
    resolve_dataset,
    resolve_run_dir,
    setup_logging,
)
from utils.device import get_device

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metrics-history continuity (resume)
# --------------------------------------------------------------------------- #
def find_latest_metrics_csv(run_dir):
    """
    Find the metrics.csv of the most recent CSVLogger version under run_dir
    (CSVLogger writes to run_dir/lightning_logs/version_x/metrics.csv,
    auto-incrementing version_x on every new Trainer/logger instance).
    Returns None if none exist yet (e.g. first-ever training run).
    """
    candidates = glob.glob(os.path.join(run_dir, "lightning_logs", "version_*", "metrics.csv"))
    if not candidates:
        return None

    def version_num(path):
        version_dir = os.path.basename(os.path.dirname(path))
        try:
            return int(version_dir.split("_")[-1])
        except ValueError:
            return -1

    candidates.sort(key=version_num)
    return candidates[-1]


def seed_csv_logger_history(csv_logger, old_metrics_path):
    """
    Preload a freshly created CSVLogger's in-memory metrics buffer with the
    rows from a previous run's metrics.csv, so the *next* time it writes to
    disk the file contains old + new rows together. CSVLogger doesn't read
    an existing metrics.csv back in on its own -- on resume it starts a new
    version_x directory with an empty in-memory list and would otherwise
    overwrite/produce a metrics.csv that starts again from epoch 0.
    """
    if old_metrics_path is None or not os.path.isfile(old_metrics_path):
        return
    with open(old_metrics_path, newline="") as f:
        old_rows = list(csv.DictReader(f))
    if not old_rows:
        return
    # Accessing .experiment creates the underlying writer (and its log dir)
    # if it doesn't exist yet; at this point it's freshly created and empty.
    csv_logger.experiment.metrics = old_rows + list(csv_logger.experiment.metrics)
    logger.info(f"Seeded metrics logger with {len(old_rows)} rows from {old_metrics_path}")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_datamodule(dataset_choice, model_config):
    """
    Two-step datamodule build: first with empty transforms to compute
    normalization stats, then the real datamodule using those stats.
    """
    train_split = dataset_choice.preferred_train_split
    data_dir = dataset_choice.data_dir
    splits_filepath = dataset_choice.splits_filepath
    ndim = dataset_choice.ndim
    input_channels = dataset_choice.input_channels

    empty_transforms = build_empty_transform(ndim)
    stats_datamodule = ImageDatamodule(
        data_dir=data_dir,
        split_file_path=splits_filepath,
        train_split_name=train_split,
        val_split_ratio=0.2,
        train_transforms=empty_transforms,
        val_transforms=empty_transforms,
        test_transforms=empty_transforms,
        num_workers=0,
        train_batch_size=1,
        val_batch_size=1,
        seed=42,
        shuffle_train=False,
    )
    stats_datamodule.setup()

    stats = stats_datamodule.dataset.get_dataset_stats(
        ndim=ndim,
        input_channels=input_channels,
        split_name=train_split,
        split_indices=stats_datamodule.train_indices,
    )
    masked = "foreground" if "foreground" in stats else "full_image"
    mean = stats[masked]["mean"]
    std = stats[masked]["std"]
    logger.info(f"Dataset stats ({masked}): mean={mean}, std={std}")

    train_transforms = build_train_transform(
        ndim,
        mean,
        std,
        model_config["data"]["train_patch_size"],
        fg_probability_3d=model_config["data"]["train_fg_probability_3d"],
    )
    val_transforms = build_val_transform(ndim, mean, std)

    datamodule = ImageDatamodule(
        data_dir=data_dir,
        split_file_path=splits_filepath,
        train_split_name=train_split,
        val_split_ratio=0.2,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=val_transforms,
        num_workers=0,
        train_batch_size=model_config["data"]["train_batch_size"],
        val_batch_size=model_config["data"]["val_batch_size"],
        seed=42,
        shuffle_train=False,
    )
    datamodule.setup()
    return datamodule


# --------------------------------------------------------------------------- #
# Model / trainer helpers
# --------------------------------------------------------------------------- #
def build_model(dataset_choice, model_config):
    return BinarySegmentator(
        lr=1e-3,
        input_channels=dataset_choice.input_channels,
        ndim=dataset_choice.ndim,
        num_layers=model_config["unet"]["num_layers"],
        features_start=model_config["unet"]["features_start"],
        bilinear=True,
        norm_op=model_config["unet"]["norm_op"],
        warmup_epochs=1,
        dropout=0.2,
        kernel_size=model_config["unet"]["kernel_size"],
        dice_loss_ratio=model_config["training"]["dice_loss_ratio"],
        val_patch_size=model_config["data"]["val_patch_size"],
        val_patch_overlap=model_config["data"]["val_patch_overlap"],
        val_sw_batch_size=model_config["data"]["val_sw_batch_size"],
    )


def get_train_callbacks(run_dir):
    return [
        ModelCheckpoint(
            dirpath=run_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
        ),
        PlotMetricsCallback(
            save_dir=run_dir, metrics=["loss", "dice", "precision", "recall"], every_n_epochs=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.00,
            verbose=True,
            mode="min",
        ),
    ]


def build_fit_trainer(run_dir, model_config, callbacks):
    csv_logger = CSVLogger(save_dir=run_dir)
    return Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=1000,
        precision="16-mixed",
        accumulate_grad_batches=model_config["training"]["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=csv_logger,
        default_root_dir=run_dir,  # safety net: anything Lightning writes by default lands here, not cwd
    )


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def train_from_scratch(datamodule, dataset_choice, model_config, run_dir):
    logger.info("Starting training from scratch")

    model = build_model(dataset_choice, model_config)
    callbacks = get_train_callbacks(run_dir)
    callbacks.append(SaveConfigCallback(config=model_config, save_dir=run_dir))

    trainer = build_fit_trainer(run_dir, model_config, callbacks)
    torch.set_float32_matmul_precision("medium")

    trainer.fit(model, datamodule=datamodule)
    logger.info("Training finished")


def resume_training(datamodule, dataset_choice, model_config, source_run_dir, run_dir):
    """
    Resumes from the checkpoint in source_run_dir, but writes everything
    (checkpoints, logs, plots, config snapshot) into a brand new run_dir --
    same as a fresh `train`. This avoids juggling CSVLogger versions,
    ModelCheckpoint dirpaths, etc. inside a single reused directory; the
    only thing carried over from source_run_dir is the checkpoint itself
    and, for plot continuity, the previous metrics history.
    """
    logger.info(f"Resuming from {source_run_dir} into new run directory {run_dir}")

    ckpt_path = find_checkpoint_in_dir(source_run_dir)
    logger.info(f"Resuming from checkpoint: {ckpt_path}")

    old_metrics_path = find_latest_metrics_csv(source_run_dir)

    model = build_model(dataset_choice, model_config)
    callbacks = get_train_callbacks(run_dir)
    callbacks.append(SaveConfigCallback(config=model_config, save_dir=run_dir))

    trainer = build_fit_trainer(run_dir, model_config, callbacks)
    seed_csv_logger_history(trainer.logger, old_metrics_path)

    torch.set_float32_matmul_precision("medium")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("Training finished")


def test_run(datamodule, run_dir):
    logger.info("Starting testing")
    device = get_device()

    ckpt_path = find_checkpoint_in_dir(run_dir)
    logger.info(f"Loading checkpoint: {ckpt_path}")

    model = BinarySegmentator.load_from_checkpoint(ckpt_path, map_location=device)
    logger.info(str(model))

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=False,  # no training curves to log during a pure test pass
        default_root_dir=run_dir,  # safety net: anything Lightning writes by default lands here, not cwd
    )
    torch.set_float32_matmul_precision("medium")

    trainer.test(model, datamodule=datamodule)
    logger.info("Testing finished")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/resume/test the UNet segmentation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared by every subcommand, so it can be passed after train/resume/test.
    dataset_parser = argparse.ArgumentParser(add_help=False)
    dataset_parser.add_argument(
        "--dataset", type=str, default="PERSEVERE",
        help="Name of the dataset to use, must be a key of available_datasets (default: PERSEVERE)",
    )

    train_parser = subparsers.add_parser(
        "train", parents=[dataset_parser], help="Train a new model from scratch"
    )
    train_parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for the new run directory (default: timestamp, handled by make_run_dir)",
    )
    train_parser.add_argument(
        "--test", action="store_true", help="Run testing right after training completes"
    )

    resume_parser = subparsers.add_parser(
        "resume", parents=[dataset_parser],
        help="Resume a previous training run's checkpoint into a new run directory",
    )
    resume_parser.add_argument(
        "--from-run", type=str, default=None, dest="from_run",
        help="Name of the existing run to resume from, i.e. whose checkpoint to load "
             "(default: most recent run)",
    )
    resume_parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for the NEW run directory this resumed training writes to "
             "(default: timestamp, handled by make_run_dir)",
    )
    resume_parser.add_argument(
        "--test", action="store_true", help="Run testing right after training completes"
    )

    test_parser = subparsers.add_parser(
        "test", parents=[dataset_parser], help="Test a previously trained run"
    )
    test_parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name of the run to test (default: most recent run)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_choice = resolve_dataset(args.dataset)
    model_config = dataset_choice.model_config

    if args.mode == "train":
        run_dir = dataset_choice.make_run_dir(base_dir=BASE_DIR, run_name=args.run_name)
    elif args.mode == "resume":
        source_run_dir = resolve_run_dir(dataset_choice, BASE_DIR, args.from_run)
        run_dir = dataset_choice.make_run_dir(base_dir=BASE_DIR, run_name=args.run_name)
    else:  # test
        run_dir = resolve_run_dir(dataset_choice, BASE_DIR, args.run_name)

    setup_logging(run_dir)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Model config: {model_config}")

    datamodule = build_datamodule(dataset_choice, model_config)

    if args.mode == "train":
        train_from_scratch(datamodule, dataset_choice, model_config, run_dir)
        if args.test:
            test_run(datamodule, run_dir)
    elif args.mode == "resume":
        resume_training(datamodule, dataset_choice, model_config, source_run_dir, run_dir)
        if args.test:
            test_run(datamodule, run_dir)
    elif args.mode == "test":
        test_run(datamodule, run_dir)


if __name__ == "__main__":
    main()