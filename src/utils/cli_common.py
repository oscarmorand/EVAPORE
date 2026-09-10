"""
Shared helpers for the project's command-line scripts (train.py, predict.py, ...).
Keeping these in one place means fixes (e.g. logging setup) only need to
happen once instead of being duplicated -- and potentially drifting -- across
every script.
"""
import glob
import logging
import os

from utils.available_datasets import available_datasets

logger = logging.getLogger(__name__)

# Directory (relative to cwd) where your own training runs live -- the same
# base_dir passed to dataset_choice.make_run_dir() in train.py.
TRAINING_BASE_DIR = "unet_pretraining"


def setup_logging(log_dir, log_filename="run.log"):
    """
    Configure logging so that every log record (this script's own
    logger.info() calls, and pytorch_lightning's training logs when
    applicable) is written both to the console and to a log file inside
    log_dir. Uses append mode, so repeated invocations targeting the same
    directory accumulate into one file.

    NOTE: pytorch_lightning (and lightning_fabric, which it depends on)
    configure their own logger at import time with `propagate = False`:
        _logger.addHandler(logging.StreamHandler())
        _logger.propagate = False
    That means their log records never bubble up to the root logger, so
    attaching a handler to root alone is not enough to capture them -- the
    file handler is attached directly to those loggers too. This is a no-op
    (just two extra, unused logger objects) in scripts that don't import
    pytorch_lightning, like predict.py.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    for logger_name in ("pytorch_lightning", "lightning_fabric"):
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(logging.INFO)
        lib_logger.addHandler(file_handler)

    logger.info(f"Logging to {log_path}")
    return log_path


def resolve_dataset(dataset_name):
    """Look up dataset_name in available_datasets, printing the valid keys on failure."""
    if dataset_name not in available_datasets:
        available = ", ".join(sorted(available_datasets.keys()))
        print(f"Unknown dataset: '{dataset_name}'")
        print(f"Available datasets: {available}")
        raise SystemExit(1)
    return available_datasets[dataset_name]


def resolve_run_dir(dataset_choice, base_dir, run_name):
    """
    Resolve the directory of an *existing* run under base_dir.

    ASSUMPTION: runs are created at "<base_dir>/<run_name>" (matching
    dataset_choice.make_run_dir(base_dir=..., run_name=...)). If that's not
    how your directory layout actually works, this is the only function you
    need to change.
    """
    if run_name is not None:
        checkpoints_dir = dataset_choice.get_checkpoints_dir(base_dir)
        run_dir = os.path.join(checkpoints_dir, run_name)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(
                f"Run directory not found: {run_dir}. "
                f"Check --run-name against the directories under {base_dir}/"
            )
        return run_dir

    # No run name given -> fall back to the most recent run
    ckpt_path = dataset_choice.get_checkpoint_by_id(base_dir, -1)
    return os.path.dirname(ckpt_path)


def find_checkpoint_in_dir(run_dir):
    """Find the checkpoint file saved directly (flat, not nested) in run_dir."""
    ckpts = glob.glob(os.path.join(run_dir, "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint (.ckpt) file found in {run_dir}")
    if len(ckpts) > 1:
        ckpts.sort(key=os.path.getmtime, reverse=True)
        logger.warning(f"Multiple checkpoints found in {run_dir}, using most recent: {ckpts[0]}")
    return ckpts[0]