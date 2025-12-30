import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from fire import Fire
import logging
import colorlog

def rename_FIVES_gt(log_level = "INFO", n_debug = -1):

    # LOGGING SETUP WITH COLOR
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)s] %(name)s: %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    ))
    # Set up the root logger with color
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = colorlog.getLogger()
    logger.setLevel(level)
    logger.handlers = [handler]  # Replace any existing handlers
    logger.info(f"✨ Logging initialized with level: {log_level.upper()} ✨")


    # PATHS
    data_dir = "/home/morand/afs/EVAPORE/data/"
    base_dir = os.path.join(data_dir, "FIVES/")

    gt_dir = os.path.join(base_dir, "img/")

    gt_files = os.listdir(gt_dir)
    gt_paths = [os.path.join(gt_dir, f) for f in gt_files if f.endswith('.png')]
    gt_indexes = [int(f.split('_')[0]) for f in gt_files if f.endswith('.png')]

    if n_debug >= 0:
        logger.warning(f"🔍 Debug mode: processing only first {n_debug} images 🔍")
        gt_paths = gt_paths[:n_debug]

    # RENAME GT
    for index, gt_path in tqdm(zip(gt_indexes, gt_paths)):
        id_str = str(index).zfill(3)
        filename = f"FIVES_{id_str}.png"
        dst_path = os.path.join(gt_dir, filename)

        logger.info(f"Renaming {gt_path} to {dst_path}")
        os.rename(gt_path, dst_path)    

if __name__ == "__main__":
    Fire(rename_FIVES_gt)