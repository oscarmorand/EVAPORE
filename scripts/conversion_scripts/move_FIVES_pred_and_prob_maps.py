import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from fire import Fire
import logging
import colorlog
import torch

def move_FIVES_pred_and_prob_maps(log_level = "INFO", n_debug = -1):

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

    dst_pred_dir = os.path.join(base_dir, "pred/")
    if not os.path.exists(dst_pred_dir):
        os.makedirs(dst_pred_dir)

    dst_prob_map_dir = os.path.join(base_dir, "probability_maps/")
    if not os.path.exists(dst_prob_map_dir):
        os.makedirs(dst_prob_map_dir)

    src_pred_dir = "/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds"
    src_prob_map_dir = "/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds/outmaps"

    pred_files = os.listdir(src_pred_dir)
    pred_paths = [os.path.join(src_pred_dir, f) for f in pred_files if f.endswith('.png')]
    pred_indexes = [int(f.split('_')[0]) for f in pred_files if f.endswith('.png')]

    prob_map_files = os.listdir(src_prob_map_dir)
    prob_map_paths = [os.path.join(src_prob_map_dir, f) for f in prob_map_files if f.endswith('.pt')]
    prob_map_indexes = [int(f.split('_')[0]) for f in prob_map_files if f.endswith('.pt')]

    dst_shape = (2048, 2048)

    if n_debug >= 0:
        logger.warning(f"🔍 Debug mode: processing only first {n_debug} images 🔍")
        pred_paths = pred_paths[:n_debug]
        prob_map_paths = prob_map_paths[:n_debug]

    # CONVERTION AND SAVE
    for index, pred_path in tqdm(zip(pred_indexes, pred_paths)):
        pred = np.array(Image.open(pred_path).resize(dst_shape, Image.NEAREST))

        id_str = str(index).zfill(3)
        filename = f"FIVES_{id_str}.png"
        dst_path = os.path.join(dst_pred_dir, filename)

        Image.fromarray(pred).save(dst_path)

    for index, prob_map_path in tqdm(zip(prob_map_indexes, prob_map_paths)):
        zero_prob_map_tensor = torch.load(prob_map_path, map_location='cpu', weights_only=False)[0]
        one_prob_map_tensor = torch.load(prob_map_path, map_location='cpu', weights_only=False)[1]

        zero_prob_map = np.array(Image.fromarray(zero_prob_map_tensor).resize(dst_shape, Image.NEAREST))
        one_prob_map = np.array(Image.fromarray(one_prob_map_tensor).resize(dst_shape, Image.NEAREST))

        prob_map = np.stack([zero_prob_map, one_prob_map], axis=0)

        id_str = str(index).zfill(3)
        filename = f"FIVES_{id_str}.pt"
        dst_path = os.path.join(dst_prob_map_dir, filename)
        
        torch.save(torch.tensor(prob_map), dst_path)

if __name__ == "__main__":
    Fire(move_FIVES_pred_and_prob_maps)