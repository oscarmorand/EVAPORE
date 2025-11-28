import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from graph.graph_creation import img_to_graph
from fire import Fire
import logging
import colorlog
from graph.graph_io import save_graph_to_json
import json
from graph_neural_networks.data.utils.pred_state import get_combined_graph

def convert_FIVES_to_graph(pred_dir, log_level = "INFO", n_debug = -1):

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
    gt_dir = os.path.join(base_dir, "gt/")

    dst_dir = os.path.join(base_dir, "raw_pred_state/")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    paths = os.listdir(gt_dir)
    paths_indexes = [int(p.split('_')[0]) for p in paths]
    paths_indexes = np.argsort(paths_indexes)
    real_paths = [None] * len(paths_indexes)
    for i, idx in enumerate(paths_indexes):
        real_paths[i] = paths[idx]

    pred_files = os.listdir(pred_dir)
    pred_paths = [os.path.join(pred_dir, f) for f in pred_files if f.endswith('.png')]
    pred_indexes = [int(f.split('_')[0]) for f in pred_files if f.endswith('.png')]

    if n_debug >= 0:
        logger.warning(f"🔍 Debug mode: processing only first {n_debug} images 🔍")
        real_paths = real_paths[:n_debug]

    # CONVERTION AND SAVE
    for i, path in enumerate(tqdm(real_paths)):
        gt_path = os.path.join(gt_dir, path)

        id = int(path.split('_')[0])

        pred_idx = pred_indexes.index(id)
        pred_path = pred_paths[pred_idx]

        gt = np.array(Image.open(gt_path))[:,:,0]

        pred = np.array(Image.open(pred_path).resize(gt.shape[1::-1]))

        G = get_combined_graph(gt, pred)

        id_str = str(id).zfill(3)
        filename = f"FIVES_{id_str}.json"
        dst_path = os.path.join(dst_dir, filename)

        save_graph_to_json(G, dst_path)

if __name__ == "__main__":
    Fire(convert_FIVES_to_graph)