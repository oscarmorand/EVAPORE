import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from graph.graph_creation import img_to_graph
from fire import Fire
import logging
import colorlog
from graph.graph_io import save_graph_to_dot
import json
from utils.path import data_dir

def convert_FIVES_to_graph(log_level = "INFO", n_debug = -1):

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
    base_dir = os.path.join(data_dir, "FIVES/")
    train_dir = os.path.join(base_dir, "train/")
    gt_dir = os.path.join(train_dir, "Ground truth/")
    images_dir = os.path.join(train_dir, "Original/")

    dst_dir = os.path.join(train_dir, "graphs/")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    paths = os.listdir(gt_dir)
    paths.sort()
    paths_indexes = [int(p.split('_')[0]) for p in paths]
    real_paths = [None] * len(paths_indexes)
    for i, idx in enumerate(paths_indexes):
        real_paths[idx - 1] = paths[i]

    if n_debug >= 0:
        logger.warning(f"🔍 Debug mode: processing only first {n_debug} images 🔍")
        real_paths = real_paths[:n_debug]

    json_path = os.path.join(train_dir, "studies.json")
    studies_info = {}
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            f.write(json.dumps(studies_info))
    with open(json_path, 'r') as f:
        studies_info = json.load(f)

    # CONVERTION AND SAVE
    for i, path in enumerate(tqdm(real_paths)):
        src_path = os.path.join(gt_dir, path)
        img_path = os.path.join(images_dir, path)

        id = int(path.split('_')[0])
        diagnosis = path.split('_')[1].split('.')[0]

        img = np.array(Image.open(src_path))[:,:,0]
        
        G = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)
        
        id_str = str(id).zfill(3)
        filename = f"{id_str}_graph.dot"
        dst_path = os.path.join(dst_dir, filename)

        save_graph_to_dot(G, dst_path)

        studies_info[id_str] = {
            "graph_path": dst_path,
            "ground_truth_path": src_path,
            "original_image_path": img_path,
            "diagnosis": diagnosis
        }

    with open(json_path, 'w') as f:
        f.write(json.dumps(studies_info, indent=4))

if __name__ == "__main__":
    Fire(convert_FIVES_to_graph)