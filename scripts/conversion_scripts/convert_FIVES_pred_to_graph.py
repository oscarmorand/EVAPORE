import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from graph.graph_creation import img_to_graph
from fire import Fire
import logging
import colorlog
from graph.graph_io import save_graph_to_json

def convert_FIVES_pred_to_graph(log_level = "INFO", n_debug = -1):

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

    dst_dir = os.path.join(base_dir, "raw_pred/")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    pred_dir = "/home/morand/afs/QTSeg/src/working/dataset/FIVES/train/preds"

    pred_files = os.listdir(pred_dir)
    pred_paths = [os.path.join(pred_dir, f) for f in pred_files if f.endswith('.png')]
    pred_indexes = [int(f.split('_')[0]) for f in pred_files if f.endswith('.png')]

    if n_debug >= 0:
        logger.warning(f"🔍 Debug mode: processing only first {n_debug} images 🔍")
        pred_paths = pred_paths[:n_debug]
        
    # CONVERTION AND SAVE
    for index, pred_path in tqdm(zip(pred_indexes, pred_paths)):
        src_path = pred_path

        dst_shape = (2048, 2048)
        img = np.array(Image.open(src_path).resize(dst_shape, Image.NEAREST))
        
        G = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)
        
        id_str = str(index).zfill(3)
        filename = f"FIVES_{id_str}.json"
        dst_path = os.path.join(dst_dir, filename)

        save_graph_to_json(G, dst_path)

if __name__ == "__main__":
    Fire(convert_FIVES_pred_to_graph)