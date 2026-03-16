import os
import numpy as np
import PIL.Image as Image
from utils.path import data_dir

def convert_DRIVE_gt(log_level = "INFO", n_debug = -1):
    # PATHS
    base_dir = os.path.join(data_dir, "DRIVE/")
    gt_dir = os.path.join(base_dir, "gt/")

    gt_names = os.listdir(gt_dir)
    for old_name in gt_names:
        old_path = os.path.join(gt_dir, old_name)
        n_str = old_name.split("_")[0]
        n = int(n_str)

        new_name = "DRIVE_" + str(n).zfill(3) + ".png"
        new_path = os.path.join(gt_dir, new_name)

        # open old gif
        img = Image.open(old_path)
        img = img.convert("L")  # convert to grayscale
        img_np = np.array(img)
        # binarize
        img_np = (img_np > 128).astype(np.uint8) * 255
        # save as png
        img_png = Image.fromarray(img_np)
        img_png.save(new_path)

def convert_DRIVE_img(log_level = "INFO", n_debug = -1):
    # PATHS
    base_dir = os.path.join(data_dir, "DRIVE/")
    img_dir = os.path.join(base_dir, "img/")

    img_names = os.listdir(img_dir)
    for old_name in img_names:
        old_path = os.path.join(img_dir, old_name)
        n_str = old_name.split("_")[0]
        n = int(n_str)

        new_name = "DRIVE_" + str(n).zfill(3) + ".png"
        new_path = os.path.join(img_dir, new_name)

        # open old tif
        img = Image.open(old_path)
        img = img.convert("RGB")  # convert to RGB
        img.save(new_path)


def convert_foreground_masks(log_level = "INFO", n_debug = -1):
    # PATHS
    base_dir = os.path.join(data_dir, "DRIVE/")
    fgm_dir = os.path.join(base_dir, "foreground_masks/")

    fgm_names = os.listdir(fgm_dir)
    for old_name in fgm_names:
        old_path = os.path.join(fgm_dir, old_name)
        n_str = old_name.split("_")[0]
        n = int(n_str)

        new_name = "DRIVE_" + str(n).zfill(3) + ".png"
        new_path = os.path.join(fgm_dir, new_name)

        # open old gif
        img = Image.open(old_path)
        img = img.convert("L")  # convert to grayscale
        img_np = np.array(img)
        # binarize
        img_np = (img_np > 128).astype(np.uint8) * 255
        # save as png
        img_png = Image.fromarray(img_np)
        img_png.save(new_path)

if __name__ == "__main__":
    #convert_DRIVE_gt()
    #convert_DRIVE_img()
    convert_foreground_masks()