import os
from fire import Fire

def create_clean_dataset(FIVES_dir = None, FIVES_clean_dir = None):
    if FIVES_dir is None:
        FIVES_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'FIVES'))
    if FIVES_clean_dir is None:
        FIVES_clean_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'FIVES_clean'))
    FIVES_dir = os.path.abspath(FIVES_dir)
    FIVES_clean_dir = os.path.abspath(FIVES_clean_dir)

    copy_to_new_folder = True
    if FIVES_dir == FIVES_clean_dir:
        copy_to_new_folder = False

    if not os.path.exists(FIVES_dir):
        raise ValueError(f"Directory {FIVES_dir} does not exist.")
    if not os.path.exists(FIVES_clean_dir):
        raise ValueError(f"Directory {FIVES_clean_dir} does not exist.")

    clean_files_idx_filepath = os.path.join(FIVES_clean_dir, 'clean_files_idx.txt')
    if not os.path.exists(clean_files_idx_filepath):
        raise ValueError(f"File {clean_files_idx_filepath} does not exist.")
    with open(clean_files_idx_filepath, 'r') as f:
        clean_files_idx = [line.strip() for line in f.readlines()]

    src_img_filepaths = [os.path.join(FIVES_dir, 'img', f"{idx}.png") for idx in clean_files_idx]
    src_gt_filepaths = [os.path.join(FIVES_dir, 'gt', f"{idx}.json") for idx in clean_files_idx]

    dst_img_filepaths = [os.path.join(FIVES_clean_dir, 'img', f"{idx}.png") for idx in clean_files_idx]
    dst_gt_filepaths = [os.path.join(FIVES_clean_dir, 'gt', f"{idx}.json") for idx in clean_files_idx]

if __name__ == "__main__":
    Fire(create_clean_dataset)