from enum import Enum
import os
from utils.path import get_data_dir, get_checkpoint_dir
from datetime import datetime
import json
import glob

class Dataset:
    def __init__(self, name: str, ndim: int, input_channels: int, preferred_train_split: str = "train"):
        self.name = name
        self.ndim = ndim
        self.input_channels = input_channels
        self.preferred_train_split = preferred_train_split
        self.data_dir = get_data_dir(self.name)
        self.checkpoint_dir = get_checkpoint_dir(self.name)
        self.splits_filepath = os.path.join(self.data_dir, "splits.json")
        self.model_config_filepath = os.path.join(self.data_dir, "model_config.json")
        self.model_config = {}
        if os.path.exists(self.model_config_filepath):
            with open(self.model_config_filepath, 'r') as f:
                self.model_config = json.load(f)
    
    def get_checkpoints_dir(self, checkpoint_folder: str):
        return os.path.join(self.checkpoint_dir, checkpoint_folder)

    def get_checkpoint_dir_by_id(self, checkpoint_folder: str, id: int):
        checkpoints_dir = self.get_checkpoints_dir(checkpoint_folder)
        checkpoint_dirname = os.listdir(checkpoints_dir)[id]
        checkpoint_dirpath = os.path.join(checkpoints_dir, checkpoint_dirname)
        return checkpoint_dirpath

    def get_checkpoint_by_id(self, checkpoint_folder: str, id: int):
        checkpoint_dirpath = self.get_checkpoint_dir_by_id(checkpoint_folder, id)
        files = glob.glob(os.path.join(checkpoint_dirpath, "*.ckpt"))
        checkpoint_name = files[0]
        checkpoint_path = os.path.join(checkpoint_dirpath, checkpoint_name)
        return checkpoint_path

    def make_run_dir(self, base_dir: str, run_name: str = None) -> str:
        """Creates checkpoints/<run_name or timestamp>/ and returns its path."""

        checkpoint_dir = self.get_checkpoints_dir(base_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        name = run_name or datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        run_dir = os.path.join(checkpoint_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        
        return run_dir


available_datasets = {
    "FIVES": Dataset("FIVES", ndim=2, input_channels=3, preferred_train_split="train_clean"),
    "DRIVE": Dataset("DRIVE", ndim=2, input_channels=3, preferred_train_split="train"),
    "CHASE": Dataset("CHASE", ndim=2, input_channels=3, preferred_train_split="train"),
    "PERSEVERE_subset": Dataset("PERSEVERE_subset", ndim=3, input_channels=1, preferred_train_split="train"),
    "PERSEVERE": Dataset("PERSEVERE", ndim=3, input_channels=1, preferred_train_split="train")
}