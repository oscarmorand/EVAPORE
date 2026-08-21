from enum import Enum
import os
from utils.path import get_data_dir, get_checkpoint_dir

class Dataset:
    def __init__(self, name: str, preferred_train_split: str = "train"):
        self.name = name
        self.preferred_train_split = preferred_train_split
        self.data_dir = get_data_dir(self.name)
        self.checkpoint_dir = get_checkpoint_dir(self.name)
        self.splits_filepath = os.path.join(self.data_dir, "splits.json")
    
    def get_checkpoint_dir(self, checkpoint_folder: str):
        return os.path.join(self.checkpoint_dir, checkpoint_folder)
    
    def get_first_checkpoint_path(self, checkpoint_folder: str):
        checkpoint_dir = self.get_checkpoint_dir(checkpoint_folder)
        return os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])


available_datasets = {
    "FIVES": Dataset("FIVES", preferred_train_split="train_clean")
    #"DRIVE": Dataset("DRIVE", preferred_train_split="train")
    #"CHASE": Dataset("CHASE", preferred_train_split="train")
}