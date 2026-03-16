from enum import Enum

class SymmetryEnforcementMode(Enum):
    DATA_AUGMENTATION = "data_augmentation"
    DOUBLE_PASS = "double_pass"
    NONE = "none" 