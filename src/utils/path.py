from os import environ
import os

data_dir = environ.get("EVAPORE_DATA_DIR")
checkpoints_dir = environ.get("EVAPORE_CHECKPOINTS_DIR")

if data_dir is None:
    raise EnvironmentError("Environment variable 'EVAPORE_DATA_DIR' is not set.")
if checkpoints_dir is None:
    raise EnvironmentError("Environment variable 'EVAPORE_CHECKPOINTS_DIR' is not set.")

def get_data_dir(*args):
    """
    Constructs a path by joining the base data directory with the provided subdirectories.

    Args:
        *args: Variable length argument list of subdirectory names.

    Returns:
        str: The constructed path to the specified location within the data directory.
    """
    return os.path.join(data_dir, *args)

def get_checkpoint_dir(*args):
    """
    Constructs a path by joining the base checkpoints directory with the provided subdirectories.

    Args:
        *args: Variable length argument list of subdirectory names.

    Returns:
        str: The constructed path to the specified location within the checkpoints directory.
    """
    return os.path.join(checkpoints_dir, *args)