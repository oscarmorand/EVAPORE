"""Generic array loading for 2D images and 3D volumes.

Chooses the loading backend (PIL vs nibabel) automatically based on file
extension, so the rest of the codebase can treat images and medical
volumes the same way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

try:
    import nibabel as nib
except ImportError:  # nibabel is only needed if you actually load volumes
    nib = None

VOLUME_SUFFIXES = (".nii", ".nii.gz", ".mgz", ".mgh")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_volume(path: Union[str, Path]) -> bool:
    suffixes = "".join(Path(path).suffixes).lower()
    return any(suffixes.endswith(ext) for ext in VOLUME_SUFFIXES)


def is_image(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


ALL_KNOWN_SUFFIXES = VOLUME_SUFFIXES + IMAGE_SUFFIXES + (".pt",)


def true_stem(path: Union[str, Path]) -> str:
    """Filename without its (possibly multi-part) extension.

    Unlike ``Path.stem``, this correctly handles multi-part extensions such
    as ``.nii.gz``: for ``scan_001.nii.gz``, ``Path.stem`` returns
    ``"scan_001.nii"`` (it only strips the last suffix), while this returns
    ``"scan_001"``.
    """
    path = Path(path)
    name = path.name
    full_suffix = "".join(path.suffixes).lower()
    for ext in sorted(ALL_KNOWN_SUFFIXES, key=len, reverse=True):
        if full_suffix.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def load_array(path: Union[str, Path], 
               grayscale: bool = False
) -> np.ndarray:
    
    """Load a 2D image or a 3D volume as a numpy array.

    - ``.png/.jpg/.bmp/.tif/...`` files are loaded with ``PIL.Image.open``.
    - ``.nii/.nii.gz/.mgz/.mgh`` files are loaded with ``nibabel.load``.

    Args:
        path: path to the file to load.
        grayscale: for 2D images only, convert to single-channel ("L") mode.
            Ignored for volumes, which are already scalar arrays.
    """
    path = Path(path)

    if is_volume(path):
        if nib is None:
            raise ImportError(
                "nibabel is required to load volume files (.nii/.nii.gz/.mgz). "
                "Install it with `pip install nibabel`."
            )
        volume = nib.load(str(path))
        return np.asanyarray(volume.dataobj)

    if is_image(path):
        image = Image.open(path)
        if grayscale:
            image = image.convert("L")
        return np.array(image)

    raise ValueError(
        f"Unsupported file extension for {path}. "
        f"Expected one of {IMAGE_SUFFIXES + VOLUME_SUFFIXES}."
    )