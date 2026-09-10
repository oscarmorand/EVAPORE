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

def _save_by_format(array: np.ndarray, path: Path, is_vol: bool, affine=None) -> None:
    """Internal: write array to path using the format implied by is_vol."""
    if is_vol:
        if nib is None:
            raise ImportError(
                "nibabel is required to save volume files (.nii/.nii.gz/.mgz). "
                "Install it with `pip install nibabel`."
            )
        vol_affine = np.eye(4) if affine is None else affine
        suffixes = "".join(path.suffixes).lower()
        if suffixes.endswith((".mgz", ".mgh")):
            image = nib.MGHImage(array, vol_affine)
        else:
            image = nib.Nifti1Image(array, vol_affine)
        nib.save(image, str(path))
    else:
        Image.fromarray(array).save(path)


def save_array_by_ndim(
    array: np.ndarray,
    folder: Union[str, Path],
    name: str,
    ndim: int,
    affine: "np.ndarray | None" = None,
) -> Path:
    """Save an array to ``folder``, picking the format from ``ndim``.

    - ``ndim == 2`` -> saved as ``name.png``.
    - ``ndim == 3`` -> saved as ``name.nii.gz``.

    Args:
        array: the array to save.
        folder: destination directory. Created if it doesn't exist.
        name: filename without extension.
        ndim: 2 for a 2D image, 3 for a 3D volume.
        affine: for volumes only, the 4x4 affine to store in the header.
            Defaults to the identity matrix if not provided.

    Returns:
        The full path the array was saved to.
    """
    if ndim not in (2, 3):
        raise ValueError(f"Unsupported ndim={ndim!r}. Expected 2 or 3.")

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ext = ".png" if ndim == 2 else ".nii.gz"
    path = folder / f"{name}{ext}"

    _save_by_format(array, path, is_vol=(ndim == 3), affine=affine)
    return path


def save_array(
    array: np.ndarray,
    path: Union[str, Path],
    affine: "np.ndarray | None" = None,
) -> None:
    """Save an array to an explicit path, inferring format from its extension.

    - ``.png/.jpg/.bmp/.tif/...`` -> saved via PIL.
    - ``.nii/.nii.gz/.mgz/.mgh`` -> saved via nibabel.

    Args:
        array: the array to save.
        path: full destination path, including extension.
        affine: for volumes only, the 4x4 affine to store in the header.
            Defaults to the identity matrix if not provided. Ignored for
            images.
    """
    path = Path(path)

    if is_volume(path):
        _save_by_format(array, path, is_vol=True, affine=affine)
    elif is_image(path):
        _save_by_format(array, path, is_vol=False)
    else:
        raise ValueError(
            f"Unsupported file extension for {path}. "
            f"Expected one of {IMAGE_SUFFIXES + VOLUME_SUFFIXES}."
        )