import os
from pathlib import Path
import numpy as np
import os
import glob

def list_tif_files(path):
    """
    If 'path' is a directory, return all .tif files inside (top-level).
    If 'path' is a .tif file, return it as a single-item list.
    Otherwise, raise ValueError.
    """
    # 1. Check if it's a directory
    if os.path.isdir(path):
        # Use glob or manual filtering
        tifs = glob.glob(os.path.join(path, '*.tif'))
        if not tifs:
            raise ValueError(f"No .tif files found in directory: {path}")
        return sorted(tifs)

    # 2. If it's a file, confirm it ends with .tif
    elif os.path.isfile(path):
        if path.lower().endswith('.tif'):
            return [path]
        else:
            raise ValueError(f"File {path} does not end with '.tif'")

    # 3. Otherwise, not a file nor a directory
    else:
        raise ValueError(f"Path {path} is not a .tif file or directory containing .tif files.")


def generate_patch_indices(size, patch_size, overlap_size):
    """
    Generates a sorted list of start indices for patches such that
    the stride is (patch_size - overlap_size) but includes a final
    patch if needed so we don't miss coverage at the boundary.
    """
    stride = patch_size - overlap_size
    indices = []
    start = 0

    while True:
        if start + patch_size >= size:
            final_start = max(size - patch_size, 0)
            indices.append(final_start)
            break
        indices.append(start)
        start += stride

    return sorted(set(indices))

def normalize_nDTM(image, std=1):
    """
    Normalize image values to the [0, 1] range,
    clipping extreme values based on standard deviation range.
    """
    min_val, max_val = -std, std
    image = np.clip(image, min_val, max_val)
    return (image - min_val) / (max_val - min_val)
