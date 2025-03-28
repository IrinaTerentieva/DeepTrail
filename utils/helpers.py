import numpy as np

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
