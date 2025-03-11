import os
import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor

# Path to your dataset
dataset_path = "/media/irina/My Book/Surmont/TrainingCNN/synth_tracks_1024px_10cm_v3.22"
threshold_area = 0.01  # 3%

# Filtering function for a single file
def process_file(file_path):
    img_path = file_path.replace("label.tif", "image.tif")

    with rasterio.open(file_path) as src:
        mask = src.read(1)

    # Binarize mask
    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    # Calculate percentage of area covered by 1s
    coverage_ratio = np.sum(binary_mask) / binary_mask.size

    # Remove files if coverage is below threshold
    if coverage_ratio < threshold_area:
        os.remove(file_path)
        if os.path.exists(img_path):
            os.remove(img_path)
        return None, file_path
    return file_path, None

# Main parallel processing function
def filter_and_remove_small_masks_parallel(dataset_path, threshold_area=0.03):
    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith("label.tif")]

    retained_files = []
    removed_files = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(process_file, files)

    for retained, removed in results:
        if retained:
            retained_files.append(retained)
        if removed:
            removed_files.append(removed)

    return retained_files, removed_files

# Run filtering and removal
retained_patches, removed_patches = filter_and_remove_small_masks_parallel(dataset_path, threshold_area)

print(f"Retained patches (coverage >= {threshold_area * 100}%): {len(retained_patches)}")
print(f"Removed patches (coverage < {threshold_area * 100}%): {len(removed_patches)}")
for rp in removed_patches[:10]:  # print first 10 removed paths
    print(rp)
