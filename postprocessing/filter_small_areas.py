import os
import cupy as cp
import rasterio
import numpy as np
from skimage.measure import label
from tqdm import tqdm
import gc


def filter_small_areas_with_conf_int(folder_path, conf_thre=0.1, threshold_area=100, chunk_size=1000):
    """
    Filters small connected areas from all .tif files in the specified folder.
    Retains original confidence levels and saves as uint8 (confidence * 100).
    Processes large arrays in chunks to avoid GPU OOM errors.
    """
    print(f"🚀 Starting filtering in folder: {folder_path}")

    # Ensure filtered folder exists
    filtered_folder = os.path.join(folder_path, "filtered_int")
    os.makedirs(filtered_folder, exist_ok=True)

    # Find TIF files
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

    if not tif_files:
        print("⚠️ No .tif files found!")
        return

    for file in tqdm(tif_files, desc="🛠️ Processing TIF files"):
        file_path = os.path.join(folder_path, file)
        print(f"🖼️ Processing: {file_path}")

        # Open raster and read data
        with rasterio.open(file_path) as src:
            transform = src.transform
            profile = src.profile
            data = src.read(1).astype(np.float32)

        # Move data to GPU with chunking
        height, width = data.shape
        filtered_data_gpu = cp.zeros_like(cp.asarray(data))

        print(f"📏 Image shape: {height}x{width}")

        # Process image in chunks
        for start_row in range(0, height, chunk_size):
            end_row = min(start_row + chunk_size, height)
            chunk = data[start_row:end_row, :]
            chunk_gpu = cp.asarray(chunk)

            # Apply confidence threshold
            conf_mask = chunk_gpu > conf_thre
            print(f"🔍 Chunk [{start_row}:{end_row}] - Mean after threshold: {conf_mask.mean():.4f}")

            # Identify connected areas
            labeled_chunk = cp.array(label(cp.asnumpy(conf_mask), connectivity=2))

            # Calculate areas for each region
            unique_labels, counts = cp.unique(labeled_chunk, return_counts=True)
            valid_labels = unique_labels[counts >= threshold_area]

            # Create mask to keep valid areas only
            valid_mask = cp.isin(labeled_chunk, valid_labels)

            # Apply mask to original data to retain confidence
            filtered_chunk = cp.where(valid_mask, chunk_gpu, 0)

            # Save filtered chunk to the larger array
            filtered_data_gpu[start_row:end_row, :] = filtered_chunk

            # Free memory
            del chunk_gpu, labeled_chunk, conf_mask, valid_mask
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            print(f"✅ Chunk [{start_row}:{end_row}] processed")

        # Scale confidence values to integers (0–100)
        # filtered_scaled = cp.clip(filtered_data_gpu * 100, 0, 100)

        # Move filtered data back to CPU and save as uint8
        filtered_cpu = cp.asnumpy(filtered_data_gpu).astype(np.uint8)

        # Update profile for the filtered output
        profile.update(dtype='uint8', compress='lzw')
        filtered_path = os.path.join(filtered_folder, f"{os.path.splitext(file)[0]}_filt{threshold_area}_uint8.tif")

        with rasterio.open(filtered_path, 'w', **profile) as dst:
            dst.write(filtered_cpu, 1)

        # Clean up memory after each file
        del filtered_data_gpu
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        print(f"✅ Filtered file saved: {filtered_path}")


# Run the function
folder = "/media/irina/My Book1/Conoco/DATA/Products/Trails/unet"
threshold = 500  # Adjust threshold as needed
conf_thre = 20

filter_small_areas_with_conf_int(folder, conf_thre, threshold)
