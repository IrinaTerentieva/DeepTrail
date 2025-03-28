import os
import gc
import numpy as np
import torch
import tempfile
import rasterio
import scipy.ndimage as ndimage
from tqdm import tqdm
from rasterio.windows import Window

from utils.rasters import generate_patch_indices, normalize_nDTM


def sliding_window_prediction(image_path, model, output_dir, patch_size, overlap_size, threshold=0.1):
    """
    Perform patch-wise prediction on a large raster using a sliding window approach.
    Outputs a GeoTIFF of predicted probabilities (0-100), thresholded, and cleaned.

    NOTE: This version requires a GPU. If none is found, it raises an error.
    """

    # --- Force GPU usage or raise an error
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found, but a GPU is required for this script.")
    device = torch.device("cuda")
    model.to(device)

    with rasterio.open(image_path) as src:
        print(f"\n[INFO] Processing: {image_path}")

        height, width = src.shape

        # Create memmaps for accumulation
        temp_dir = tempfile.gettempdir()
        accum_path = os.path.join(temp_dir, "accumulation.dat")
        count_path = os.path.join(temp_dir, "count_array.dat")

        accumulation = np.memmap(accum_path, dtype=np.float32, mode='w+', shape=(height, width))
        count_array = np.memmap(count_path, dtype=np.int32, mode='w+', shape=(height, width))
        accumulation[:] = 0.0
        count_array[:] = 0
        accumulation.flush()
        count_array.flush()

        # We read the raster in chunks to handle large dimensions
        read_chunk_size = 2048

        for row_start in tqdm(range(0, height, read_chunk_size), desc="Chunks (Inference)"):
            row_end = min(row_start + read_chunk_size, height)

            for col_start in range(0, width, read_chunk_size):
                col_end = min(col_start + read_chunk_size, width)
                window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                chunk_data = src.read(1, window=window)
                chunk_data = np.nan_to_num(chunk_data, nan=0.0)

                if src.nodata is not None:
                    chunk_data[chunk_data == src.nodata] = 0

                # Example of normalizing your data
                chunk_data = normalize_nDTM(chunk_data)

                row_inds = generate_patch_indices(chunk_data.shape[0], patch_size, overlap_size)
                col_inds = generate_patch_indices(chunk_data.shape[1], patch_size, overlap_size)

                for i in row_inds:
                    for j in col_inds:
                        local_patch = chunk_data[i:i + patch_size, j:j + patch_size]
                        if local_patch.size == 0 or np.all(local_patch == 0):
                            continue

                        actual_h, actual_w = local_patch.shape
                        patch_input = np.zeros((patch_size, patch_size), dtype=local_patch.dtype)
                        patch_input[:actual_h, :actual_w] = local_patch

                        # Move patch to GPU
                        tensor_patch = torch.from_numpy(patch_input).unsqueeze(0).unsqueeze(0).float().to(device)

                        with torch.no_grad():
                            # Forward pass
                            logits = model(pixel_values=tensor_patch).logits
                            probs = torch.sigmoid(logits)
                            upsampled = torch.nn.functional.interpolate(
                                probs, size=(patch_size, patch_size),
                                mode="bilinear", align_corners=False
                            )
                            # Bring back to CPU for accumulation
                            prob_array = upsampled.squeeze().cpu().numpy()

                        global_i = row_start + i
                        global_j = col_start + j

                        accumulation[global_i:global_i + actual_h, global_j:global_j + actual_w] += prob_array[
                                                                                                    :actual_h,
                                                                                                    :actual_w]
                        count_array[global_i:global_i + actual_h, global_j:global_j + actual_w] += 1

                        del tensor_patch, logits, probs, upsampled, prob_array
                        torch.cuda.empty_cache()

                # Flush memmaps after each chunk
                accumulation.flush()
                count_array.flush()

        # Write output to GeoTIFF
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_prediction.tif'
        output_path = os.path.join(output_dir, output_filename)

        profile = src.profile.copy()
        profile.update({'driver': 'GTiff', 'dtype': 'uint8', 'count': 1, 'compress': 'lzw'})
        profile.pop('nodata', None)  # remove nodata from profile

        with rasterio.open(output_path, 'w', **profile) as dst:
            for row_start in tqdm(range(0, height, 1024), desc="Chunks (Write)"):
                row_end = min(row_start + 1024, height)
                for col_start in range(0, width, 1024):
                    col_end = min(col_start + 1024, width)

                    accum = accumulation[row_start:row_end, col_start:col_end]
                    count = count_array[row_start:row_end, col_start:col_end]
                    prob_chunk = np.divide(accum, np.maximum(count, 1))

                    # Threshold + connected component filtering
                    bin_mask = (prob_chunk >= threshold).astype(np.uint8)
                    labeled, n = ndimage.label(bin_mask)
                    sizes = ndimage.sum(bin_mask, labeled, index=range(1, n + 1))
                    small = [idx for idx, s in enumerate(sizes, start=1) if s < 200]
                    if small:
                        bin_mask[np.isin(labeled, small)] = 0
                    prob_chunk[bin_mask == 0] = 0.0

                    scaled = (prob_chunk * 100).astype(np.uint8)
                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    dst.write(scaled, 1, window=window)

        del accumulation, count_array
        gc.collect()

        print(f"[INFO] Prediction saved to {output_path}")
        return output_path
