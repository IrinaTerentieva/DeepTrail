import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from tqdm import tqdm
from multiprocessing import Pool

# Paths
input_folder = "/media/irina/My Book/Surmont/Products/Trails/trailformer_epoch_31_val_loss_0.0592"
output_folder = "/media/irina/My Book/Surmont/Products/Trails/Density_Outputs"
os.makedirs(output_folder, exist_ok=True)

# Parameters
cell_size = 5  # Output grid cell size in meters
num_workers = 16  # Number of CPU cores to use

# List all .tif files
tif_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".tif")]
print(f"Found {len(tif_files)} raster files.")

# Function to compute density and save output raster
def compute_density_raster(tif_path):
    try:
        # Open the raster
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()
            transform = src.transform  # Original raster transform

            # Get raster metadata
            raster_res = src.res[0]  # Assuming square pixels, get pixel resolution
            width, height = src.width, src.height
            minx, maxy = transform.c, transform.f  # Upper-left corner of raster
            maxx = minx + (width * raster_res)
            miny = maxy - (height * raster_res)

            # Define new output resolution
            scale_factor = int(cell_size / raster_res)
            out_width = width // scale_factor
            out_height = height // scale_factor
            out_transform = from_origin(minx, maxy, cell_size, cell_size)

            # Read full raster into memory
            trail_raster = src.read(1)

            # Initialize density output
            density_raster = np.zeros((out_height, out_width), dtype=np.float32)

            # Compute density using block processing
            for i in range(out_height):
                for j in range(out_width):
                    # Define window in original raster
                    x_start = j * scale_factor
                    x_end = x_start + scale_factor
                    y_start = i * scale_factor
                    y_end = y_start + scale_factor

                    # Extract sub-array
                    cell_array = trail_raster[y_start:y_end, x_start:x_end]

                    # Compute density (fraction of pixels that contain trails)
                    total_pixels = cell_array.size
                    trail_pixels = np.count_nonzero(cell_array)
                    density = trail_pixels / total_pixels if total_pixels > 0 else 0

                    # Assign to output raster
                    density_raster[i, j] = density

        # Remove 'transform' key from profile to avoid duplicate values
        profile.pop("transform", None)
        profile.update(dtype="float32", compress="lzw", count=1, height=out_height, width=out_width)

        # Save density raster
        output_path = os.path.join(output_folder, os.path.basename(tif_path).replace(".tif", "_density.tif"))
        with rasterio.open(output_path, 'w', **profile, transform=out_transform) as dst:
            dst.write(density_raster, 1)

        print(f"Saved density raster: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error processing {tif_path}: {e}")
        return None

# Use multiprocessing for efficiency
if __name__ == "__main__":
    print(f"Processing {len(tif_files)} raster files with {num_workers} CPUs...")

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(compute_density_raster, tif_files), total=len(tif_files), desc="Processing Density Rasters"))

    print(f"Finished processing. {len([r for r in results if r])} density rasters saved.")
