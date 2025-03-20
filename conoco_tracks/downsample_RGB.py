import os
import time
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from multiprocessing import Pool


def reproject_in_chunks(input_path, output_path, dst_crs="EPSG:2956", dst_res=0.1):
    """Reprojects and resamples a single TIFF file in chunks."""
    with rasterio.open(input_path) as src:
        # Calculate the output transform, width, and height based on the target CRS and resolution.
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=dst_res
        )

        # Update metadata with tiling options (using 2048x2048 blocks for speed).
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'tiled': True,
            'blockxsize': 2048,
            'blockysize': 2048
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            start_time = time.time()
            # Process each block window (each approximately 2048x2048 pixels, except edges)
            for ji, window in dst.block_windows(1):
                print(f"Processing window {ji} of {os.path.basename(input_path)}: {window}")
                # Allocate an array for the window.
                data = np.empty((src.count, window.height, window.width), dtype=src.dtypes[0])

                # Reproject the corresponding portion of the source data into this window.
                reproject(
                    source=rasterio.band(src, list(range(1, src.count + 1))),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst.window_transform(window),
                    dst_crs=dst_crs,
                    resampling=Resampling.average
                )

                # Write the computed block to the destination dataset.
                dst.write(data, window=window)
            print(f"Finished processing {os.path.basename(input_path)} in {time.time() - start_time:.2f} seconds.")


def process_file_worker(filename, input_folder, output_folder):
    """Worker function to process one file."""
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    print(f"Started processing {filename}")
    reproject_in_chunks(input_path, output_path)
    print(f"Saved: {output_path}")


def main():
    input_folder = "/media/irina/data/tmp"
    output_folder = input_folder + "_10cm"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all TIFF files (case-insensitive)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]
    print("Files to process:", files)

    # Process 16 files concurrently using a multiprocessing Pool.
    with Pool(processes=10) as pool:
        pool.starmap(process_file_worker, [(f, input_folder, output_folder) for f in files])


if __name__ == "__main__":
    main()
