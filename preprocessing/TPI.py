import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import generic_filter
from multiprocessing import Pool, cpu_count

def tpi_function(window):
    """
    Compute TPI for a given window.
    """
    center = window[len(window) // 2]
    mean_surrounding = np.nanmean(window)
    return center - mean_surrounding

def process_chunk(args):
    """
    Process a chunk of the raster for parallel execution.
    """
    dtm_chunk, window_size = args
    return generic_filter(dtm_chunk, tpi_function, size=window_size, mode='nearest')

def calculate_tpi_parallel(input_raster, output_raster, window_size=5, num_workers=16):
    """
    Calculate Topographic Position Index (TPI) from a DTM raster using multiprocessing.

    Args:
        input_raster (str): Path to the input DTM file.
        output_raster (str): Path to save the output TPI file.
        window_size (int): Size of the moving window (default 5x5).
        num_workers (int): Number of CPU cores to use (default 16).
    """
    with rasterio.open(input_raster) as src:
        dtm = src.read(1)  # Read the first band
        profile = src.profile  # Copy metadata

        # Split the raster into chunks for parallel processing
        num_splits = min(num_workers, cpu_count())  # Limit to available cores
        dtm_chunks = np.array_split(dtm, num_splits, axis=0)  # Split along rows

        # Run parallel processing
        with Pool(processes=num_splits) as pool:
            results = pool.map(process_chunk, [(chunk, window_size) for chunk in dtm_chunks])

        # Reconstruct the processed raster
        tpi = np.vstack(results)

        # Update metadata
        profile.update(dtype=rasterio.float32, nodata=np.nan)

        # Save the result
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(tpi.astype(rasterio.float32), 1)

# Paths
input_dtm_path = "/media/irina/My Book1/Conoco/DATA/Drone_PPC_2024/ndtm/Area_C_combined_ncdtm.tif"
output_tpi_path = "/media/irina/My Book1/Conoco/DATA/Drone_PPC_2024/ndtm/Area_C_combined_tpi.tif"

# Run TPI calculation with multiprocessing
calculate_tpi_parallel(input_dtm_path, output_tpi_path, window_size=15, num_workers=16)

print(f"TPI raster saved at: {output_tpi_path}")
