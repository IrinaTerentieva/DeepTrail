import rasterio
import numpy as np
from rasterio.windows import Window

# Input VRT file and output TIFF file paths
input_path = "/media/irina/My Book/Surmont/raw/Surmont_nDTM10cm_2022.vrt"
output_path = "/media/irina/My Book/Surmont/raw/Surmont_nDTM10cm_2022.tif"

# Open the input dataset
with rasterio.open(input_path) as src:
    # Copy the metadata and update for optimized output
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        compress="DEFLATE",  # Use DEFLATE compression
        predictor=2,  # Predictor for floating point data
        tiled=True,  # Enable tiling for better performance
        blockxsize=256,  # Tile width (adjust as needed)
        blockysize=256,  # Tile height (adjust as needed)
        dtype=rasterio.float32,  # Data type
        nodata=None  # Set nodata if desired
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        # Process the dataset in blocks (tiles)
        for ji, window in src.block_windows(1):
            # Read the block data from band 1
            data = src.read(1, window=window)
            # Clip values to the range [-1, 1]
            data = np.clip(data, -1, 1)
            # Write the processed block to the output file
            dst.write(data, 1, window=window)

print(f"Mosaic saved to: {output_path}")
