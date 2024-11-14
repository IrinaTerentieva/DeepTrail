import os
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import numpy as np


def tile_raster(input_folder, output_folder, num_tiles=16):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            input_raster_path = os.path.join(input_folder, filename)
            print(f"Processing {input_raster_path}")

            with rasterio.open(input_raster_path) as src:
                width = src.width
                height = src.height
                tile_size_x = width // int(np.sqrt(num_tiles))
                tile_size_y = height // int(np.sqrt(num_tiles))

                # Loop to create 16 tiles (4x4 grid) from the image
                count = 1
                for i in range(0, height, tile_size_y):
                    for j in range(0, width, tile_size_x):
                        window = Window(j, i, tile_size_x, tile_size_y)
                        transform = src.window_transform(window)

                        output_tile_path = os.path.join(
                            output_folder,
                            f"{os.path.splitext(filename)[0]}_tile{count}.tif"
                        )

                        # Read and write the tile
                        with rasterio.open(
                                output_tile_path,
                                'w',
                                driver='GTiff',
                                height=tile_size_y,
                                width=tile_size_x,
                                count=src.count,
                                dtype=src.dtypes[0],
                                crs=src.crs,
                                transform=transform
                        ) as dst:
                            for band in range(1, src.count + 1):
                                dst.write(src.read(band, window=window), band)

                        count += 1
                print(f"Tiled into {count - 1} pieces and saved to {output_folder}")


# Usage:
input_folder = '/media/irro/All/HumanFootprint/DATA/Products/Unet/LiDea'
output_folder = '/media/irro/All/HumanFootprint/DATA/Products/Unet/LiDea/LiDea_Tiles'

tile_raster(input_folder, output_folder)
