import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import os

# Define paths
input_gpkg = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label_connected_mean.gpkg'
original_tif = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm/1_label.tif'
output_tif = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label_connected.tif'

# Load the connected lines from the GeoPackage
gdf = gpd.read_file(input_gpkg)

# Buffer the lines by 1 pixel (based on the original raster resolution)
with rasterio.open(original_tif) as src:
    pixel_size = src.res[0]  # Assume square pixels
    transform = src.transform
    width, height = src.width, src.height
    crs = src.crs

# Apply a buffer of 1 pixel
gdf['geometry'] = gdf.buffer(pixel_size)

# Create an empty array for rasterization
raster_shape = (height, width)

# Rasterize the buffered lines, setting them to a value of 1
rasterized_lines = rasterize(
    [(geom, 2) for geom in gdf.geometry],
    out_shape=raster_shape,
    transform=transform,
    fill=0,
    dtype='uint8'
)

# Save the rasterized output to a new GeoTIFF
with rasterio.open(
        output_tif, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='uint8',
        crs=crs,
        transform=transform) as dst:
    dst.write(rasterized_lines, 1)

print(f"Rasterized file saved to {output_tif}")
