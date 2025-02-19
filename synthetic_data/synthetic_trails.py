import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import os
from shapely.ops import unary_union
import random
import scipy.ndimage as ndimage

# Paths for the input vector (combined original/synthetic) and raster files
vector_path = "/home/irina/HumanFootprint/DATA/manual/intermediate/LiDea_pilot_synthetic_trails.gpkg"
raster_path = "file:///media/irina/My Book1/LiDea_Pilot/nDTM/LideaPilot_10cm_nDTM.tif"
output_tif = raster_path.replace('tif', 'synth_trails.tif')

# Load the combined vector file (it should have a "trail_id" column)
gdf = gpd.read_file(vector_path)

# Open the raster to get its transform, profile, and data.
with rasterio.open(raster_path) as src:
    transform = src.transform
    profile = src.profile
    # Read the single-band raster as float32 (to allow subtraction)
    data = src.read(1).astype(np.float32)
    # Compute pixel size (assuming square pixels)
    pixel_size = abs(transform.a)

# Create a copy of the raster data to modify
data_modified = data.copy()

# Prepare a list to hold label polygons (buffered trail areas)
label_records = []

# Process each unique trail (using its trail_id)
trail_ids = gdf['trail_id'].unique()

for tid in trail_ids:
    # Select all features (both original and synthetic) for this trail
    trail_features = gdf[gdf['trail_id'] == tid]
    # Merge the geometries for this trail
    trail_geometry = trail_features.unary_union

    # Generate a random buffer (in pixels) between 3 and 6; convert to map units.
    random_buffer_pixels = random.uniform(3, 6)
    buffer_distance = random_buffer_pixels * pixel_size

    # Buffer the merged trail geometry using the random buffer distance.
    trail_buffer = trail_geometry.buffer(buffer_distance)

    # Save this buffered polygon as a label for the trail.
    label_records.append({
        "trail_id": tid,
        "buffer_pixels": random_buffer_pixels,
        "buffer_distance": buffer_distance,
        "geometry": trail_buffer
    })

    # Rasterize the buffered polygon to create a mask with the same shape as the raster.
    mask = rasterize(
        [(trail_buffer, 1)],
        out_shape=data.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # Create a smooth random subtraction surface:
    # 1. Generate a random field over the entire raster with values between -0.05 and 0.1.
    random_field = np.random.uniform(low=-0.01, high=0.15, size=data.shape)
    # 2. Apply a Gaussian filter to smooth the random field (sigma=10 for smooth transitions).
    smooth_subtraction = ndimage.gaussian_filter(random_field, sigma=5)

    # Create a gradient mask inside the buffered trail:
    # Compute the distance transform on the mask (distance of each pixel inside to the boundary)
    distance_inside = ndimage.distance_transform_edt(mask)
    max_distance = np.max(distance_inside) if np.max(distance_inside) > 0 else 1
    # Normalize: pixels at the center (max distance) become 1; at the boundary, near 0.
    gradient_mask = distance_inside / max_distance

    # Combine the smooth subtraction surface with the gradient so that the center is burned more.
    adjusted_subtraction = smooth_subtraction * gradient_mask

    # Subtract the adjusted (modulated) surface from the raster values within the buffered area.
    data_modified[mask == 1] = data_modified[mask == 1] - adjusted_subtraction[mask == 1]

# Optionally, clip resulting values (e.g., to avoid negatives)
# data_modified[data_modified < 0] = 0

# Save the modified raster as a new TIFF.
with rasterio.open(output_tif, 'w', **profile) as dst:
    dst.write(data_modified, 1)

print("Synthetic trail raster saved to:", output_tif)

# Save the label polygons (buffered trails) as a GeoPackage.
# Use the same base name as the synthetic raster but with a .gpkg extension.
output_gpkg = os.path.splitext(output_tif)[0] + ".gpkg"
gdf_labels = gpd.GeoDataFrame(label_records, geometry="geometry", crs=gdf.crs)
gdf_labels.to_file(output_gpkg, driver="GPKG")

print("Label polygons saved to:", output_gpkg)
