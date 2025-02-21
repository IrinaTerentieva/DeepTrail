import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import os
from shapely.ops import unary_union
import random
import scipy.ndimage as ndimage
import gc
import tempfile

# --- Options ---
inverted = True  # If True, invert the raster before processing.

# Paths for the input vector (combined original/synthetic) and raster files
vector_path = "/home/irina/HumanFootprint/DATA/manual/intermediate/LiDea_pilot_synthetic_trails.gpkg"
raster_path = "file:///media/irina/My Book1/LiDea_Pilot/nDTM/LideaPilot_10cm_nDTM.tif"

output_tif = raster_path.replace('.tif', 'synth_trails_v4.tif')

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

# If the inverted option is enabled, invert the raster.
if inverted:
    max_val = data.max()
    data = - data
    print("[INFO] Raster has been inverted.")

# Create a memory-mapped copy of the raster data to reduce RAM usage.
temp_filename = "/media/irina/My Book1/LiDea_Pilot/temp/data_modified.dat"
data_modified = np.memmap(temp_filename, dtype=np.float32, mode='w+', shape=data.shape)
data_modified[:] = data[:]
del data  # free the original data array
gc.collect()

# Prepare a list to hold label polygons (buffered trail areas)
label_records = []

# Process each unique trail (using its trail_id)
trail_ids = gdf['trail_id'].unique()

for tid in trail_ids:
    # Select all features (both original and synthetic) for this trail
    trail_features = gdf[gdf['trail_id'] == tid]
    # Merge the geometries for this trail
    trail_geometry = trail_features.union_all()

    # Generate a random buffer (in pixels) between 3 and 6; convert to map units.
    random_buffer_pixels = random.uniform(5, 6.5)
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
        out_shape=(profile['height'], profile['width']),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # Instead of processing the full raster, determine the bounding box of the mask.
    rows, cols = np.where(mask == 1)
    if len(rows) == 0 or len(cols) == 0:
        # Skip if the mask is empty
        continue
    row_min, row_max = rows.min(), rows.max() + 1
    col_min, col_max = cols.min(), cols.max() + 1

    # Crop the mask and define the working region
    mask_crop = mask[row_min:row_max, col_min:col_max]
    shape_crop = mask_crop.shape

    # Create a random field on just the cropped area
    random_field_crop = np.random.uniform(low=-0.02, high=0.2, size=shape_crop)
    # Smooth the random field with a Gaussian filter (sigma=5 for moderate smoothing)
    smooth_subtraction_crop = ndimage.gaussian_filter(random_field_crop, sigma=5)
    # Compute a distance transform on the cropped mask
    distance_inside_crop = ndimage.distance_transform_edt(mask_crop)
    max_distance = distance_inside_crop.max() if distance_inside_crop.max() > 0 else 1
    # Create a gradient mask: center gets values near 1; boundaries near 0
    gradient_mask_crop = distance_inside_crop / max_distance
    # Multiply the smooth subtraction field by the gradient mask
    adjusted_subtraction_crop = smooth_subtraction_crop * gradient_mask_crop

    # Update only the cropped portion of the memmap where mask is 1
    region = (slice(row_min, row_max), slice(col_min, col_max))
    # Subtract the adjusted subtraction values where mask_crop==1
    data_modified_region = data_modified[region]
    data_modified_region[mask_crop == 1] = data_modified_region[mask_crop == 1] - adjusted_subtraction_crop[mask_crop == 1]
    data_modified[region] = data_modified_region  # update the memmap

    # Clean up intermediate arrays and force garbage collection.
    del mask, mask_crop, random_field_crop, smooth_subtraction_crop, distance_inside_crop, gradient_mask_crop, adjusted_subtraction_crop, data_modified_region
    gc.collect()

# Flush the memmap to ensure all data is written.
data_modified.flush()

# Save the modified raster as a new TIFF.
with rasterio.open(output_tif, 'w', **profile) as dst:
    dst.write(np.array(data_modified), 1)

print("Synthetic trail raster saved to:", output_tif)

# Save the label polygons (buffered trails) as a GeoPackage.
# Use the same base name as the synthetic raster but with a .gpkg extension.
output_gpkg = os.path.splitext(output_tif)[0] + ".gpkg"
gdf_labels = gpd.GeoDataFrame(label_records, geometry="geometry", crs=gdf.crs)
gdf_labels.to_file(output_gpkg, driver="GPKG")
print("Label polygons saved to:", output_gpkg)
