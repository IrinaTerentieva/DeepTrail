import glob
import os
import gc
import random
import tempfile
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.ops import unary_union
import scipy.ndimage as ndimage

# --- Options ---
inverted = False  # If True, invert the raster before processing.

# Path to the vector file with trails
vector_path = "/home/irina/HumanFootprint/DATA/manual/intermediate/Surmont_synthetic_trails.gpkg"

# Collect all TIF files ending with "blended.tif"
input_folder = "/media/irina/My Book/Surmont/nDTM"
tif_paths = glob.glob(os.path.join(input_folder, "*blended.tif"))

# Load the trails GeoDataFrame (must have a "trail_id" column)
gdf = gpd.read_file(vector_path)
trail_ids = gdf['trail_id'].unique()

for raster_path in tif_paths:
    print(f"[INFO] Processing raster: {raster_path}")

    # Build output paths by appending a suffix or otherwise modifying the filename
    output_tif = raster_path.replace('.tif', '_synth_trails.tif')
    output_gpkg = output_tif.replace('.tif', '.gpkg')

    # Open raster to get profile, transform, etc.
    with rasterio.open(raster_path) as src:
        profile = src.profile
        transform = src.transform
        # Read the single band as float32
        data = src.read(1).astype(np.float32)
        pixel_size = abs(transform.a)

    # Optionally invert
    if inverted:
        max_val = data.max()
        data = - data
        print("[INFO] Raster has been inverted.")

    # Create memmap to avoid large in-memory array
    temp_filename = os.path.join(tempfile.gettempdir(), "data_modified.dat")
    data_modified = np.memmap(
        temp_filename, dtype=np.float32, mode='w+', shape=data.shape
    )
    data_modified[:] = data[:]
    del data
    gc.collect()

    # Prepare a list to hold label polygons (buffered trail areas)
    label_records = []

    # Process each unique trail
    for tid in trail_ids:
        # Subset geometry for this trail
        trail_features = gdf[gdf['trail_id'] == tid]
        trail_geometry = trail_features.union_all()

        # Random buffer range in pixels
        random_buffer_pixels = random.uniform(4, 6.5)
        buffer_distance = random_buffer_pixels * pixel_size

        # Buffer the merged trail geometry
        trail_buffer = trail_geometry.buffer(buffer_distance)

        # Store for output GPKG
        label_records.append({
            "trail_id": tid,
            "buffer_pixels": random_buffer_pixels,
            "buffer_distance": buffer_distance,
            "geometry": trail_buffer
        })

        # Rasterize the buffered polygon
        mask = rasterize(
            [(trail_buffer, 1)],
            out_shape=(profile['height'], profile['width']),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Determine bounding box of the mask to reduce area of processing
        rows, cols = np.where(mask == 1)
        if len(rows) == 0 or len(cols) == 0:
            continue
        row_min, row_max = rows.min(), rows.max() + 1
        col_min, col_max = cols.min(), cols.max() + 1

        # Crop the mask
        mask_crop = mask[row_min:row_max, col_min:col_max]
        shape_crop = mask_crop.shape

        # Create and smooth random field
        random_field_crop = np.random.uniform(low=-0.02, high=0.2, size=shape_crop)
        smooth_subtraction_crop = ndimage.gaussian_filter(random_field_crop, sigma=5)

        # Distance transform
        distance_inside_crop = ndimage.distance_transform_edt(mask_crop)
        max_distance = distance_inside_crop.max() if distance_inside_crop.max() > 0 else 1
        gradient_mask_crop = distance_inside_crop / max_distance

        # Combine random field with distance gradient
        adjusted_subtraction_crop = smooth_subtraction_crop * gradient_mask_crop

        # Update memmap on the cropped region
        region = (slice(row_min, row_max), slice(col_min, col_max))
        data_modified_region = data_modified[region]
        data_modified_region[mask_crop == 1] -= adjusted_subtraction_crop[mask_crop == 1]
        data_modified[region] = data_modified_region

        # Clean up
        del (mask, mask_crop, random_field_crop, smooth_subtraction_crop,
             distance_inside_crop, gradient_mask_crop, adjusted_subtraction_crop,
             data_modified_region)
        gc.collect()

    # Flush memmap
    data_modified.flush()

    # Write final output raster
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(np.array(data_modified), 1)
    print(f"[INFO] Synthetic trail raster saved to: {output_tif}")

    # Write label polygons (buffered areas)
    gdf_labels = gpd.GeoDataFrame(label_records, geometry="geometry", crs=gdf.crs)
    gdf_labels.to_file(output_gpkg, driver="GPKG")
    print(f"[INFO] Label polygons saved to: {output_gpkg}")

    # Clean up memmap
    del data_modified
    gc.collect()

print("[INFO] Processing complete for all matched TIF files.")
