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
from shapely.affinity import translate
import scipy.ndimage as ndimage

# --- Options & Parameters ---
buffer_meters = 8  # Buffer distance in meters for the centerline.
shift_meters = 5  # Shift distance (in meters) for sampling adjacent raster.
gaussian_sigma_blur = 2  # Sigma for light Gaussian blur after blending.
blend_weight = 3  # Weight factor for blending blurred values.
inverted = False

def increase_low_values(x):
    if x <= -0.1:
        return x + 0.2
    elif -0.1 < x <= -0.05:
        return x + 0.1
    elif -0.5 < x <= -0.02:
        return x + 0.04
    elif -0.05 < x <= 0.03:
        return x + 0.02
    elif 0.03 < x <= 0.1:
        return x - 0.05
    elif 0.1 < x <= 0.3:
        return x - 0.1
    else:
        return x

vec_increase_low_values = np.vectorize(increase_low_values)

# --- Paths ---
vector_path = '/media/irina/My Book/Surmont/vector_data/FLM/FLM_centerline_Surmont.gpkg'
raster_folder = '/media/irina/My Book/Surmont/nDTM'

tif_files = glob.glob(os.path.join(raster_folder, '*.tif'))
print(f"[INFO] Found {len(tif_files)} TIFF files to process.")

for raster_path in tif_files:
    output_tif = raster_path.replace('.tif', '_blended.tif')

    # If the output file already exists, skip
    if os.path.exists(output_tif):
        print(f"[INFO] {output_tif} already exists. Skipping processing.")
        continue

    # --- Load the Vector & Clip to Raster Boundaries ---
    gdf = gpd.read_file(vector_path)
    # Remove duplicate geometries.
    gdf = gdf[~gdf.geometry.apply(lambda g: g.wkt).duplicated()]
    print("After duplicate removal, number of features:", len(gdf))
    print("Geometry types:", gdf.geometry.geom_type.unique())

    # Open the raster to get its bounds.
    with rasterio.open(raster_path) as src:
        raster_bounds = src.bounds
    gdf = gdf.clip(raster_bounds)

    if 'trail_id' not in gdf.columns:
        gdf['trail_id'] = range(1, len(gdf) + 1)

    # --- Open the Raster ---
    with rasterio.open(raster_path) as src:
        transform = src.transform
        profile = src.profile
        data = src.read(1).astype(np.float32)
        pixel_size = abs(transform.a)

    # --- (Optional) Inversion ---
    if inverted:
        max_val = data.max()
        data = max_val - data
        print("[INFO] Raster has been inverted.")

    # --- Create Memory-Mapped Raster ---
    temp_filename = os.path.join(tempfile.gettempdir(), "data_modified.dat")
    data_modified = np.memmap(temp_filename, dtype=np.float32, mode='w+', shape=data.shape)
    data_modified[:] = data[:]
    del data
    gc.collect()

    # --- BLENDING STEP: Shift Buffer Sampling & Increase Low Values ---
    print("[INFO] Starting blending step: sampling adjacent raster and blending.")
    label_records = []
    trail_ids = gdf['trail_id'].unique()

    # Calculate shift in pixel units.
    shift_pixels = shift_meters / pixel_size
    shift_pixels_x = int(round(shift_pixels))  # x offset in pixels
    shift_pixels_y = int(round(shift_pixels))  # y offset in pixels

    for tid in trail_ids:
        trail_features = gdf[gdf['trail_id'] == tid]
        if trail_features.empty:
            continue
        trail_geometry = trail_features.unary_union

        # Create the original buffer.
        trail_buffer = trail_geometry.buffer(buffer_meters)
        label_records.append({
            "trail_id": tid,
            "buffer_distance": buffer_meters,
            "geometry": trail_buffer
        })

        # Shifted version of the buffer: NW => negative x, positive y.
        shifted_buffer = translate(trail_buffer, xoff=-shift_meters, yoff=shift_meters)

        # Rasterize both the original and shifted buffers.
        mask_orig = rasterize(
            [(trail_buffer, 1)],
            out_shape=(profile['height'], profile['width']),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        mask_shift = rasterize(
            [(shifted_buffer, 1)],
            out_shape=(profile['height'], profile['width']),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Determine bounding box of the original mask.
        rows, cols = np.where(mask_orig == 1)
        if len(rows) == 0 or len(cols) == 0:
            continue
        row_min, row_max = rows.min(), rows.max() + 1
        col_min, col_max = cols.min(), cols.max() + 1
        region = (slice(row_min, row_max), slice(col_min, col_max))

        # Extract relevant crops
        mask_crop_orig = mask_orig[row_min:row_max, col_min:col_max]
        mask_crop_shift = mask_shift[row_min:row_max, col_min:col_max]
        region_data = data_modified[region].copy()

        # Sample region_data for the shifted buffer
        sampled_region = np.copy(region_data)
        moved_region = np.roll(sampled_region, shift=(shift_pixels_y, shift_pixels_x), axis=(0, 1))

        # Transform low values in the shifted sample
        transformed_region = vec_increase_low_values(moved_region)

        # Build a gradient mask from the original buffer
        distance_inside = ndimage.distance_transform_edt(mask_crop_orig)
        max_distance = distance_inside.max() if distance_inside.max() > 0 else 1
        gradient_mask = distance_inside / max_distance  # 0 at edge, 1 at center

        # Blend
        blended_region = region_data * (1 - gradient_mask) + transformed_region * gradient_mask

        # Light blur
        blurred_region = ndimage.gaussian_filter(blended_region, sigma=gaussian_sigma_blur)
        final_region = region_data * (1 - 0.5 * gradient_mask) + blurred_region * (0.5 * gradient_mask)

        # Update only pixels within the original buffer
        region_data[mask_crop_orig == 1] = final_region[mask_crop_orig == 1]
        data_modified[region] = region_data

        del (mask_orig, mask_shift, mask_crop_orig, mask_crop_shift, region_data, sampled_region,
             moved_region, transformed_region, distance_inside, gradient_mask,
             blended_region, blurred_region, final_region)
        gc.collect()

    print("[INFO] Blending step completed using shifted buffer sampling with low-value increase.")

    # --- Post-Processing: Set Out-of-Range Values to 0 ---
    print("[INFO] Post-processing: setting values outside [-5,5] to 0.")
    final_array = np.array(data_modified)
    final_array[(final_array < -5) | (final_array > 5)] = 0
    data_modified.flush()

    # --- Save Final Raster ---
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(final_array, 1)
    print("Final blended raster saved to:", output_tif)

    # Clean up
    del data_modified, final_array
    gc.collect()

print("[INFO] Processing complete for all matched TIF files.")
