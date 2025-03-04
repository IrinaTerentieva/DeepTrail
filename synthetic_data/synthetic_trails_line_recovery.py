#!/usr/bin/env python3
import glob
import os
import gc
import random
import tempfile
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.coords import BoundingBox
from shapely.geometry import box
import scipy.ndimage as ndimage

# --- Options ---
inverted = False
burn_probability = 0.80  # Probability that a given trail will be processed in "burn" mode

# "Burn" widths & noise (in pixels)
burn_min_pixels = 4.5
burn_max_pixels = 7.5
burn_random_low = 0.13
burn_random_high = 0.2

# If a feature's offset_distance > 2 meters, we raise the min burn width to 6.0 pixels
conditional_burn_min = 6

# "Blend" widths & noise
blend_min_pixels = 8.0
blend_max_pixels = 12.0
blend_random_low = -0.03
blend_random_high = 0.12

# Coarse lumps scale factor for "blend"
blend_coarse_scale = 0.1
blend_small_sigma = 2

# A gap (in pixels) to ensure a few pixels are kept between trails
gap_pixels = 2

# Parameter: maximum offset (in pixels) for full scaling (adjust as needed)
max_offset_pixels = 20

# New: if offset (in meters) is less than 1.4, we use only the minimum width.
min_offset_threshold = 1.4

# Paths
vector_path = "/home/irina/HumanFootprint/DATA/manual/intermediate/Surmont_synthetic_trails_narrow.gpkg"
input_folder = "/media/irina/My Book/Surmont/nDTM/blended_with_segformer"
tif_paths = glob.glob(os.path.join(input_folder, "*blend.tif"))

print(tif_paths)

# Read trails
gdf = gpd.read_file(vector_path)

# --- Fractal noise generator for blend areas ---
def generate_fractal_noise(shape, octaves=4, persistence=0.5, low=blend_random_low, high=blend_random_high):
    noise = np.zeros(shape, dtype=np.float32)
    for i in range(octaves):
        scale = 2 ** i
        # generate noise at a lower resolution
        small_shape = (max(1, shape[0] // scale), max(1, shape[1] // scale))
        noise_small = np.random.uniform(low=low, high=high, size=small_shape).astype(np.float32)
        # upscale noise to full shape
        zoom_factors = (shape[0] / small_shape[0], shape[1] / small_shape[1])
        noise_up = ndimage.zoom(noise_small, zoom_factors, order=1)
        noise += noise_up * (persistence ** i)
    return noise

for raster_path in tif_paths:
    print(f"[INFO] Processing raster: {raster_path}")

    output_tif = raster_path.replace('.tif', '_synth_trails_v3.tif')
    output_gpkg = output_tif.replace('.tif', '.gpkg')

    if os.path.exists(output_gpkg):
        print('Skipping')
        continue

    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        transform = src.transform

        # Reproject vector if needed
        if gdf.crs != src.crs:
            print("[WARNING] CRS mismatch: reprojecting vector to match raster CRS.")
            gdf_local = gdf.to_crs(src.crs)
        else:
            gdf_local = gdf

        # Filter to intersecting features
        bounds: BoundingBox = src.bounds
        raster_polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        gdf_sub = gdf_local[gdf_local.geometry.intersects(raster_polygon)]
        if gdf_sub.empty:
            print("  [WARNING] No trails intersect this raster. Skipping...")
            continue

        data = src.read(1).astype(np.float32)
        pixel_size = abs(transform.a)  # e.g., 0.1 m per pixel

    if inverted:
        data = -data
        print("[INFO] Raster has been inverted.")

    # Create a memmap for modified data
    temp_filename = os.path.join(tempfile.gettempdir(), "data_modified.dat")
    data_modified = np.memmap(temp_filename, dtype=np.float32, mode='w+', shape=data.shape)
    data_modified[:] = data[:]
    del data
    gc.collect()

    label_records = []
    # Get unique trail IDs from the filtered vector features
    trail_ids_sub = gdf_sub['trail_id'].unique()

    # --- Randomly assign a mode to each trail ---
    # This ensures that the random split is done once per trail.
    mode_assignment = {tid: ('burn' if random.random() < burn_probability else 'blend')
                       for tid in trail_ids_sub}

    for tid in trail_ids_sub:
        # Subset for this trail
        trail_features = gdf_sub[gdf_sub['trail_id'] == tid]
        trail_geometry = trail_features.unary_union

        # Look up an attribute named "offset_distance" (in meters, if available)
        offset_distance_attr = 0
        if 'offset_distance' in trail_features.columns:
            offset_distance_attr = float(trail_features.iloc[0]['offset_distance'] or 0)
        # Compute offset in pixels (if available)
        offset_pixels = offset_distance_attr / pixel_size if offset_distance_attr else None

        # Get the pre-assigned mode for this trail
        mode = mode_assignment[tid]

        if mode == 'burn':
            # --- BURN processing ---
            # Adjust the effective minimum based on offset_distance:
            if offset_distance_attr > 2.0:
                effective_burn_min = conditional_burn_min
            else:
                effective_burn_min = burn_min_pixels

            # If offset is less than threshold, use only min width.
            if offset_distance_attr < min_offset_threshold:
                width_pixels = effective_burn_min
            elif offset_pixels is not None and offset_pixels > gap_pixels:
                # Linearly interpolate width: more offset -> wider burn
                ratio = min(offset_pixels, max_offset_pixels) / max_offset_pixels
                width_pixels_calc = effective_burn_min + (burn_max_pixels - effective_burn_min) * ratio
                candidate_max = offset_pixels - gap_pixels
                width_pixels = min(width_pixels_calc, candidate_max)
            else:
                width_pixels = random.uniform(effective_burn_min, burn_max_pixels)
            buffer_distance = width_pixels * pixel_size

            trail_buffer = trail_geometry.buffer(buffer_distance)
            label_records.append({
                "trail_id": tid,
                "mode": "burn",
                "offset_distance_attr": offset_distance_attr,
                "burn_min_in_use": effective_burn_min,
                "buffer_pixels": width_pixels,
                "buffer_distance": buffer_distance,
                "geometry": trail_buffer
            })

            # Rasterize the burn buffer
            mask = rasterize(
                [(trail_buffer, 1)],
                out_shape=(profile['height'], profile['width']),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )

            rows, cols = mask.nonzero()
            if rows.size == 0 or cols.size == 0:
                continue
            row_min, row_max = rows.min(), rows.max() + 1
            col_min, col_max = cols.min(), cols.max() + 1

            mask_crop = mask[row_min:row_max, col_min:col_max]
            shape_crop = mask_crop.shape

            random_field_crop = np.random.uniform(
                low=burn_random_low,
                high=burn_random_high,
                size=shape_crop
            )
            smooth_subtraction_crop = ndimage.gaussian_filter(random_field_crop, sigma=5)

            distance_inside_crop = ndimage.distance_transform_edt(mask_crop)
            max_distance = distance_inside_crop.max() or 1
            gradient_mask_crop = distance_inside_crop / max_distance
            adjusted_subtraction_crop = smooth_subtraction_crop * gradient_mask_crop

            region = (slice(row_min, row_max), slice(col_min, col_max))
            data_modified_region = data_modified[region]
            # Subtract the field only where mask==1 (burn)
            data_modified_region[mask_crop == 1] -= adjusted_subtraction_crop[mask_crop == 1]
            data_modified[region] = data_modified_region

            del (mask, mask_crop, random_field_crop, smooth_subtraction_crop,
                 distance_inside_crop, gradient_mask_crop, adjusted_subtraction_crop,
                 data_modified_region)
            gc.collect()

        else:
            # --- BLEND processing ---
            blend_pixels = random.uniform(blend_min_pixels, blend_max_pixels)
            buffer_distance = blend_pixels * pixel_size

            blend_buffer = trail_geometry.buffer(buffer_distance)
            label_records.append({
                "trail_id": tid,
                "mode": "blend",
                "offset_distance_attr": offset_distance_attr,
                "buffer_pixels": blend_pixels,
                "buffer_distance": buffer_distance,
                "geometry": blend_buffer
            })

            mask = rasterize(
                [(blend_buffer, 1)],
                out_shape=(profile['height'], profile['width']),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )

            rows, cols = mask.nonzero()
            if rows.size == 0 or cols.size == 0:
                continue
            row_min, row_max = rows.min(), rows.max() + 1
            col_min, col_max = cols.min(), cols.max() + 1

            mask_crop = mask[row_min:row_max, col_min:col_max]
            shape_crop = mask_crop.shape

            # Generate fractal noise for blend areas:
            random_field_crop = generate_fractal_noise(shape_crop, octaves=4, persistence=0.5,
                                                         low=blend_random_low, high=blend_random_high)

            distance_inside_crop = ndimage.distance_transform_edt(mask_crop)
            max_distance = distance_inside_crop.max() or 1
            gradient_mask_crop = distance_inside_crop / max_distance

            adjusted_field_crop = random_field_crop * gradient_mask_crop
            region = (slice(row_min, row_max), slice(col_min, col_max))
            data_modified_region = data_modified[region]
            # Blend: add the adjusted field to the data
            data_modified_region[mask_crop == 1] += adjusted_field_crop[mask_crop == 1]
            data_modified[region] = data_modified_region

            del (mask, mask_crop, random_field_crop,
                 adjusted_field_crop, distance_inside_crop, gradient_mask_crop,
                 data_modified_region)
            gc.collect()

    # Done modifying; flush memmap changes
    data_modified.flush()

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(np.array(data_modified), 1)

    # Save label polygons (with assigned mode information)
    gdf_labels = gpd.GeoDataFrame(label_records, geometry="geometry", crs=gdf_sub.crs)
    gdf_labels.to_file(output_gpkg, driver="GPKG")

    print(f"[INFO] Raster saved: {output_tif}")
    print(f"[INFO] Polygons saved: {output_gpkg}")

    del data_modified
    gc.collect()

print("[INFO] Processing complete for all matched TIF files.")
