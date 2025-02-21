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
burn_probability = 0.80

# "Burn" widths & noise
# We'll still define these global defaults:
burn_min_pixels = 4.0
burn_max_pixels = 6.5
burn_random_low = -0.02
burn_random_high = 0.2

# If a feature's offset_distance > 2, we raise the min burn width to 5.0
conditional_burn_min = 5.5

# "Blend" widths & noise
blend_min_pixels = 8.0
blend_max_pixels = 12.0
blend_random_low = -0.03
blend_random_high = 0.12

# Coarse lumps scale factor for "blend"
blend_coarse_scale = 0.1
blend_small_sigma = 2

# Paths
# vector_path = "/home/irina/HumanFootprint/DATA/manual/intermediate/Surmont_synthetic_trails.gpkg"
# input_folder = "/media/irina/My Book/Surmont/nDTM"
# tif_paths = glob.glob(os.path.join(input_folder, "*blended.tif"))

vector_path = "/home/irina/HumanFootprint/DATA/manual/intermediate/LiDea_pilot_synthetic_trails.gpkg"
input_folder = "/media/irina/My Book1/LiDea_Pilot/nDTM"
tif_paths = glob.glob(os.path.join(input_folder, "LideaPilot_10cm_nDTM.tif"))

# Read trails
gdf = gpd.read_file(vector_path)

for raster_path in tif_paths:
    print(f"[INFO] Processing raster: {raster_path}")

    output_tif = raster_path.replace('.tif', '_synth_trails_v5.tif')
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
        pixel_size = abs(transform.a)

    if inverted:
        data = -data
        print("[INFO] Raster has been inverted.")

    # memmap
    temp_filename = os.path.join(tempfile.gettempdir(), "data_modified.dat")
    data_modified = np.memmap(temp_filename, dtype=np.float32, mode='w+', shape=data.shape)
    data_modified[:] = data[:]
    del data
    gc.collect()

    label_records = []
    trail_ids_sub = gdf_sub['trail_id'].unique()

    for tid in trail_ids_sub:
        # Subset for this trail
        trail_features = gdf_sub[gdf_sub['trail_id'] == tid]
        trail_geometry = trail_features.unary_union

        # ------------------------------------------------------------------
        # We'll look up an attribute named "offset_distance" (example name).
        # If offset_distance > 2.0, use a bigger burn_min_pixels.
        # If multiple features share the same trail_id, pick the first.
        # Adjust as needed for your data schema.
        # ------------------------------------------------------------------
        offset_distance_attr = 0
        if 'offset_distance' in trail_features.columns:
            offset_distance_attr = float(trail_features.iloc[0]['offset_distance'] or 0)

        # Decide burn vs blend
        if random.random() < burn_probability:
            # --- BURN ---
            if offset_distance_attr > 2.0:
                local_burn_min = conditional_burn_min
            else:
                local_burn_min = burn_min_pixels

            width_pixels = random.uniform(local_burn_min, burn_max_pixels)
            buffer_distance = width_pixels * pixel_size

            trail_buffer = trail_geometry.buffer(buffer_distance)
            label_records.append({
                "trail_id": tid,
                "mode": "burn",
                "offset_distance_attr": offset_distance_attr,
                "burn_min_in_use": local_burn_min,
                "buffer_pixels": width_pixels,
                "buffer_distance": buffer_distance,
                "geometry": trail_buffer
            })

            # Rasterize
            mask = rasterize(
                [(trail_buffer, 1)],
                out_shape=(profile['height'], profile['width']),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )

            rows, cols = np.where(mask == 1)
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
            data_modified_region[mask_crop == 1] -= adjusted_subtraction_crop[mask_crop == 1]
            data_modified[region] = data_modified_region

            del (mask, mask_crop, random_field_crop, smooth_subtraction_crop,
                 distance_inside_crop, gradient_mask_crop, adjusted_subtraction_crop,
                 data_modified_region)
            gc.collect()

        else:
            # --- BLEND ---
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

            rows, cols = np.where(mask == 1)
            if rows.size == 0 or cols.size == 0:
                continue
            row_min, row_max = rows.min(), rows.max() + 1
            col_min, col_max = cols.min(), cols.max() + 1

            mask_crop = mask[row_min:row_max, col_min:col_max]
            shape_crop = mask_crop.shape

            small_h = max(1, int(shape_crop[0] * blend_coarse_scale))
            small_w = max(1, int(shape_crop[1] * blend_coarse_scale))

            random_small = np.random.uniform(
                low=blend_random_low,
                high=blend_random_high,
                size=(small_h, small_w)
            )
            random_small = ndimage.gaussian_filter(random_small, sigma=blend_small_sigma)

            zoom_h = shape_crop[0] / small_h
            zoom_w = shape_crop[1] / small_w
            random_field_crop = ndimage.zoom(random_small, (zoom_h, zoom_w), order=1)

            distance_inside_crop = ndimage.distance_transform_edt(mask_crop)
            max_distance = distance_inside_crop.max() or 1
            gradient_mask_crop = distance_inside_crop / max_distance

            adjusted_field_crop = random_field_crop * gradient_mask_crop
            region = (slice(row_min, row_max), slice(col_min, col_max))
            data_modified_region = data_modified[region]
            data_modified_region[mask_crop == 1] += adjusted_field_crop[mask_crop == 1]
            data_modified[region] = data_modified_region

            del (mask, mask_crop, random_small, random_field_crop,
                 adjusted_field_crop, distance_inside_crop, gradient_mask_crop,
                 data_modified_region)
            gc.collect()

    # Done modifying
    data_modified.flush()

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(np.array(data_modified), 1)

    # Save label polygons
    gdf_labels = gpd.GeoDataFrame(label_records, geometry="geometry", crs=gdf_sub.crs)
    gdf_labels.to_file(output_gpkg, driver="GPKG")

    print(f"[INFO] Raster saved: {output_tif}")
    print(f"[INFO] Polygons saved: {output_gpkg}")

    del data_modified
    gc.collect()

print("[INFO] Processing complete for all matched TIF files.")
