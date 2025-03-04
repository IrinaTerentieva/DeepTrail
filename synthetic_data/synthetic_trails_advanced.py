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
from shapely.geometry import box, LineString, MultiLineString, Polygon
import scipy.ndimage as ndimage
from scipy.interpolate import splprep, splev

# --- Options ---
inverted = False
burn_probability = 0.8  # Probability that a given trail will be processed in "burn" mode

# "Burn" widths & noise (in pixels)
burn_min_pixels = 4
burn_max_pixels = 6
burn_random_low = 0.12
burn_random_high = 0.15

# If a feature's offset_distance > 2 meters, we raise the min burn width to 6.0 pixels
conditional_burn_min = 6

# "Blend" widths & noise
blend_min_pixels = 8.0
blend_max_pixels = 12.0
blend_random_low = 0.02
blend_random_high = 0.12

# Coarse lumps scale factor for "blend"
blend_coarse_scale = 0.1
blend_small_sigma = 3

# A gap (in pixels) to ensure a few pixels are kept between trails
gap_pixels = 2

# Parameter: maximum offset (in pixels) for full scaling (adjust as needed)
max_offset_pixels = 20

# If offset (in meters) is less than 1.4, we use only the minimum width.
min_offset_threshold = 1.4

# New output folder
out_folder = "/media/irina/My Book/Surmont/nDTM_synth_trails_v.3.2"
os.makedirs(out_folder, exist_ok=True)

# Paths for inputs
vector_path = "/home/irina/HumanFootprint/DATA/manual/intermediate/Surmont_synthetic_trails_narrow.gpkg"
input_folder = "/media/irina/My Book/Surmont/nDTM/blended_with_segformer"
tif_paths = glob.glob(os.path.join(input_folder, "*blend.tif"))

print("Input TIF paths:")
print(tif_paths)

# Read trails
gdf = gpd.read_file(vector_path)

import random
import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
from scipy.interpolate import splprep, splev


def strong_smooth_geometry(geom, smoothing_level=None):
    """
    Smooths a geometry using a two-step process:
      1. Simplify the geometry to remove small-scale details.
      2. Apply spline interpolation to produce a smooth curve.

    The smoothing_level parameter can be 'low', 'medium', or 'strong'.
    If not provided, one is chosen randomly with equal probability.

    Preset parameters:
      - 'low': minimal smoothing (preserves more detail)
          smoothing_factor = 20, num_points = 100, simplify_tolerance = 0.2
      - 'medium': moderate smoothing
          smoothing_factor = 50, num_points = 50, simplify_tolerance = 0.8
      - 'strong': heavy smoothing (more generalized, fewer points)
          smoothing_factor = 100, num_points = 20, simplify_tolerance = 1.5
    """

    if smoothing_level == 'low':
        smoothing_factor = 20
        num_points = 100
        simplify_tolerance = 0.2
    elif smoothing_level == 'medium':
        smoothing_factor = 100
        num_points = 20
        simplify_tolerance = 0.5
    else:
        # Default fallback if an unknown level is provided.
        smoothing_factor = 20
        num_points = 50
        simplify_tolerance = 0.2

    if geom.is_empty:
        return geom

    # Simplify the geometry first.
    simplified = geom.simplify(simplify_tolerance, preserve_topology=True)

    # Process based on geometry type.
    if simplified.geom_type == 'LineString':
        x, y = simplified.xy
        try:
            tck, u = splprep([x, y], s=smoothing_factor)
            unew = np.linspace(0, 1, num_points)
            out = splev(unew, tck)
            return LineString(zip(out[0], out[1]))
        except Exception:
            return simplified
    elif simplified.geom_type == 'MultiLineString':
        smoothed = [strong_smooth_geometry(line, smoothing_level) for line in simplified.geoms]
        return MultiLineString(smoothed)
    elif simplified.geom_type == 'Polygon':
        smoothed_exterior = strong_smooth_geometry(simplified.exterior, smoothing_level)
        return Polygon(smoothed_exterior)
    else:
        return simplified


# --- Fractal noise generator for blend areas ---
def generate_fractal_noise(shape, octaves=4, persistence=0.5, low=blend_random_low, high=blend_random_high):
    noise = np.zeros(shape, dtype=np.float32)
    for i in range(octaves):
        scale = 2 ** i
        small_shape = (max(1, shape[0] // scale), max(1, shape[1] // scale))
        noise_small = np.random.uniform(low=low, high=high, size=small_shape).astype(np.float32)
        zoom_factors = (shape[0] / small_shape[0], shape[1] / small_shape[1])
        noise_up = ndimage.zoom(noise_small, zoom_factors, order=1)
        noise += noise_up * (persistence ** i)
    return noise

for raster_path in tif_paths:
    print(f"[INFO] Processing raster: {raster_path}")

    base_name = os.path.basename(raster_path)
    output_tif = os.path.join(out_folder, base_name.replace('.tif', '_synth_trails_v3.2.tif'))
    output_gpkg = output_tif.replace('.tif', '.gpkg')

    if os.path.exists(output_gpkg):
        print("Skipping (output exists):", output_gpkg)
        continue

    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        transform = src.transform

        if gdf.crs != src.crs:
            print("[WARNING] CRS mismatch: reprojecting vector to match raster CRS.")
            gdf_local = gdf.to_crs(src.crs)
        else:
            gdf_local = gdf

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

    temp_filename = os.path.join(tempfile.gettempdir(), "data_modified.dat")
    data_modified = np.memmap(temp_filename, dtype=np.float32, mode='w+', shape=data.shape)
    data_modified[:] = data[:]
    del data
    gc.collect()

    label_records = []
    trail_ids_sub = gdf_sub['trail_id'].unique()

    mode_assignment = {tid: ('burn' if random.random() < burn_probability else 'blend')
                       for tid in trail_ids_sub}

    for tid in trail_ids_sub:
        trail_features = gdf_sub[gdf_sub['trail_id'] == tid]

        raw_geom = trail_features.unary_union

        smoothing_level = random.choice(['low', 'medium', 'no'])
        if smoothing_level != 'no':
            trail_geometry = strong_smooth_geometry(raw_geom, smoothing_level=smoothing_level)
        else:
            trail_geometry = raw_geom

        offset_distance_attr = 0
        if 'offset_distance' in trail_features.columns:
            offset_distance_attr = float(trail_features.iloc[0]['offset_distance'] or 0)
        offset_pixels = offset_distance_attr / pixel_size if offset_distance_attr else None

        mode = mode_assignment[tid]

        if mode == 'burn':
            if offset_distance_attr > 2.0:
                effective_burn_min = conditional_burn_min
            else:
                effective_burn_min = burn_min_pixels

            if offset_distance_attr < min_offset_threshold:
                width_pixels = effective_burn_min
            elif offset_pixels is not None and offset_pixels > gap_pixels:
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
                "smoothing_level": smoothing_level,
                "geometry": trail_buffer
            })

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

            # Generate the burn offset values:
            random_field_crop = np.random.uniform(low=burn_random_low, high=burn_random_high, size=shape_crop)
            smooth_subtraction_crop = ndimage.gaussian_filter(random_field_crop, sigma=5)
            distance_inside_crop = ndimage.distance_transform_edt(mask_crop)
            max_distance = distance_inside_crop.max() or 1
            gradient_mask_crop = distance_inside_crop / max_distance
            adjusted_subtraction_crop = smooth_subtraction_crop * gradient_mask_crop

            # Save the average burn value for this patch as an attribute:
            burn_value_avg = np.mean(adjusted_subtraction_crop[mask_crop == 1])

            region = (slice(row_min, row_max), slice(col_min, col_max))
            data_modified_region = data_modified[region]
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
                "mode": "burn",
                "offset_distance_attr": offset_distance_attr,
                "burn_min_in_use": effective_burn_min,
                "buffer_pixels": width_pixels,
                "buffer_distance": buffer_distance,
                "smoothing_level": smoothing_level,
                "avg_burn_value": burn_value_avg,   # new attribute
                "geometry": trail_buffer
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

            # Generate fractal noise with adjusted parameters for coarser, more blocky patterns.
            random_field_crop = generate_fractal_noise(shape_crop, octaves=3, persistence=0.3,
                                                         low=blend_random_low, high=blend_random_high)
            # Apply a strong Gaussian filter to smooth out fine details, yielding larger contiguous areas.
            random_field_crop = ndimage.gaussian_filter(random_field_crop, sigma=10)
            # Optionally adjust scaling:
            random_field_crop = random_field_crop * 1.2 - 0.02

            distance_inside_crop = ndimage.distance_transform_edt(mask_crop)
            max_distance = distance_inside_crop.max() or 1
            gradient_mask_crop = distance_inside_crop / max_distance

            adjusted_field_crop = random_field_crop * gradient_mask_crop
            region = (slice(row_min, row_max), slice(col_min, col_max))
            data_modified_region = data_modified[region]
            data_modified_region[mask_crop == 1] += adjusted_field_crop[mask_crop == 1]
            data_modified[region] = data_modified_region

            del (mask, mask_crop, random_field_crop,
                 adjusted_field_crop, distance_inside_crop, gradient_mask_crop,
                 data_modified_region)
            gc.collect()

    data_modified.flush()

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(np.array(data_modified), 1)

    gdf_labels = gpd.GeoDataFrame(label_records, geometry="geometry", crs=gdf_sub.crs)
    gdf_labels.to_file(output_gpkg, driver="GPKG")

    print(f"[INFO] Raster saved: {output_tif}")
    print(f"[INFO] Polygons saved: {output_gpkg}")

    del data_modified
    gc.collect()

print("[INFO] Processing complete for all matched TIF files.")
