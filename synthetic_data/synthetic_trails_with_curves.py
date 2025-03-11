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
from shapely.affinity import translate
import scipy.ndimage as ndimage
from scipy.interpolate import splprep, splev

# --- Options (unchanged) ---
inverted = False
burn_probability = 0.8

burn_min_pixels = 4
burn_max_pixels = 6  # still used for main trails, but extras will max out at 5
burn_random_low = 0.12
burn_random_high = 0.18
conditional_burn_min = 6

blend_min_pixels = 8.0
blend_max_pixels = 12.0
blend_random_low = 0.02
blend_random_high = 0.12
blend_coarse_scale = 0.1
blend_small_sigma = 2

gap_pixels = 2
max_offset_pixels = 20
min_offset_threshold = 1.4

out_folder = "/media/irina/My Book/Surmont/nDTM_synth_trails_v.3.22"
os.makedirs(out_folder, exist_ok=True)

vector_path = "/home/irina/HumanFootprint/DATA/manual/intermediate/Surmont_synthetic_trails_narrow.gpkg"
input_folder = "/media/irina/My Book/Surmont/nDTM/blended_with_segformer"
tif_paths = glob.glob(os.path.join(input_folder, "*blend.tif"))

print("Input TIF paths:")
print(tif_paths)

# Main trails
gdf = gpd.read_file(vector_path)

# Extra trails
extra_trails_path = "/media/irina/My Book/Surmont/manual/art_trail_variants.gpkg"
gdf_extra = gpd.read_file(extra_trails_path)
gdf_extra = gdf_extra[gdf_extra.geometry.notnull()]

def strong_smooth_geometry(geom, smoothing_level=None):
    """ Same as your original function. """
    if smoothing_level == 'low':
        smoothing_factor = 20
        num_points = 100
        simplify_tolerance = 0.2
    elif smoothing_level == 'medium':
        smoothing_factor = 50
        num_points = 50
        simplify_tolerance = 0.8
    elif smoothing_level == 'strong':
        smoothing_factor = 100
        num_points = 20
        simplify_tolerance = 1.5
    else:
        smoothing_factor = 20
        num_points = 50
        simplify_tolerance = 0.2

    if geom.is_empty:
        return geom

    simplified = geom.simplify(simplify_tolerance, preserve_topology=True)

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
        ext_smooth = strong_smooth_geometry(simplified.exterior, smoothing_level)
        return Polygon(ext_smooth)
    else:
        return simplified

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

def pick_random_spot_on_line(line_geom):
    """Return a point at random distance along line_geom."""
    if line_geom.is_empty:
        return None
    if line_geom.geom_type == 'MultiLineString':
        sublines = list(line_geom.geoms)
        line_geom = random.choice(sublines)
    length = line_geom.length
    if length <= 0:
        return None
    d = random.uniform(0, length)
    return line_geom.interpolate(d)

def place_extra_line_on_main_line_as_is(extra_line, main_line, patch_polygon, max_tries=100):
    """
    1. Pick a random anchor point on main_line (ANY point along it).
    2. Shift the extra line so that its first endpoint matches that anchor.
    3. Check if the entire line is within patch_polygon.
    4. Return the placed geometry or None if fails.
    """
    if extra_line.is_empty or main_line.is_empty:
        return None

    # Use the first coordinate
    coords = []
    if extra_line.geom_type == 'MultiLineString':
        sub = list(extra_line.geoms)[0]
        coords = list(sub.coords)
    elif extra_line.geom_type == 'LineString':
        coords = list(extra_line.coords)
    else:
        return None

    if len(coords) < 1:
        return None

    start_pt = coords[0]

    for _ in range(max_tries):
        anchor = pick_random_spot_on_line(main_line)
        if anchor is None:
            continue

        dx = anchor.x - start_pt[0]
        dy = anchor.y - start_pt[1]
        placed = translate(extra_line, xoff=dx, yoff=dy)

        if patch_polygon.contains(placed):
            return placed
    return None

for raster_path in tif_paths:
    print(f"[INFO] Processing raster: {raster_path}")

    base_name = os.path.basename(raster_path)
    output_tif = os.path.join(out_folder, base_name.replace('.tif', '_synth_trails_v3.22.tif'))
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
        patch_polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

        gdf_sub = gdf_local[gdf_local.geometry.intersects(patch_polygon)]
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

    mode_assignment = {
        tid: ('burn' if random.random() < burn_probability else 'blend')
        for tid in trail_ids_sub
    }

    # ----------------------------------------------------------------
    # MAIN TRAILS (unchanged logic)
    # ----------------------------------------------------------------
    for tid in trail_ids_sub:
        trail_features = gdf_sub[gdf_sub['trail_id'] == tid]
        raw_geom = trail_features.union_all()

        smoothing_level = random.choice(['low', 'medium', 'no'])
        if smoothing_level != 'no':
            trail_geometry = strong_smooth_geometry(raw_geom, smoothing_level=smoothing_level)
        else:
            trail_geometry = raw_geom

        offset_distance_attr = 0
        if 'offset_distance' in trail_features.columns:
            offset_distance_attr = float(trail_features.iloc[0].get('offset_distance', 0) or 0)
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

            random_field_crop = np.random.uniform(low=burn_random_low, high=burn_random_high, size=shape_crop)
            smooth_sub_crop = ndimage.gaussian_filter(random_field_crop, sigma=5)
            distance_inside_crop = ndimage.distance_transform_edt(mask_crop)
            max_distance = distance_inside_crop.max() or 1
            gradient_mask_crop = distance_inside_crop / max_distance
            adjusted_sub_crop = smooth_sub_crop * gradient_mask_crop

            region = (slice(row_min, row_max), slice(col_min, col_max))
            data_modified_region = data_modified[region]
            # Subtract for burn
            data_modified_region[mask_crop == 1] -= adjusted_sub_crop[mask_crop == 1]
            data_modified[region] = data_modified_region

            del mask, mask_crop, random_field_crop, smooth_sub_crop
            del distance_inside_crop, gradient_mask_crop, adjusted_sub_crop, data_modified_region
            gc.collect()

        else:
            # Blend logic
            blend_pixels = random.uniform(blend_min_pixels, blend_max_pixels)
            buffer_distance = blend_pixels * pixel_size
            blend_buffer = trail_geometry.buffer(buffer_distance)

            label_records.append({
                "trail_id": tid,
                "mode": "blend",
                "offset_distance_attr": offset_distance_attr,
                "burn_min_in_use": "NA",
                "buffer_pixels": "NA",
                "buffer_distance": buffer_distance,
                "smoothing_level": smoothing_level,
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

            random_field_crop = generate_fractal_noise(shape_crop, octaves=3, persistence=0.3,
                                                       low=blend_random_low, high=blend_random_high)
            random_field_crop = ndimage.gaussian_filter(random_field_crop, sigma=10)
            random_field_crop = random_field_crop * 1.2 - 0.02

            distance_inside_crop = ndimage.distance_transform_edt(mask_crop)
            max_distance = distance_inside_crop.max() or 1
            gradient_mask_crop = distance_inside_crop / max_distance
            adjusted_field_crop = random_field_crop * gradient_mask_crop

            region = (slice(row_min, row_max), slice(col_min, col_max))
            data_modified_region = data_modified[region]
            # Add for blend
            data_modified_region[mask_crop == 1] += adjusted_field_crop[mask_crop == 1]
            data_modified[region] = data_modified_region

            del mask, mask_crop, random_field_crop
            del adjusted_field_crop, distance_inside_crop, gradient_mask_crop, data_modified_region
            gc.collect()

    # ==============
    # EXTRA TRAILS
    # ==============
    if not gdf_sub.empty and len(gdf_extra) > 0:
        # pick some random extras
        n_extras = min(10, 15)
        chosen_extras = gdf_extra.sample(n=n_extras)

        for idx, row_extra in chosen_extras.iterrows():
            main_row = gdf_sub.sample(n=1).iloc[0]
            main_geom = main_row.geometry
            if main_geom.is_empty:
                continue

            placed_geom = place_extra_line_on_main_line_as_is(
                row_extra.geometry,
                main_geom,
                patch_polygon,
                max_tries=50
            )
            if placed_geom is None:
                continue

            # Force BURN, no smoothing
            # => Use max=5 for extra trails
            width_pixels = random.uniform(burn_min_pixels, 6)
            buffer_distance = width_pixels * pixel_size

            new_buffer = placed_geom.buffer(buffer_distance)
            label_records.append({
                "trail_id": f"EXTRA_{idx}",
                "mode": "burn",
                "offset_distance_attr": 0,
                "burn_min_in_use": burn_min_pixels,
                "buffer_pixels": width_pixels,
                "buffer_distance": buffer_distance,
                "smoothing_level": "no",
                "geometry": new_buffer
            })

            mask2 = rasterize(
                [(new_buffer, 1)],
                out_shape=(profile['height'], profile['width']),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )

            rows2, cols2 = mask2.nonzero()
            if rows2.size == 0 or cols2.size == 0:
                continue

            rmin, rmax = rows2.min(), rows2.max() + 1
            cmin, cmax = cols2.min(), cols2.max() + 1

            mask_crop2 = mask2[rmin:rmax, cmin:cmax]
            shape_crop2 = mask_crop2.shape

            # Make sure the random range is strictly for burning
            # so values are positive and subtracted from data
            random_field_crop2 = np.random.uniform(low=burn_random_low, high=burn_random_high, size=shape_crop2)
            smooth_sub_crop2 = ndimage.gaussian_filter(random_field_crop2, sigma=5)
            distance_inside_crop2 = ndimage.distance_transform_edt(mask_crop2)
            max_dist2 = distance_inside_crop2.max() or 1
            gradient_mask_crop2 = distance_inside_crop2 / max_dist2
            adjusted_sub_crop2 = smooth_sub_crop2 * gradient_mask_crop2

            region2 = (slice(rmin, rmax), slice(cmin, cmax))
            data_modified_region2 = data_modified[region2]
            # Subtract => ensures a "burn" effect
            data_modified_region2[mask_crop2 == 1] -= adjusted_sub_crop2[mask_crop2 == 1]
            data_modified[region2] = data_modified_region2

            del (mask2, mask_crop2, random_field_crop2, smooth_sub_crop2,
                 distance_inside_crop2, gradient_mask_crop2, adjusted_sub_crop2, data_modified_region2)
            gc.collect()

    data_modified.flush()

    # ======================================
    # SHRINK LABEL POLYGONS BY 1 PIXEL
    # ======================================
    gdf_labels = gpd.GeoDataFrame(label_records, geometry="geometry", crs=gdf_sub.crs)

    if not gdf_labels.empty:
        # 1 pixel = pixel_size (meters, for instance)
        shrink_dist = pixel_size
        new_geoms = []
        for geom in gdf_labels.geometry:
            if geom is None or geom.is_empty:
                new_geoms.append(None)
                continue
            shrunk = geom.buffer(-shrink_dist)
            if not shrunk.is_empty:
                # buffer(0) to fix potential self-intersections
                shrunk = shrunk.buffer(0)
                if shrunk.is_empty:
                    new_geoms.append(None)
                else:
                    new_geoms.append(shrunk)
            else:
                new_geoms.append(None)

        gdf_labels["geometry"] = new_geoms
        gdf_labels = gdf_labels[gdf_labels.geometry.notnull() & ~gdf_labels.geometry.is_empty]

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(np.array(data_modified), 1)

    if not gdf_labels.empty:
        gdf_labels.to_file(output_gpkg, driver="GPKG")
        print(f"[INFO] Polygons saved: {output_gpkg}")
    else:
        print("All label polygons vanished after shrinking or none existed. No label file saved.")

    print(f"[INFO] Raster saved: {output_tif}")

    del data_modified
    gc.collect()

print("[INFO] Processing complete for all matched TIF files.")
