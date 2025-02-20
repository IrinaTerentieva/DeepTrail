import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import os
from shapely.ops import unary_union
from shapely.affinity import translate
import random
import scipy.ndimage as ndimage
import gc
import tempfile

# --- Options & Parameters ---
buffer_meters = 4        # Buffer distance in meters for the centerline.
shift_meters = 10          # How far to shift the buffer (in meters) diagonally NW.
gaussian_sigma_blur = 2   # Sigma for any additional blur if desired.
blend_weight = 1        # Blending weight factor (0 means no effect, 1 means full effect)

# --- Paths ---
vector_path = '/media/irina/My Book/Surmont/vector_data/FLM/FLM_centerline_Surmont.gpkg'
raster_path = '/media/irina/My Book/Surmont/temp/test.tif'
output_tif = raster_path.replace('.tif', '_blended.tif')

# --- Load Vector & Clip to Raster Boundaries ---
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
inverted = False
if inverted:
    max_val = data.max()
    data = max_val - data
    print("[INFO] Raster has been inverted.")

# --- Create Memory-mapped Raster ---
temp_filename = os.path.join(tempfile.gettempdir(), "data_modified.dat")
data_modified = np.memmap(temp_filename, dtype=np.float32, mode='w+', shape=data.shape)
data_modified[:] = data[:]
del data
gc.collect()

# --- BLENDING STEP: Shift Buffer Sampling & Blend ---
print("[INFO] Starting blending step: sampling adjacent raster and blending.")
label_records = []  # To store buffer polygons.
trail_ids = gdf['trail_id'].unique()
for tid in trail_ids:
    # Get the trail geometry.
    trail_features = gdf[gdf['trail_id'] == tid]
    if trail_features.empty:
        continue
    trail_geometry = trail_features.unary_union

    # Create the original buffer (footprint) around the centerline.
    trail_buffer = trail_geometry.buffer(buffer_meters)
    label_records.append({
        "trail_id": tid,
        "buffer_distance": buffer_meters,
        "geometry": trail_buffer
    })

    # Create a shifted version of the buffer: shift diagonally NW.
    # NW: x offset negative, y offset positive.
    shifted_buffer = translate(trail_buffer, xoff=-shift_meters, yoff=shift_meters)

    # Rasterize both the original buffer and the shifted buffer.
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

    # Determine the bounding box from the original mask.
    rows, cols = np.where(mask_orig == 1)
    if len(rows) == 0 or len(cols) == 0:
        continue
    row_min, row_max = rows.min(), rows.max() + 1
    col_min, col_max = cols.min(), cols.max() + 1
    region = (slice(row_min, row_max), slice(col_min, col_max))

    # Crop the original mask and the shifted mask.
    mask_crop_orig = mask_orig[row_min:row_max, col_min:col_max]
    mask_crop_shift = mask_shift[row_min:row_max, col_min:col_max]
    # Extract the corresponding raster region.
    region_data = data_modified[region].copy()

    # Now, sample the adjacent raster using the shifted mask.
    # The idea is: where mask_crop_shift==1, we sample the raster.
    sampled_region = np.copy(region_data)
    sampled_region[mask_crop_shift == 1] = region_data[mask_crop_shift == 1]

    # Next, we need to move this sampled region back to the original location.
    # Since we shifted the buffer by (-shift_meters, shift_meters) in map units,
    # we "move back" by shifting the sampled_region by the inverse in pixel units.
    # Calculate shift in pixels:
    shift_pixels_x = int(round(shift_meters / pixel_size))
    shift_pixels_y = int(round(shift_meters / pixel_size))
    # To move back, shift by (shift_pixels_x, -shift_pixels_y).
    # We can use np.roll to shift arrays.
    moved_region = np.roll(sampled_region, shift=(shift_pixels_y, shift_pixels_x), axis=(0, 1))

    # Compute a gradient mask from the original buffer's distance transform (0 at edge, 1 at center).
    distance_inside = ndimage.distance_transform_edt(mask_crop_orig)
    max_distance = distance_inside.max() if distance_inside.max() > 0 else 1
    gradient_mask = distance_inside / max_distance

    # Blend the moved (sampled) region with the original region:
    # Final value = original * (1 - weight) + moved * weight,
    # where weight is given by the gradient mask.
    blended_region = region_data * (1 - gradient_mask) + moved_region * gradient_mask

    # Optionally, apply a light Gaussian blur to smooth the transition further.
    blurred_region = ndimage.gaussian_filter(blended_region, sigma=gaussian_sigma_blur)
    final_region = region_data * (1 - 0.5 * gradient_mask) + blurred_region * (0.5 * gradient_mask)

    # Update only pixels within the original buffer.
    region_data[mask_crop_orig == 1] = final_region[mask_crop_orig == 1]
    data_modified[region] = region_data

    # Clean up for this trail.
    del mask_orig, mask_shift, mask_crop_orig, mask_crop_shift, region_data, sampled_region, moved_region, distance_inside, gradient_mask, blended_region, blurred_region, final_region
    gc.collect()
print("[INFO] Blending step completed using shifted buffer sampling.")

# --- Post-Processing: Set Out-of-Range Values to 0 ---
print("[INFO] Post-processing: setting values outside [-5,5] to 0.")
final_array = np.array(data_modified)
final_array[(final_array < -5) | (final_array > 5)] = 0
data_modified.flush()

# --- Save Final Raster ---
with rasterio.open(output_tif, 'w', **profile) as dst:
    dst.write(final_array, 1)
print("Final blended raster saved to:", output_tif)

# --- Save Label Polygons ---
output_gpkg = os.path.splitext(output_tif)[0] + ".gpkg"
gdf_labels = gpd.GeoDataFrame(label_records, geometry="geometry", crs=gdf.crs)
gdf_labels.to_file(output_gpkg, driver="GPKG")
print("Label polygons saved to:", output_gpkg)
