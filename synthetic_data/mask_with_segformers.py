import os
import glob
import gc
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

# ------------------------
# User parameters
# ------------------------
input_folder = "/media/irina/My Book/Surmont/nDTM_synth_trails"
segformer_folder = "/media/irina/My Book/Surmont/nDTM_10cm_segformer"

# Buffer distance (meters) for label geometry
buffer_distance_m = 5.0

# Minimum confidence in segformer raster (changed to 3)
confidence_threshold = 3

# Where to put final output
output_folder = os.path.join(input_folder, "blended_outputs")
os.makedirs(output_folder, exist_ok=True)

# Collect all TIFs in input_folder that match *_blended_synth_trails.tif
tif_paths = glob.glob(os.path.join(input_folder, "*_blended_synth_trails.tif"))
# tif_paths = ['/media/irina/My Book/Surmont/nDTM_synth_trails/510_6228_nDTM_blended_synth_trails.tif']

def get_base(filepath):
    """Return the file basename without extension."""
    return os.path.splitext(os.path.basename(filepath))[0]

def construct_label_path(raster_path):
    """
    The label file has the same name but .gpkg extension.
    E.g. 493_6223_nDTM_blended_synth_trails.tif -> 493_6223_nDTM_blended_synth_trails.gpkg
    """
    raster_dir = os.path.join(os.path.dirname(raster_path), 'labels')
    output_basename = os.path.basename(raster_path.replace(".tif", ".gpkg"))
    return os.path.join(raster_dir, output_basename)

def construct_segformer_path(input_base):
    """
    The segformer raster name is like:
    e.g. 493_6223_nDTM_blended_synth_trails -> 493_6223_nDTM_preds_segformer.tif
    So we replace '_blended_synth_trails' with '_preds_segformer'
    """
    if "_blended_synth_trails" not in input_base:
        return None
    segformer_base = input_base.replace("_blended_synth_trails", "_preds_segformer")
    return os.path.join(segformer_folder, segformer_base + ".tif")

for input_raster_path in tif_paths:
    base_name = get_base(input_raster_path)  # e.g. 493_6223_nDTM_blended_synth_trails
    print(f"[INFO] Processing: {input_raster_path}")

    # ---------------------------------------
    # 1) Open input raster
    # ---------------------------------------
    with rasterio.open(input_raster_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        data = src.read(1).astype(np.float32)

    height, width = data.shape

    # ---------------------------------------
    # 2) Find and load the label
    # ---------------------------------------
    label_path = construct_label_path(input_raster_path)
    if not os.path.isfile(label_path):
        print(f"  [WARNING] No label .gpkg found for {base_name}. Skipping...")
        continue

    gdf = gpd.read_file(label_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Filter label to only features with "mode"="burn"
    gdf_burn = gdf[gdf["mode"] == "burn"]
    if not gdf_burn.empty:
        label_union_burn = gdf_burn.unary_union
        burn_buffer = label_union_burn.buffer(buffer_distance_m)
    else:
        burn_buffer = None

    # Rasterize maskA: 1 = inside buffer of "burn" features
    if burn_buffer is not None and not burn_buffer.is_empty:
        maskA = rasterize(
            [(burn_buffer, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
    else:
        # If no burn geometry, maskA is all zeros
        maskA = np.zeros((height, width), dtype=np.uint8)

    # ---------------------------------------
    # 3) Segformer raster => *preds_segformer.tif
    # ---------------------------------------
    segformer_path = construct_segformer_path(base_name)
    if segformer_path is None or not os.path.isfile(segformer_path):
        print(f"  [WARNING] No segformer raster found for {base_name}.\n    Expected: {segformer_path}\n    Skipping...")
        continue

    with rasterio.open(segformer_path) as conf_src:
        if conf_src.crs != crs:
            print("  [WARNING] segformer raster CRS != input raster CRS (attempting read anyway).")
        conf_data = conf_src.read(1).astype(np.float32)

    # Create maskB where >= 3
    maskB = (conf_data >= confidence_threshold).astype(np.uint8)

    # region_of_interest = (outside the burn buffer) & (segformer >= 3)
    region_of_interest = ((maskA == 0) & (maskB == 1)).astype(np.uint8)

    if region_of_interest.sum() == 0:
        print("  [INFO] No pixels to process in this tile. Skipping modifications.")
        continue

    # ---------------------------------------
    # Just +0.1 in the region_of_interest
    # ---------------------------------------
    data[region_of_interest == 1] += 0.05

    # Save final
    out_name = base_name + "_postblend.tif"
    output_path = os.path.join(output_folder, out_name)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)

    print(f"  [INFO] Saved updated raster to {output_path}")

    # Cleanup
    del data, conf_data, maskA, maskB, region_of_interest
    gc.collect()

print("[INFO] Done processing all TIF files.")
