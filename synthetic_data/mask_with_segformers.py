import os
import glob
import gc
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from concurrent.futures import ProcessPoolExecutor, as_completed



################ CODE FOR RECOVERING TRAILS USING THEIR LOCATION FROM SEGFORMERS!

# ------------------------
# User parameters
# ------------------------
input_folder = "/media/irina/My Book/Surmont/nDTM"
segformer_folder = "/media/irina/My Book/Surmont/nDTM_10cm_segformer"

# Buffer distance (meters) for label geometry
buffer_distance_m = 10.0

# Minimum confidence in segformer raster
confidence_threshold = 3

# Where to put final output
output_folder = os.path.join(input_folder, "blended_with_segformer")
os.makedirs(output_folder, exist_ok=True)

# Collect all TIFs in input_folder that match *_blended_synth_trails.tif
tif_paths = glob.glob(os.path.join(input_folder, "*.tif"))
print(tif_paths)
# Example override for testing a single file:
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
    if "nDTM" not in input_base:
        return None
    segformer_base = input_base.replace("_nDTM", "_nDTM_preds_segformer")
    return os.path.join(segformer_folder, segformer_base + ".tif")

def process_tif(input_raster_path):
    """Process a single TIF file: skip areas in burn buffer, add +0.05 where segformer >= 3."""
    base_name = get_base(input_raster_path)
    print(f"[INFO] Processing: {input_raster_path}")

    # 1) Open input raster
    with rasterio.open(input_raster_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        data = src.read(1).astype(np.float32)

    height, width = data.shape
    maskA = np.zeros((height, width), dtype=np.uint8)

    # 3) Segformer raster => *preds_segformer.tif
    segformer_path = construct_segformer_path(base_name)
    print('segformer_path: ', segformer_path)
    if segformer_path is None or not os.path.isfile(segformer_path):
        msg = f"  [WARNING] No segformer raster found for {base_name}.\n    Expected: {segformer_path}\n    Skipping..."
        print(msg)
        return msg

    with rasterio.open(segformer_path) as conf_src:
        if conf_src.crs != crs:
            print("  [WARNING] segformer raster CRS != input raster CRS (attempting read anyway).")
        conf_data = conf_src.read(1).astype(np.float32)

    # Create maskB where >= confidence_threshold (3)
    maskB = (conf_data >= confidence_threshold).astype(np.uint8)

    # region_of_interest = (outside burn buffer) & (segformer >= 3)
    region_of_interest = ((maskA == 0) & (maskB == 1)).astype(np.uint8)
    roi_sum = region_of_interest.sum()
    if roi_sum == 0:
        msg = "  [INFO] No pixels to process in this tile. Skipping modifications."
        print(msg)
        return msg

    # Just +0.05 in the region_of_interest
    data[region_of_interest == 1] += 0.05

    # Save final
    out_name = base_name + "_postblend.tif"
    output_path = os.path.join(output_folder, out_name)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)

    print(f"  [INFO] Processed {base_name}. Pixels updated: {roi_sum}. Output => {output_path}")
    del data, conf_data, maskA, maskB, region_of_interest
    gc.collect()
    return f"Finished {base_name}"

def main():
    print("[INFO] Starting parallel processing.")
    results = []
    # Use up to 16 workers
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=16) as executor:
        future_to_file = {executor.submit(process_tif, path): path for path in tif_paths}

        for future in as_completed(future_to_file):
            path = future_to_file[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[ERROR] {path} generated an exception: {exc}")
            else:
                results.append(result)
                print(f"[INFO] Completed: {result}")

    print("[INFO] All tasks finished.")
    return results

if __name__ == "__main__":
    main()
