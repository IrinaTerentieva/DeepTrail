import os
import glob
import random
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from tqdm import tqdm
import multiprocessing

def process_tif(args):
    tif_file, tif_index, label_dir, output_dir, patch_size, num_random_patches = args
    base_tif = os.path.basename(tif_file)
    # Derive corresponding label file name:
    # e.g., "521_6228_nDTM_blended_synth_trails_postblend.tif" -> "521_6228_nDTM_blended_synth_trails.gpkg"
    if base_tif.endswith("_postblend.tif"):
        base_label = base_tif.replace("_postblend.tif", ".gpkg")
    else:
        base_label = os.path.splitext(base_tif)[0] + ".gpkg"
    label_path = os.path.join(label_dir, base_label)
    if not os.path.exists(label_path):
        print(f"Label file {label_path} not found for image {base_tif}, skipping.")
        return

    print(f"Processing image: {base_tif}")
    # Read and filter label features for "burn" mode
    try:
        gdf = gpd.read_file(label_path)
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        return

    burn_gdf = gdf[gdf["mode"] == "burn"]
    if burn_gdf.empty:
        print(f"No features with mode 'burn' in {base_label}, skipping.")
        return

    try:
        with rasterio.open(tif_file) as src:
            width = src.width
            height = src.height

            # Define valid range for random center pixels so that the patch is fully inside.
            min_col = patch_size // 2
            max_col = width - patch_size // 2 - 1
            min_row = patch_size // 2
            max_row = height - patch_size // 2 - 1

            if max_col <= min_col or max_row <= min_row:
                print(f"Image {base_tif} is too small for patch extraction, skipping.")
                return

            for i in range(num_random_patches):
                col = random.randint(min_col, max_col)
                row_pix = random.randint(min_row, max_row)
                window = Window(
                    col_off=col - patch_size // 2,
                    row_off=row_pix - patch_size // 2,
                    width=patch_size,
                    height=patch_size
                )
                try:
                    patch_img = src.read(1, window=window)
                except Exception as e:
                    print(f"Error reading image patch at ({col}, {row_pix}) in {base_tif}: {e}")
                    continue

                patch_transform = src.window_transform(window)
                # Compute pixel size (assumes square pixels)
                pixel_size = abs(patch_transform.a)

                # Narrow (buffer inward) each burn feature by 1 pixel before rasterizing.
                buffered_geoms = []
                for geom in burn_gdf.geometry:
                    if geom is not None:
                        narrowed = geom.buffer(-pixel_size)
                        if not narrowed.is_empty:
                            buffered_geoms.append(narrowed)

                # Rasterize the narrowed geometries: pixels inside become 1, outside remain 0.
                patch_label = rasterize(
                    [(geom, 1) for geom in buffered_geoms],
                    out_shape=(patch_size, patch_size),
                    transform=patch_transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8
                )

                # Compute a global patch number to ensure sequential naming across images.
                global_patch_number = tif_index * num_random_patches + i + 1
                img_out_path = os.path.join(output_dir, f"{global_patch_number}_image.tif")
                label_out_path = os.path.join(output_dir, f"{global_patch_number}_label.tif")

                # Update profile for image patch
                profile = src.profile.copy()
                profile.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": patch_transform,
                    "count": 1
                })

                # Write image patch
                with rasterio.open(img_out_path, "w", **profile) as dst:
                    dst.write(patch_img, 1)

                # Update profile for label patch (using uint8 and nodata 0)
                label_profile = profile.copy()
                label_profile.update({
                    "dtype": rasterio.uint8,
                    "nodata": 0
                })
                with rasterio.open(label_out_path, "w", **label_profile) as dst:
                    dst.write(patch_label, 1)

                print(f"Extracted patch {global_patch_number} from {base_tif} at random center ({col}, {row_pix})")
    except Exception as e:
        print(f"Error processing {base_tif}: {e}")
        return

def extract_patches_parallel():
    # Directories and parameters
    tif_dir = "/media/irina/My Book/Surmont/nDTM_synth_trails/blended_outputs"
    label_dir = "/media/irina/My Book/Surmont/nDTM_synth_trails/labels"
    patch_size = 512  # patch size in pixels
    num_random_patches = 40  # number of patches per image

    output_dir = f"/home/irina/HumanFootprint/DATA/Training_CNN/synth_tracks_{patch_size}px_10cm"

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted list of TIFF files
    tif_files = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))
    if not tif_files:
        print(f"No TIFF files found in {tif_dir}")
        return

    # Build arguments for each TIFF file
    args_list = []
    for idx, tif_file in enumerate(tif_files):
        args_list.append((tif_file, idx, label_dir, output_dir, patch_size, num_random_patches))

    # Use 16 cores – each process will handle one TIFF file at a time.
    with multiprocessing.Pool(processes=16) as pool:
        list(tqdm(pool.imap_unordered(process_tif, args_list), total=len(args_list)))
    print("Patch extraction complete.")

if __name__ == "__main__":
    extract_patches_parallel()
