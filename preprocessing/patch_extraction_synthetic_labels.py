#!/usr/bin/env python3
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
import scipy.ndimage as ndimage

def rotate_and_crop(image, angle, crop_size, order=1):
    """
    Rotates the given image by 'angle' (in degrees) and then crops the central
    square of size (crop_size x crop_size). The parameter 'order' sets the
    interpolation order (order=1 for continuous images, order=0 for categorical labels).
    """
    rotated = ndimage.rotate(image, angle, reshape=True, order=order)
    new_h, new_w = rotated.shape
    start_row = (new_h - crop_size) // 2
    start_col = (new_w - crop_size) // 2
    # If the rotated image is too small, return the original.
    if start_row < 0 or start_col < 0:
        return image
    cropped = rotated[start_row:start_row + crop_size, start_col:start_col + crop_size]
    return cropped

def process_tif(args):
    tif_file, tif_index, label_dir, output_dir, patch_size, num_random_patches = args
    base_tif = os.path.basename(tif_file)
    # Derive corresponding label file name:
    # e.g., "521_6228_nDTM_blended_synth_trails_postblend.tif" -> "521_6228_nDTM_blended_synth_trails.gpkg"
    if base_tif.endswith(".tif"):
        base_label = base_tif.replace(".tif", ".gpkg")
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

                # With 50% probability, perform a random rotation.
                if random.random() < 0.5:
                    # Choose a random angle in the range 20-70 degrees.
                    base_angle = random.uniform(20, 70)
                    # Add a random multiple of 90 to distribute rotations among quadrants.
                    quadrant_offset = random.choice([0, 90, 180, 270])
                    angle = base_angle + quadrant_offset
                    patch_img = rotate_and_crop(patch_img, angle, patch_size, order=1)
                    patch_label = rotate_and_crop(patch_label, angle, patch_size, order=0)

                # Compute a global patch number to ensure sequential naming across images.
                global_patch_number = tif_index * num_random_patches + i + 1
                img_out_path = os.path.join(output_dir, f"{global_patch_number}_image.tif")
                label_out_path = os.path.join(output_dir, f"{global_patch_number}_label.tif")

                # Update profile for image patch
                profile_patch = src.profile.copy()
                profile_patch.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": patch_transform,
                    "count": 1
                })

                # Write image patch
                with rasterio.open(img_out_path, "w", **profile_patch) as dst:
                    dst.write(patch_img, 1)

                # Update profile for label patch (using uint8 and nodata 0)
                label_profile = profile_patch.copy()
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
    tif_dir = "/media/irina/My Book/Surmont/nDTM_synth_trails_v.3.22"
    label_dir = "/media/irina/My Book/Surmont/nDTM_synth_trails_v.3.22/labels"
    patch_size = 1024  # patch size in pixels
    num_random_patches = 20  # number of patches per image

    output_dir = f"/media/irina/My Book/Surmont/TrainingCNN/synth_tracks_{patch_size}px_10cm_v3.22"

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
