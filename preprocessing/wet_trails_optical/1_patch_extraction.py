import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import pandas as pd

def extract_patch(raster_path, geom, patch_size=512):
    """
    Extracts a patch of size patch_size x patch_size pixels from the raster centered at the given geometry.
    If the geometry is a MultiPoint, the first point is used.
    Returns (patch_array, profile) if successful, or None if the patch would go out of bounds.
    """
    # Handle MultiPoint: choose the first point
    if geom.geom_type == "MultiPoint":
        point = list(geom.geoms)[0]
    elif geom.geom_type == "Point":
        point = geom
    else:
        # For other types, use the centroid
        point = geom.centroid

    with rasterio.open(raster_path) as src:
        try:
            row, col = src.index(point.x, point.y)
        except Exception as e:
            print(f"Error converting point {point} in {raster_path}: {e}")
            return None
        half = patch_size // 2
        # Ensure the full window is within raster bounds.
        if (row - half < 0 or row + half > src.height or
            col - half < 0 or col + half > src.width):
            return None
        window = Window(col - half, row - half, patch_size, patch_size)
        patch = src.read(window=window)
        patch_transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update({
            "height": patch_size,
            "width": patch_size,
            "transform": patch_transform,
            "count": src.count
        })
        return patch, profile

def save_patch(patch, profile, out_path):
    """
    Saves the patch (a numpy array) to out_path using the provided rasterio profile.
    """
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(patch)
    print(f"Saved patch to {out_path}")

if __name__ == '__main__':
    # Define file paths.
    # Note: This points file is expected to have the "raster1" column (from the previous processing)
    points_path = "/media/irina/My Book/Surmont/intermediate/points_with_rasters.gpkg"
    rasters_folder = "/media/irina/My Book1/Conoco/DATA/Orthos/3_Orthomosaics_10cm"  # Folder with full-resolution orthomosaics
    label_rasters_folder = "/media/irina/My Book/Surmont/intermediate/segformer_wettrails"  # Folder with label rasters
    output_folder = "/media/irina/My Book/Surmont/TrainingCNN/wettrail_RGB10cm_512px"
    os.makedirs(output_folder, exist_ok=True)
    patch_size = 512  # 512 pixels (~51.2 m at 10 cm resolution)

    # Load the enriched points layer.
    gdf_points = gpd.read_file(points_path)
    print(gdf_points.columns)

    counter = 1  # For naming output patches
    for idx, row in gdf_points.iterrows():
        point = row.geometry

        # Get the raster name from the "raster1" column.
        raster_base = row.get("raster1")
        print(raster_base)

        if not raster_base:
            # print(f"Point {point} has no 'raster1' value, skipping.")
            continue

        # Derive the image raster filename by removing the "labels_" prefix (if present)
        if raster_base.startswith("labels_"):
            image_filename = raster_base[len("labels_"):]
        else:
            image_filename = raster_base

        # Construct full paths.
        image_raster_path = os.path.join(rasters_folder, image_filename)
        if not os.path.exists(image_raster_path):
            print(f"Image raster {image_raster_path} does not exist for point {point}, skipping.")
            continue

        label_raster_path = os.path.join(label_rasters_folder, raster_base)
        if not os.path.exists(label_raster_path):
            print(f"Label raster {label_raster_path} does not exist for point {point}, skipping.")
            continue

        # Extract image patch.
        result_img = extract_patch(image_raster_path, point, patch_size=patch_size)
        if result_img is None:
            print(f"Image patch extraction failed for point {point} using {image_raster_path}, skipping.")
            continue
        patch_img, profile_img = result_img

        # Extract label patch.
        result_lbl = extract_patch(label_raster_path, point, patch_size=patch_size)
        if result_lbl is None:
            print(f"Label patch extraction failed for point {point} using {label_raster_path}, skipping.")
            continue
        patch_lbl, profile_lbl = result_lbl

        # Ensure label profile is set to single band and proper data type.
        profile_lbl.update({"count": 1, "dtype": rasterio.uint8})

        # Define output filenames.
        out_image_path = os.path.join(output_folder, f"{counter}_image.tif")
        out_label_path = os.path.join(output_folder, f"{counter}_label.tif")

        # Save the patches.
        save_patch(patch_img, profile_img, out_image_path)
        save_patch(patch_lbl, profile_lbl, out_label_path)

        counter += 1

    print("Patch extraction completed.")
