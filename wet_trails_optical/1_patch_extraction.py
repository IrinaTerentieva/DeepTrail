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
    # Positive points from wet trails and negative points from segmentation output.

    neg_points_path = "/media/irina/My Book/Surmont/intermediate/segformer_wettrails/negative_points.gpkg"
    pos_points_path = "/media/irina/My Book/Surmont/manual/points_with_rasters_upd.gpkg"

    # Folder with full-resolution orthomosaics.
    rasters_folder = "/media/irina/My Book1/Conoco/DATA/Orthos/3_Orthomosaics_10cm"
    # Folder with label rasters.
    label_rasters_folder = "/media/irina/My Book/Surmont/intermediate/segformer_wettrails"
    # Output folder for patch pairs.
    output_folder = "/media/irina/My Book/Surmont/TrainingCNN/wettrail_RGB10cm_512px"
    os.makedirs(output_folder, exist_ok=True)
    patch_size = 512  # 512 pixels (~51.2 m at 10 cm resolution)

    # Threshold for fraction of zeros in the image patch (e.g., 0.5 means >50% zeros is too many)
    zero_threshold = 0.5

    # O{therwise combine positive and negative points.
    gdf_pos = gpd.read_file(pos_points_path)
    print('Number of object points: ', len(gdf_pos))
    if os.path.exists(neg_points_path):
        gdf_neg = gpd.read_file(neg_points_path)
        print('Number of NON object points: ', len(gdf_neg))

        gdf_points = pd.concat([gdf_pos, gdf_neg], ignore_index=True)
    else:
        gdf_points = gdf_pos

    print('Number of ALL object points: ', len(gdf_points))
    print("Columns in the points file:", gdf_points.columns.tolist())

    counter = 1  # For naming output patches

    for idx, row in gdf_points.iterrows():

        # print(row)
        geom = row.geometry

        # Get the raster name from the "raster1" column.
        # (This column should have been created in the previous processing step.)
        raster_base = row.get("raster1")
        if not raster_base:
            print(f"Point {geom} has no 'raster1' value, skipping.")
            continue

        # Derive the image raster filename by removing the "labels_" prefix (if present)
        if raster_base.startswith("labels_"):
            image_filename = raster_base[len("labels_"):]
        else:
            image_filename = raster_base

        # Construct full paths.
        image_raster_path = os.path.join(rasters_folder, image_filename)
        if not os.path.exists(image_raster_path):
            print(f"Image raster {image_raster_path} does not exist for point {geom}, skipping.")
            continue

        label_raster_path = os.path.join(label_rasters_folder, raster_base)
        if not os.path.exists(label_raster_path):
            print(f"Label raster {label_raster_path} does not exist for point {geom}, skipping.")
            continue

        # Extract image patch.
        result_img = extract_patch(image_raster_path, geom, patch_size=patch_size)
        if result_img is None:
            print(f"Image patch extraction failed for point {geom} using {image_raster_path}, skipping.")
            continue
        patch_img, profile_img = result_img

        # Check if the image patch has too many zeros (likely a boundary)
        zero_fraction = np.sum(patch_img == 0) / patch_img.size
        if zero_fraction > zero_threshold:
            print(f"Image patch for point {geom} has too many zeros ({zero_fraction:.2f}), skipping pair.")
            continue

        # Extract corresponding label patch.
        result_lbl = extract_patch(label_raster_path, geom, patch_size=patch_size)
        if result_lbl is None:
            print(f"Label patch extraction failed for point {geom} using {label_raster_path}, skipping.")
            continue
        patch_lbl, profile_lbl = result_lbl

        # Ensure label profile is set for single band and proper data type.
        profile_lbl.update({"count": 1, "dtype": rasterio.uint8})

        # Define output filenames.
        out_image_path = os.path.join(output_folder, f"{counter}_image.tif")
        out_label_path = os.path.join(output_folder, f"{counter}_label.tif")

        # Save the patches.
        save_patch(patch_img, profile_img, out_image_path)
        save_patch(patch_lbl, profile_lbl, out_label_path)

        counter += 1

    print("Patch extraction completed.")
