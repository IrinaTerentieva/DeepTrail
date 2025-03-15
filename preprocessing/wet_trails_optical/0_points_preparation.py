import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point, box
from shapely.ops import unary_union
import pandas as pd


# ---------------------------
# 1. Data Preparation Functions
# ---------------------------

def combine_labels(labels_folder):
    """
    Reads all .gpkg files in the given folder, concatenates them into a single GeoDataFrame.
    """
    gdf_list = []
    for file in os.listdir(labels_folder):
        if file.endswith(".gpkg"):
            path = os.path.join(labels_folder, file)
            gdf = gpd.read_file(path)
            gdf_list.append(gdf)
    if not gdf_list:
        raise ValueError("No GeoPackage files found in the labels folder.")
    combined = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
    return combined


def rasterize_labels(gdf_labels, template_raster_path, out_raster_path):
    """
    Rasterizes vector labels to the grid and resolution of a template raster.
    All label pixels are burned as value 1.
    The output raster will have only 0 (background) and 1 (object) with no nodata value.
    """
    # Filter out invalid or empty geometries.
    valid_labels = gdf_labels[gdf_labels.geometry.notnull() & ~gdf_labels.geometry.is_empty]

    with rasterio.open(template_raster_path) as src:
        transform = src.transform
        out_shape = (src.height, src.width)

        # Create (geometry, value) tuples for all valid features.
        shapes = [(geom, 1) for geom in valid_labels.geometry if geom is not None]

        rasterized = rasterize(
            shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=rasterio.uint8
        )

        # Copy and update metadata.
        meta = src.meta.copy()
        meta.pop('nodata', None)  # Remove any nodata key
        meta.update({'count': 1, 'dtype': rasterio.uint8})

        os.makedirs(os.path.dirname(out_raster_path), exist_ok=True)
        with rasterio.open(out_raster_path, 'w', **meta) as dst:
            dst.write(rasterized, 1)

    print(f"Rasterized labels saved to {out_raster_path}")


def generate_negative_points(disposition_gpkg, raster_path, num_points=100, min_distance=20, max_distance=100):
    """
    Generates random points within the ring (buffer of max_distance minus buffer of min_distance)
    of the footprint (from disposition layer) and restricts them to lie within the raster boundaries.
    """
    # Load disposition footprints.
    gdf_disp = gpd.read_file(disposition_gpkg)
    footprint = unary_union(gdf_disp.geometry)

    # Create buffer rings.
    buffer_inner = footprint.buffer(min_distance)
    buffer_outer = footprint.buffer(max_distance)
    eligible_area = buffer_outer.difference(buffer_inner)

    # Get raster boundary.
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    raster_polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Eligible area is the intersection with the raster extent.
    eligible_area = eligible_area.intersection(raster_polygon)
    if eligible_area.is_empty:
        raise ValueError("No eligible area found after intersecting with raster boundaries.")

    # Generate random points.
    points = []
    minx, miny, maxx, maxy = eligible_area.bounds
    attempts = 0
    max_attempts = num_points * 100
    while len(points) < num_points and attempts < max_attempts:
        rand_x = np.random.uniform(minx, maxx)
        rand_y = np.random.uniform(miny, maxy)
        pt = Point(rand_x, rand_y)
        if eligible_area.contains(pt):
            points.append(pt)
        attempts += 1
    if len(points) < num_points:
        print(f"Warning: Only generated {len(points)} points after {attempts} attempts.")
    gdf_points = gpd.GeoDataFrame(geometry=points, crs=gdf_disp.crs)
    return gdf_points


# ---------------------------
# 2. SegFormer Dataset Outline (PyTorch)
# ---------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class WetTrailsDataset(Dataset):
    def __init__(self, image_files, label_files, transform=None):
        """
        image_files: list of file paths to the orthomosaic images.
        label_files: list of file paths to the corresponding label rasters.
        transform: optional function to apply transforms on (image, label).
        """
        self.image_files = image_files
        self.label_files = label_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Open image and label using PIL.
        image = Image.open(self.image_files[idx]).convert("RGB")
        label = Image.open(self.label_files[idx])
        if self.transform:
            image, label = self.transform(image, label)
        # Convert label to tensor (assumes labels are stored as pixel values).
        label_tensor = torch.tensor(np.array(label), dtype=torch.long)
        return image, label_tensor


# ---------------------------
# 3. Main Execution: Data Preparation Pipeline
# ---------------------------
if __name__ == '__main__':
    # Define your file paths.
    wet_trails_points_path = "/media/irina/My Book/Surmont/manual/wet_trails_Surmont_2024.gpkg"
    rasters_folder = "/media/irina/My Book1/Conoco/DATA/Orthos/3_Orthomosaics_10cm"
    labels_folder = "/media/irina/My Book/Surmont/Products/wet_trails"
    disposition_path = "/media/irina/My Book1/Conoco/DATA/Manual/surmont_footprint_by_mosaic.gpkg"

    # (Optional) Load wet trails points.
    gdf_wet_trails = gpd.read_file(wet_trails_points_path)
    print(f"Loaded {len(gdf_wet_trails)} wet trail points.")

    # Combine label GeoPackages.
    gdf_labels = combine_labels(labels_folder)
    print(f"Combined labels: {len(gdf_labels)} features.")

    # Rasterize labels for each orthomosaic.
    output_folder = "/media/irina/My Book/Surmont/intermediate/segformer_wettrails"
    os.makedirs(output_folder, exist_ok=True)
    raster_files = [f for f in os.listdir(rasters_folder) if f.lower().endswith(".tif")]
    rasterized_label_files = []
    for file in raster_files:
        raster_path = os.path.join(rasters_folder, file)
        out_raster_path = os.path.join(output_folder, f"labels_{file}")
        rasterize_labels(gdf_labels, raster_path, out_raster_path)
        rasterized_label_files.append(out_raster_path)

    # Generate negative sample points for ALL rasters and combine into one GeoPackage.
    negative_points_list = []
    for file in raster_files:
        raster_path = os.path.join(rasters_folder, file)
        gdf_negative = generate_negative_points(disposition_path, raster_path, num_points=100, min_distance=20,
                                                max_distance=100)
        # Optionally, add a field to track the source raster filename
        gdf_negative["source_raster"] = file
        negative_points_list.append(gdf_negative)

    if negative_points_list:
        gdf_negative_all = gpd.GeoDataFrame(pd.concat(negative_points_list, ignore_index=True),
                                            crs=negative_points_list[0].crs)
        negative_output_path = os.path.join(output_folder, "negative_points.gpkg")
        gdf_negative_all.to_file(negative_output_path, driver='GPKG')
        print(f"Negative sample points for all rasters saved to {negative_output_path}")
    else:
        print("No negative points generated; check raster files and disposition layer.")
