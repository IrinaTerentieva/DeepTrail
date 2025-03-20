import os
import geopandas as gpd
import rasterio

# Define file paths.
negative_points_path = "/media/irina/My Book/Surmont/intermediate/segformer_wettrails/negative_points.gpkg"
rasters_folder = "/media/irina/My Book1/Conoco/DATA/Orthos/3_Orthomosaics_10cm/"

# List all raster files in the given directory (assumed to be TIFF files).
raster_files = [os.path.join(rasters_folder, f)
                for f in os.listdir(rasters_folder) if f.lower().endswith(".tif")]

def find_raster_for_point(raster_files, point):
    """
    For a given point geometry, returns the basename of the first raster that contains the point.
    """
    for raster_path in raster_files:
        try:
            with rasterio.open(raster_path) as src:
                bounds = src.bounds
                if bounds.left <= point.x <= bounds.right and bounds.bottom <= point.y <= bounds.top:
                    return 'labels_' + os.path.basename(raster_path)
        except Exception as e:
            print(f"Error processing raster {raster_path}: {e}")
    return None

# Load the negative points GeoPackage.
gdf_neg = gpd.read_file(negative_points_path)

# For each point, assign the raster basename that intersects the point.
gdf_neg["raster1"] = gdf_neg.geometry.apply(lambda geom: find_raster_for_point(raster_files, geom) if geom is not None else None)

# Save the updated GeoPackage (you can overwrite or save to a new file).
gdf_neg.to_file(negative_points_path, driver="GPKG")
print("Updated negative_points.gpkg with column 'raster1'")
