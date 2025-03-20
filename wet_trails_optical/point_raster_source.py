import os
import geopandas as gpd
import rasterio
from rasterio.windows import Window


def get_intersecting_rasters_for_point(point, raster_files):
    """
    For a given point geometry, returns a list of raster file base names for which the pixel value is not 0.
    If the geometry is a MultiPoint, the first point is used.
    """
    # If the geometry is a MultiPoint, use its first component.
    if point.geom_type == "MultiPoint":
        point = list(point.geoms)[0]

    intersecting_rasters = []
    for raster_path in raster_files:
        try:
            with rasterio.open(raster_path) as src:
                # Check if point is within raster bounds.
                if src.bounds.left <= point.x <= src.bounds.right and src.bounds.bottom <= point.y <= src.bounds.top:
                    # Convert the point to pixel coordinates.
                    row, col = src.index(point.x, point.y)
                    # Ensure the pixel coordinates are within raster dimensions.
                    if 0 <= row < src.height and 0 <= col < src.width:
                        # Read the single pixel value.
                        window = Window(col, row, 1, 1)
                        pixel_value = src.read(1, window=window)[0, 0]
                        # If the value is not 0, consider this raster as intersecting.
                        if pixel_value != 0:
                            intersecting_rasters.append(os.path.basename(raster_path))
        except Exception as e:
            print(f"Error processing raster {raster_path} for point {point}: {e}")
    return intersecting_rasters


if __name__ == '__main__':
    # Set your file paths.
    points_path = "/media/irina/My Book/Surmont/manual/wet_trails_Surmont_2024.gpkg"  # Input points layer
    rasters_folder = "/media/irina/My Book/Surmont/intermediate/segformer_wettrails"  # Folder containing label rasters (.tif)
    output_points_path = "/media/irina/My Book/Surmont/intermediate/points_with_rasters.gpkg"  # Output GeoPackage

    # Load the points.
    gdf_points = gpd.read_file(points_path)

    # List all .tif files in the rasters folder.
    raster_files = [os.path.join(rasters_folder, f)
                    for f in os.listdir(rasters_folder) if f.lower().endswith('.tif')]

    # For each point, get the list of intersecting rasters (non-zero value at the pixel).
    raster_info = []
    max_count = 0
    for idx, row in gdf_points.iterrows():
        point = row.geometry
        rasters_found = get_intersecting_rasters_for_point(point, raster_files)
        raster_info.append(rasters_found)
        if len(rasters_found) > max_count:
            max_count = len(rasters_found)

    # Create new columns "raster1", "raster2", ... based on the maximum number of rasters found.
    for i in range(max_count):
        col_name = f"raster{i + 1}"
        # For each point, assign the i-th raster name if available, otherwise None.
        gdf_points[col_name] = [rasters[i] if len(rasters) > i else None for rasters in raster_info]

    # Optionally, add a column that stores all intersecting rasters as a concatenated string.
    gdf_points["rasters_all"] = [";".join(rasters) for rasters in raster_info]

    # Save the updated points layer.
    gdf_points.to_file(output_points_path, driver="GPKG")
    print(f"Updated point layer with intersecting rasters saved to {output_points_path}")
