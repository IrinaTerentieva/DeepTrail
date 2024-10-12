import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import numpy as np
import rasterio
from skimage.graph import route_through_array
from shapely.ops import unary_union
import pandas as pd
import os

def plot_probability_map(probability_map):
    """Plot the probability map."""
    plt.figure(figsize=(10, 10))
    plt.imshow(probability_map, cmap='viridis')
    plt.colorbar(label='Probability')
    plt.title('Original Probability Map')
    plt.show()

def plot_cost_map(cost_map):
    """Plot the cost map."""
    plt.figure(figsize=(10, 10))
    plt.imshow(cost_map, cmap='hot')
    plt.colorbar(label='Cost')
    plt.title('Cost Map')
    plt.show()

def remove_duplicate_lines(gdf, tolerance=5):
    """
    Simplify the GeoDataFrame by removing nearby duplicate lines based on proximity.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame containing LineString geometries.
        tolerance (int): Distance threshold for considering lines as duplicates.

    Returns:
        GeoDataFrame: Simplified GeoDataFrame without duplicate lines.
    """
    gdf['buffer'] = gdf.geometry.buffer(tolerance)
    gdf['representative'] = gdf['buffer'].apply(lambda x: unary_union([x]))
    simplified_gdf = gdf.drop_duplicates(subset='representative')
    return simplified_gdf.drop(columns=['buffer', 'representative'])

def connect_segments(gdf, raster_path, plot_endpoints=True, plot_connections=True,
                     connection_threshold=200, max_connections=3, tolerance=5, high_cost_factor=1.5,
                     low_prob_threshold=3):
    """
    Connect segments of lines in the GeoDataFrame based on the cost/probability raster.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing LineString geometries to be connected.
        raster_path (str): Path to the raster probability map.
        plot_endpoints (bool): Whether to plot start and end points of each segment.
        plot_connections (bool): Whether to plot newly generated connections.
        connection_threshold (float): Maximum distance threshold for connecting segments.
        max_connections (int): Maximum number of closest points to consider for connection.
        tolerance (int): Distance tolerance for removing duplicate lines.
        high_cost_factor (float): Multiplier for increasing costs in high-cost areas.
        low_prob_threshold (int): Probability threshold below which areas are considered very costly.

    Returns:
        GeoDataFrame: Simplified GeoDataFrame with connected line segments.
    """
    with rasterio.open(raster_path) as src:
        probability_map = src.read(1)
        transform = src.transform

    plot_probability_map(probability_map)

    points, line_ids = [], []

    for idx, row in gdf.iterrows():
        line = row.geometry
        if line.is_empty or line.geom_type != 'LineString':
            continue
        start_point, end_point = Point(line.coords[0]), Point(line.coords[-1])
        points.extend([start_point, end_point])
        line_ids.extend([idx, idx])

    cost_map = 100 - probability_map
    cost_map = np.where(cost_map > 50, cost_map * high_cost_factor, cost_map)
    cost_map[probability_map < low_prob_threshold] = 1000
    plot_cost_map(cost_map)

    all_coords = np.array([(p.x, p.y) for p in points])
    tree = cKDTree(all_coords)
    closest_connections, median_costs, step_costs, lengths = [], [], [], []
    used_indices = set()

    def point_to_raster_indices(point, transform):
        col, row = ~transform * (point.x, point.y)
        return int(row), int(col)

    for i, point in enumerate(all_coords):
        if i in used_indices:
            continue

        distances, indices = tree.query(point, k=min(len(all_coords), max_connections + 1))
        indices = indices[1:]

        best_connection, best_median_cost, best_step_cost = None, float('inf'), float('inf')

        for nearest_idx in indices:
            if line_ids[i] == line_ids[nearest_idx]:
                continue
            start_raster_idx = point_to_raster_indices(Point(all_coords[i]), transform)
            end_raster_idx = point_to_raster_indices(Point(all_coords[nearest_idx]), transform)

            indices_path, _ = route_through_array(cost_map, start_raster_idx, end_raster_idx, fully_connected=True)
            path_costs = [cost_map[row, col] for row, col in indices_path]

            median_cost = np.median(path_costs)
            step_cost = np.mean(path_costs)
            path_coords = [src.transform * (col, row) for row, col in indices_path]

            if len(path_coords) > 1:
                path_line = LineString(path_coords)
                path_length = path_line.length

                if median_cost < best_median_cost:
                    best_connection = path_line
                    best_median_cost = median_cost
                    best_step_cost = step_cost

        if best_connection is not None:
            closest_connections.append(best_connection)
            median_costs.append(best_median_cost)
            step_costs.append(best_step_cost)
            lengths.append(best_connection.length)

    connections_gdf = gpd.GeoDataFrame({
        'geometry': closest_connections,
        'median_cost': median_costs,
        'step_cost': step_costs,
        'length': lengths
    }, crs=gdf.crs)

    combined_gdf = pd.concat([gdf, connections_gdf], ignore_index=True)
    simplified_gdf = remove_duplicate_lines(combined_gdf, tolerance=tolerance)

    return simplified_gdf

# Load the shapefile
# input_shapefile = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label.shp'
input_shapefile = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label_connected_200.0_cost_mean.gpkg'

base_name = os.path.splitext(os.path.basename(input_shapefile))[0]

gdf = gpd.read_file(input_shapefile)
gdf = gdf.to_crs(epsg=2956)
gdf['length'] = gdf.geometry.length

# Filter out lines shorter than a certain length
gdf = gdf[gdf.geometry.length >= 1]

# Set raster file path and parameters
raster_file = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label.tif'
threshold_distance = 200.0

# Call function with parameters for flexibility
simplified_gdf = connect_segments(gdf, raster_file, plot_endpoints=False, plot_connections=False,
                                  connection_threshold=threshold_distance, high_cost_factor=1.5,
                                  low_prob_threshold=3)

# Set output path including relevant parameters for identification
output_gpkg = f'/media/irro/All/HumanFootprint/DATA/intermediate/{base_name}_connected_{threshold_distance}_cost_mean.gpkg'
simplified_gdf.to_file(output_gpkg, driver='GPKG')
print(f"Saved simplified output as GeoPackage to {output_gpkg}")
