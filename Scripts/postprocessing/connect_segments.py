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
    plt.figure(figsize=(10, 10))
    plt.imshow(probability_map, cmap='viridis')
    plt.colorbar(label='Probability')
    plt.title('Original Probability Map')
    plt.show()


def plot_cost_map(cost_map):
    plt.figure(figsize=(10, 10))
    plt.imshow(cost_map, cmap='hot')
    plt.colorbar(label='Cost')
    plt.title('Cost Map')
    plt.show()


def remove_duplicate_lines(gdf, tolerance=5):
    """
    Simplifies the GeoDataFrame by removing lines that are very close to each other.
    The tolerance defines the distance below which lines are considered duplicative.
    """
    # Spatially join lines that are within the tolerance
    gdf['buffer'] = gdf.geometry.buffer(tolerance)
    gdf['representative'] = gdf['buffer'].apply(lambda x: unary_union([x]))

    # Drop duplicates based on spatial overlap (first occurrence is kept)
    simplified_gdf = gdf.drop_duplicates(subset='representative')

    # Drop the helper columns
    simplified_gdf = simplified_gdf.drop(columns=['buffer', 'representative'])

    return simplified_gdf


def visualize_and_simplify_lines(gdf, raster_path, plot_endpoints=True, plot_connections=True,
                                 connection_threshold=None, buffer_size=10, tolerance=5):
    # Load the raster (probability map)
    with rasterio.open(raster_path) as src:
        probability_map = src.read(1)
        transform = src.transform

    # Debug: Plot the probability map
    plot_probability_map(probability_map)

    # Create lists to store the points and line IDs
    points = []
    line_ids = []

    # Extract start and end points from each LineString and assign line IDs
    for idx, row in gdf.iterrows():
        line = row.geometry
        if line.is_empty or line.geom_type != 'LineString':
            continue  # Skip if not LineString or empty

        # Extract start and end points
        start_point = Point(line.coords[0])
        end_point = Point(line.coords[-1])

        # Append points to the list, assign the same line ID to both start and end points
        points.append(start_point)
        points.append(end_point)
        line_ids.append(idx)  # Same line ID for both start and end points
        line_ids.append(idx)  # Same line ID for both start and end points

    # Modify the cost map: Set very high cost for areas with zero probability
    cost_map = 100 - probability_map  # Convert probability to cost
    cost_map = np.where(cost_map > 50, cost_map * 1.5, cost_map)
    cost_map[probability_map < 3] = 200  # Assign very large cost to 0-probability areas

    # Debug: Plot the cost map
    plot_cost_map(cost_map)

    # Check how many areas have high cost
    zero_prob_count = np.sum(probability_map == 0)
    high_cost_count = np.sum(cost_map == 200)
    print(f"Number of 0-probability areas: {zero_prob_count}")
    print(f"Number of areas with high cost: {high_cost_count}")

    # Convert points to numpy array for efficient distance calculations
    all_coords = np.array([(p.x, p.y) for p in points])

    # Use KDTree for efficient nearest-neighbor search
    tree = cKDTree(all_coords)
    closest_connections = []
    median_costs = []  # To store median costs for each connection
    lengths = []  # To store the length of each connection

    used_indices = set()  # Keep track of connected points

    # Helper function to convert geographic coordinates to raster indices
    def point_to_raster_indices(point, transform):
        col, row = ~transform * (point.x, point.y)
        return int(row), int(col)

    # Loop through each point to connect with all points in the 10-meter buffer
    for i, point in enumerate(all_coords):
        if i in used_indices:
            continue

        # Find all points within a 10-meter buffer
        buffer_points = tree.query_ball_point(point, r=buffer_size)

        best_connection = None
        best_median_cost = float('inf')  # Set to infinity to find the best (lowest) median cost

        # Try to connect to all points in the buffer
        for nearest_idx in buffer_points:
            if i == nearest_idx or line_ids[i] == line_ids[nearest_idx]:
                continue  # Skip self and points from the same line

            # Convert points to raster indices
            start_raster_idx = point_to_raster_indices(Point(all_coords[i]), transform)
            end_raster_idx = point_to_raster_indices(Point(all_coords[nearest_idx]), transform)

            # Calculate the least cost path based on the probability map
            indices, _ = route_through_array(cost_map, start_raster_idx, end_raster_idx, fully_connected=True)

            # Extract the costs along the path
            path_costs = [cost_map[row, col] for row, col in indices]

            # Calculate the median cost for the path
            median_cost = np.median(path_costs)

            # Convert raster indices back to geographic coordinates
            path_coords = [src.transform * (col, row) for row, col in indices]

            # Only create LineString if there are at least two points
            if len(path_coords) > 1:
                path_line = LineString(path_coords)
                path_length = path_line.length

                # Store the connection if the median cost is the best found so far
                if median_cost < best_median_cost and path_length < 15:
                    best_connection = path_line
                    best_median_cost = median_cost

        # If a valid connection was found, store it
        if best_connection is not None:
            closest_connections.append(best_connection)
            median_costs.append(best_median_cost)
            lengths.append(best_connection.length)

    # Create GeoDataFrame for the newly generated connections
    connections_gdf = gpd.GeoDataFrame({
        'geometry': closest_connections,
        'median_cost': median_costs,
        'length': lengths
    }, crs=gdf.crs)

    # Combine with original GeoDataFrame
    combined_gdf = pd.concat([gdf, connections_gdf], ignore_index=True)

    # Simplify the combined GeoDataFrame by removing duplicate lines based on proximity
    simplified_gdf = remove_duplicate_lines(combined_gdf, tolerance=tolerance)

    return simplified_gdf


# Load the shapefile
input_shapefile = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label.shp'
base_name = os.path.splitext(os.path.basename(input_shapefile))[0]

gdf = gpd.read_file(input_shapefile)

# Filter out lines shorter than the threshold
gdf = gdf[gdf.geometry.length >= 5]

# Set the raster file path (probability map)
raster_file = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label.tif'

# Set threshold distance for connecting points (e.g., 100 meters)
threshold_distance = 200.0

# Call the function with visualization options and a connection threshold
simplified_gdf = visualize_and_simplify_lines(gdf, raster_file, plot_endpoints=False, plot_connections=False,
                                              connection_threshold=threshold_distance)

# Set the output path by adding "_connected" to the original filename
output_gpkg = f'/media/irro/All/HumanFootprint/DATA/intermediate/{base_name}_connected.gpkg'
simplified_gdf.to_file(output_gpkg, driver='GPKG')
print(f"Saved simplified output as GeoPackage to {output_gpkg}")

