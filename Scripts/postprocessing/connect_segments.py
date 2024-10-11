import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import numpy as np
import rasterio
from skimage.graph import route_through_array

def visualize_lines_with_least_cost_paths(gdf, raster_path, plot_endpoints=True, plot_connections=True, connection_threshold=None):
    # Load the raster (probability map)
    with rasterio.open(raster_path) as src:
        probability_map = src.read(1)
        transform = src.transform

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

    # Convert points to numpy array for efficient distance calculations
    all_coords = np.array([(p.x, p.y) for p in points])

    # Use KDTree for efficient nearest-neighbor search
    tree = cKDTree(all_coords)
    closest_connections = []

    used_indices = set()  # Keep track of connected points

    # Helper function to convert geographic coordinates to raster indices
    def point_to_raster_indices(point, transform):
        col, row = ~transform * (point.x, point.y)
        return int(row), int(col)

    for i, point in enumerate(all_coords):
        if i in used_indices:
            continue
        distances, indices = tree.query(point, k=2)
        nearest_idx = indices[1]  # Get the closest point index (skip self with index 0)
        nearest_distance = distances[1]  # Get the distance to the closest point

        # Check if the points belong to different lines and are within the threshold
        if line_ids[i] != line_ids[nearest_idx] and (connection_threshold is None or nearest_distance <= connection_threshold):
            used_indices.add(i)
            used_indices.add(nearest_idx)

            # Convert points to raster indices
            start_raster_idx = point_to_raster_indices(Point(all_coords[i]), transform)
            end_raster_idx = point_to_raster_indices(Point(all_coords[nearest_idx]), transform)

            # Calculate the least cost path based on the probability map
            cost_map = 100 - probability_map  # Convert probability to cost (lower values = less cost)
            indices, _ = route_through_array(cost_map, start_raster_idx, end_raster_idx, fully_connected=True)

            # Convert raster indices back to geographic coordinates
            path_coords = [src.transform * (col, row) for row, col in indices]

            # Only create LineString if there are at least two points
            if len(path_coords) > 1:
                path_line = LineString(path_coords)
                closest_connections.append(path_line)

    # Create GeoDataFrames for points and connections
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=gdf.crs)
    connections_gdf = gpd.GeoDataFrame(geometry=closest_connections, crs=gdf.crs)

    # Assign a color to each line ID
    unique_ids = list(set(line_ids))
    colors = {line_id: plt.cm.get_cmap('tab20')(i % 20) for i, line_id in enumerate(unique_ids)}

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the original LineStrings
    gdf.plot(ax=ax, color='blue', linewidth=1, label='LineStrings')

    if plot_endpoints:
        # Plot points colored by their corresponding line ID
        for i, point in points_gdf.iterrows():
            ax.plot(point.geometry.x, point.geometry.y, marker='o', color=colors[line_ids[i]], markersize=10)

    if plot_connections:
        # Plot the least cost path connections
        connections_gdf.plot(ax=ax, color='orange', linestyle='--', linewidth=1, label='Connections')

    # Add title and legend
    plt.title('LineStrings with Points and Least Cost Connections')
    plt.legend()

    # Show plot if any visualization flag is true
    if plot_endpoints or plot_connections:
        plt.show()

    return connections_gdf

# Load the shapefile
input_shapefile = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label.shp'
gdf = gpd.read_file(input_shapefile)

# Set the raster file path (probability map)
raster_file = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label.tif'

# Set threshold distance for connecting points (e.g., 100 meters)
threshold_distance = 100.0

# Call the function with visualization options and a connection threshold
connections_gdf = visualize_lines_with_least_cost_paths(gdf, raster_file, plot_endpoints=True, plot_connections=True, connection_threshold=threshold_distance)

# Specify the output path for the GeoPackage
output_gpkg = '/media/irro/All/HumanFootprint/DATA/intermediate/least_cost_connections.gpkg'

# Save the GeoDataFrame with the connections to the GeoPackage
connections_gdf.to_file(output_gpkg, driver='GPKG')

print(f"Saved output as GeoPackage to {output_gpkg}")
