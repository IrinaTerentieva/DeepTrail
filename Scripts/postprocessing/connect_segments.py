import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import numpy as np
import random


def visualize_lines_with_endpoints(gdf, plot_endpoints=True, plot_connections=True, connection_threshold=None):
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

    for i, point in enumerate(all_coords):
        if i in used_indices:
            continue
        distances, indices = tree.query(point, k=2)
        nearest_idx = indices[1]  # Get the closest point index (skip self with index 0)
        nearest_distance = distances[1]  # Get the distance to the closest point

        # Check if the points belong to different lines and are within the threshold
        if line_ids[i] != line_ids[nearest_idx] and (
                connection_threshold is None or nearest_distance <= connection_threshold):
            used_indices.add(i)
            used_indices.add(nearest_idx)
            closest_connections.append(LineString([points[i], points[nearest_idx]]))

    # Create GeoDataFrames for points and connections
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=gdf.crs)
    connections_gdf = gpd.GeoDataFrame(geometry=closest_connections, crs=gdf.crs)

    # Assign a color to each line ID
    unique_ids = list(set(line_ids))
    random.seed(42)  # Fix the random seed for reproducibility
    colors = {line_id: plt.cm.get_cmap('tab20')(random.randint(0, 19)) for line_id in unique_ids}

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the original LineStrings
    gdf.plot(ax=ax, color='blue', linewidth=1, label='LineStrings')

    if plot_endpoints:
        # Plot points colored by their corresponding line ID
        for i, point in points_gdf.iterrows():
            ax.plot(point.geometry.x, point.geometry.y, marker='o', color=colors[line_ids[i]], markersize=10)

    if plot_connections:
        # Plot the closest connections between endpoints
        connections_gdf.plot(ax=ax, color='orange', linestyle='--', linewidth=1, label='Connections')

    # Add title and legend
    plt.title('LineStrings with Points and Closest Connections')
    plt.legend()

    # Show plot if any visualization flag is true
    if plot_endpoints or plot_connections:
        plt.show()


# Load the shapefile
input_path = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label.shp'
gdf = gpd.read_file(input_path)

# Set threshold distance for connecting points (e.g., 100 meters)
threshold_distance = 100.0

# Call the function with visualization options and a connection threshold
visualize_lines_with_endpoints(gdf, plot_endpoints=True, plot_connections=True, connection_threshold=threshold_distance)
