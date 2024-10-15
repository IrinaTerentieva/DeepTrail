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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Classification thresholds for trail length
SHORT_TRAIL_THRESHOLD = 50  # Example threshold for short trails
MEDIUM_TRAIL_THRESHOLD = 150  # Example threshold for medium trails


def classify_trail_length(length, length_median = 10):
    """Classify trail length into 'short' or 'long' based on median length."""
    return 'long_trail' if length >= length_median else 'short_trail'

def connect_segments(gdf, raster_path, connection_threshold=200, max_connections=3, tolerance=5, high_cost_factor=1.5,
                     low_prob_threshold=3):

    print("Starting connect_segments")

    with rasterio.open(raster_path) as src:
        probability_map = src.read(1)
        transform = src.transform

    points, line_ids = [], []
    line_probabilities = []
    line_classifications = []

    # Calculate median length for classification
    threshold_length = np.percentile(gdf['length'], 75)
    print('Threshold trail length: ', threshold_length)

    for idx, row in gdf.iterrows():
        line = row.geometry
        if line.is_empty or line.geom_type != 'LineString':
            continue
        start_point, end_point = Point(line.coords[0]), Point(line.coords[-1])
        points.extend([start_point, end_point])
        line_ids.extend([idx, idx])

        # Calculate average probability along the line
        line_coords = np.array(line.coords)
        line_raster_indices = np.apply_along_axis(lambda coord: transform * (coord[0], coord[1]), 1, line_coords)

        # Filter out invalid indices
        line_raster_indices = line_raster_indices.astype(float)
        valid_indices = [
            (int(row), int(col)) for col, row in line_raster_indices
            if not np.isnan(row) and not np.isnan(col)
            and 0 <= int(row) < probability_map.shape[0]
            and 0 <= int(col) < probability_map.shape[1]
        ]

        if valid_indices:
            line_prob = np.mean([probability_map[row, col] for row, col in valid_indices])
        else:
            line_prob = np.nan

        line_probabilities.append(line_prob)

        # Classify the line
        classification = classify_trail_length(row['length'], threshold_length)
        line_classifications.append(classification)

    # Use a cost map based on probability
    cost_map = 100 - probability_map
    cost_map = np.where(cost_map > 50, cost_map * high_cost_factor, cost_map)
    cost_map[probability_map < low_prob_threshold] = 1000

    all_coords = np.array([(p.x, p.y) for p in points])
    tree = cKDTree(all_coords)
    closest_connections, median_costs, step_costs, lengths, connection_types = [], [], [], [], []
    used_indices = set()

    def point_to_raster_indices(point, transform):
        col, row = ~transform * (point.x, point.y)
        return int(row), int(col)

    # PRIORITIZE long_trails FIRST
    long_trail_indices = [i for i, length in enumerate(gdf['length']) if classify_trail_length(length, threshold_length) == 'long_trail']

    def process_trails(indices, connection_threshold, max_connections):
        for i in indices:
            # Allow a maximum of 3 connections if the point belongs to a long trail
            current_max_connections = max_connections
            if classify_trail_length(gdf.loc[line_ids[i], 'length'], threshold_length) == 'long_trail':
                current_max_connections = 3

            distances, nearest_indices = tree.query(all_coords[i], k=min(len(all_coords), current_max_connections + 1))
            nearest_indices = nearest_indices[1:]  # Skip itself

            best_connection, best_median_cost, best_step_cost = None, float('inf'), float('inf')
            connection_type = 'unknown'

            for nearest_idx in nearest_indices:
                # Skip if trying to connect to the same line
                if line_ids[i] == line_ids[nearest_idx]:
                    continue

                try:
                    start_raster_idx = point_to_raster_indices(Point(all_coords[i]), transform)
                    end_raster_idx = point_to_raster_indices(Point(all_coords[nearest_idx]), transform)

                    # Skip if indices are invalid (e.g., out of bounds)
                    if not (0 <= start_raster_idx[0] < cost_map.shape[0] and 0 <= start_raster_idx[1] < cost_map.shape[
                        1]):
                        continue
                    if not (0 <= end_raster_idx[0] < cost_map.shape[0] and 0 <= end_raster_idx[1] < cost_map.shape[1]):
                        continue

                    indices_path, _ = route_through_array(cost_map, start_raster_idx, end_raster_idx,
                                                          fully_connected=True)
                    path_costs = [cost_map[row, col] for row, col in indices_path]

                    median_cost = np.median(path_costs)
                    step_cost = np.mean(path_costs)
                    path_coords = [transform * (col, row) for row, col in indices_path]

                    if len(path_coords) > 1:
                        path_line = LineString(path_coords)

                        if median_cost < best_median_cost:
                            best_connection = path_line
                            best_median_cost = median_cost
                            best_step_cost = step_cost

                            # Determine connection type
                            classification_1 = classify_trail_length(gdf.loc[line_ids[i], 'length'], threshold_length)
                            classification_2 = classify_trail_length(gdf.loc[line_ids[nearest_idx], 'length'],
                                                                     threshold_length)
                            if classification_1 == 'long_trail' and classification_2 == 'long_trail':
                                connection_type = 'long-long'
                            elif classification_1 == 'short_trail' and classification_2 == 'short_trail':
                                connection_type = 'short-short'
                            else:
                                connection_type = 'short-long'
                except ValueError as e:
                    print(f"Error while processing connection: {e}")
                    continue

            if best_connection is not None:
                # Record the connection
                closest_connections.append(best_connection)
                median_costs.append(best_median_cost)
                step_costs.append(best_step_cost)
                lengths.append(best_connection.length)
                connection_types.append(connection_type)

    print('Process long trails')
    # First, connect long trails with higher priority
    process_trails(long_trail_indices, connection_threshold=300, max_connections=10)

    print('Process short trails')
    # Connect the remaining trails
    process_trails(range(len(all_coords)), connection_threshold=100, max_connections=3)

    # Create a GeoDataFrame for the connected lines
    connections_gdf = gpd.GeoDataFrame({
        'geometry': closest_connections,
        'median_cost': median_costs,
        'step_cost': step_costs,
        'length': lengths,
        'connection_type': connection_types
    }, crs=gdf.crs)

    print(f'Merging connections of gdf of {len(gdf)} length and connections_gdf of {len(connections_gdf)} length')
    print('gdf: ', gdf.head())
    print('connections_gdf: ', connections_gdf.head())

    # Combine the original and newly created connections
    gdf['connection_type'] = 'trail'
    combined_gdf = pd.concat([gdf, connections_gdf], ignore_index=True)
    print('Merged')

    # Extend the lists for new connections
    num_new_connections = len(connections_gdf)
    line_probabilities.extend([np.nan] * num_new_connections)
    line_classifications.extend(['trail'] * num_new_connections)

    # Use the provided 'length' for the line length
    if 'length' in combined_gdf.columns:
        combined_gdf['line_length'] = combined_gdf['length']
    else:
        combined_gdf['line_length'] = combined_gdf.geometry.length

    combined_gdf['avg_probability'] = line_probabilities
    combined_gdf['classification'] = line_classifications

    # Create a GeoDataFrame for the points with classifications
    point_classifications = [
        classify_trail_length(combined_gdf.loc[line_id, 'line_length'], threshold_length)
        for line_id in line_ids
    ]

    points_gdf = gpd.GeoDataFrame({
        'geometry': points,
        'line_id': line_ids,
        'classification': point_classifications
    }, crs=gdf.crs)

    # # Save points with classifications
    # points_output_path = os.path.join(os.path.dirname(raster_path), 'endpoints.gpkg')
    # points_gdf.to_file(points_output_path, driver='GPKG')
    # print(f"Saved points with classifications to {points_output_path}")

    return combined_gdf

def process_file(centerline_file, centerline_folder, raster_folder, output_folder, threshold_distance):
    base_name = os.path.splitext(centerline_file)[0]
    raster_file = os.path.join(raster_folder, f"{base_name}.tif")
    centerline_path = os.path.join(centerline_folder, centerline_file)

    if os.path.exists(raster_file):
        print(f"Processing: {centerline_path} with {raster_file}")

        # Load centerline GeoPackage
        gdf = gpd.read_file(centerline_path)

        print('Starting connect_segment')
        simplified_gdf = connect_segments(gdf, raster_file, connection_threshold=threshold_distance, high_cost_factor=1.5,
                                                     low_prob_threshold=3)
        print('Filtering out short lines')
        # Filter out lines shorter than a certain length
        simplified_gdf = simplified_gdf[simplified_gdf.geometry.length >= 1]

        print('Saving connections')
        # Save the connected output as GeoPackage
        output_gpkg = os.path.join(output_folder, f"{base_name}_connect.gpkg")
        simplified_gdf.to_file(output_gpkg, driver='GPKG')
        print(f"Saved simplified output as GeoPackage to {output_gpkg}")
    else:
        print(f"Raster file not found for {centerline_file}")

def process_files_in_folder(centerline_folder, raster_folder, output_folder, threshold_distance=200, max_workers=4):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List of centerline files
    centerline_files = [f for f in os.listdir(centerline_folder) if f.endswith('.gpkg')]

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, centerline_file, centerline_folder, raster_folder, output_folder,
                            threshold_distance)
            for centerline_file in centerline_files
        ]

        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()  # Raises any exception encountered during processing
            except Exception as exc:
                print(f"Generated an exception: {exc}")

def main():
    # Paths for the input centerline files, corresponding rasters, and output directory
    centerline_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm/centerline/test'
    raster_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm'
    output_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm/connected_segment'

    # Process files in parallel with 4 workers
    process_files_in_folder(centerline_folder, raster_folder, output_folder, threshold_distance=200, max_workers=1)

if __name__ == "__main__":
    main()