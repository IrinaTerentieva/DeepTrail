import geopandas as gpd
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
import numpy as np
import rasterio
from skimage.graph import route_through_array
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.ops import linemerge, unary_union


def merge_connected_lines(gdf):
    """
    Merge all intersecting or connected lines from the original GeoDataFrame into single entities.
    Accumulate their lengths as well.

    Args:
        gdf (GeoDataFrame): Original GeoDataFrame containing LineString geometries.

    Returns:
        GeoDataFrame: GeoDataFrame with merged LineString geometries and updated lengths.
    """
    # Merge all connected lines into multi-lines
    merged_lines = unary_union(gdf.geometry)

    # Handle different resulting types after merging
    merged_geometries = []
    merged_lengths = []

    if merged_lines.geom_type == 'LineString':
        # If the result is a single LineString
        merged_geometries.append(merged_lines)
        merged_lengths.append(merged_lines.length)
    elif merged_lines.geom_type == 'MultiLineString':
        # If the result is multiple LineStrings (e.g., MultiLineString)
        for line in merged_lines.geoms:  # Access individual LineStrings in MultiLineString
            merged_geometries.append(line)
            merged_lengths.append(line.length)
    else:
        raise ValueError(f"Unexpected geometry type: {merged_lines.geom_type}")

    # Create the GeoDataFrame for the merged geometries
    merged_gdf = gpd.GeoDataFrame({'geometry': merged_geometries, 'length': merged_lengths}, crs=gdf.crs)

    return merged_gdf


def classify_trail_length(length, length_median = 10):
    """Classify trail length into 'short' or 'long' based on median length."""
    return 'long_trail' if length >= length_median else 'short_trail'

def connect_segments(
    gdf, raster_path, connection_threshold=200, tolerance=5, high_cost_factor=1.5,
    low_prob_threshold=3, long_trail_check_points=20, long_trail_max_connections=5,
    short_trail_check_points=5, short_trail_max_connections=1
):

    gdf = merge_connected_lines(gdf)
    print('Initial lines connected')

    with rasterio.open(raster_path) as src:
        probability_map = src.read(1)
        transform = src.transform

    points, line_ids = [], []
    line_probabilities = []
    line_classifications = []

    # Calculate median length for classification
    threshold_length = np.percentile(gdf['length'], 70)
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
    cost_map = np.array(100 - probability_map, dtype=np.float32)
    cost_map = np.where(cost_map > 50, cost_map * high_cost_factor, cost_map)
    cost_map[probability_map < low_prob_threshold] = 1000

    all_coords = np.array([(p.x, p.y) for p in points])
    tree = cKDTree(all_coords)
    closest_connections, median_costs, step_costs, lengths, connection_types = [], [], [], [], []

    def point_to_raster_indices(point, transform):
        col, row = ~transform * (point.x, point.y)
        return int(row), int(col)

    # PRIORITIZE long_trails FIRST
    long_trail_indices = [i for i, length in enumerate(gdf['length']) if classify_trail_length(length, threshold_length) == 'long_trail']

    def process_trails(indices, check_points, max_connections, prioritize_long_trail=True):
        for i in indices:
            distances, nearest_indices = tree.query(all_coords[i], k=min(len(all_coords), check_points + 1))
            nearest_indices = nearest_indices[1:]  # Skip itself

            connections = 0
            # Try to connect to other long-trail points first if prioritize_long_trail is True
            if prioritize_long_trail and classify_trail_length(gdf.loc[line_ids[i], 'length'],
                                                               threshold_length) == 'long_trail':
                long_trail_nearest_indices = [
                    idx for idx in nearest_indices
                    if classify_trail_length(gdf.loc[line_ids[idx], 'length'], threshold_length) == 'long_trail'
                ]
                # If there are any long-trail candidates, try to connect them first
                nearest_indices = long_trail_nearest_indices if long_trail_nearest_indices else nearest_indices

            for nearest_idx in nearest_indices:
                if connections >= max_connections:
                    break
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

                        closest_connections.append(path_line)
                        median_costs.append(median_cost)
                        step_costs.append(step_cost)
                        lengths.append(path_line.length)

                        # Determine connection type
                        classification_1 = classify_trail_length(gdf.loc[line_ids[i], 'length'], threshold_length)
                        classification_2 = classify_trail_length(gdf.loc[line_ids[nearest_idx], 'length'],
                                                                 threshold_length)
                        if classification_1 == 'long_trail' and classification_2 == 'long_trail':
                            connection_types.append('long-long')
                        elif classification_1 == 'short_trail' and classification_2 == 'short_trail':
                            connection_types.append('short-short')
                        else:
                            connection_types.append('short-long')

                        connections += 1
                except ValueError as e:
                    print(f"Error while processing connection: {e}")
                    continue

    print('Process long trails')
    # First, connect long trails with higher priority
    process_trails(long_trail_indices, check_points=long_trail_check_points, max_connections=long_trail_max_connections)

    print('Process short trails')
    # Connect the remaining trails
    process_trails(range(len(all_coords)), check_points=short_trail_check_points, max_connections=short_trail_max_connections)

    # Create a GeoDataFrame for the connected lines
    connections_gdf = gpd.GeoDataFrame({
        'geometry': closest_connections,
        'median_cost': median_costs,
        'step_cost': step_costs,
        'length': lengths,
        'connection_type': connection_types
    }, crs=gdf.crs)

    print(f'Merging connections of gdf of {len(gdf)} length and connections_gdf of {len(connections_gdf)} length')

    # Combine the original and newly created connections
    gdf['connection_type'] = 'trail'
    combined_gdf = pd.concat([gdf, connections_gdf], ignore_index=True)
    print('Merged')

    # Create a GeoDataFrame for the points with classifications
    point_classifications = [
        classify_trail_length(combined_gdf.loc[line_id, 'length'], threshold_length)
        for line_id in line_ids
    ]

    points_gdf = gpd.GeoDataFrame({
        'geometry': points,
        'line_id': line_ids,
        'classification': point_classifications
    }, crs=gdf.crs)

    # Save points with classifications
    points_output_path = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm/connected_segment/endpoints.gpkg'
    points_gdf.to_file(points_output_path, driver='GPKG')
    print(f"Saved points with classifications to {points_output_path}")

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
        simplified_gdf = connect_segments(gdf, raster_file, connection_threshold=threshold_distance,
                                          tolerance=5, high_cost_factor=1.5,
                                          low_prob_threshold=3, long_trail_check_points=10,
                                          long_trail_max_connections=5,
                                          short_trail_check_points=1, short_trail_max_connections=1)

        print('Filtering out short lines')
        # Filter out lines shorter than a certain length
        simplified_gdf = simplified_gdf[simplified_gdf.geometry.length >= 1]

        print('Saving connections')
        # Save the connected output as GeoPackage
        output_gpkg = os.path.join(output_folder, f"{base_name}_connected.gpkg")
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
    # # Paths for the input centerline files, corresponding rasters, and output directory
    # centerline_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm/centerline/test'
    # output_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm/connected_segment'
    #
    # centerline_folder = '/media/irro/All/HumanFootprint/DATA/Test_Models/temp/centerline_20thre'
    # raster_folder = '/media/irro/All/HumanFootprint/DATA/Test_Models/temp'
    # output_folder = '/media/irro/All/HumanFootprint/DATA/Test_Models/temp/connected_segment'
    #
    # centerline_folder = '/media/irro/All/HumanFootprint/DATA/intermediate/connect'
    # raster_folder = '/media/irro/All/HumanFootprint/DATA/Products/Unet/Kirby/DTM10cm'
    # output_folder = '/media/irro/All/HumanFootprint/DATA/intermediate/connect/connected_segment'

    centerline_folder = '/media/irro/All/HumanFootprint/DATA/Test_Models/centerline'
    raster_folder = '/media/irro/All/BlueberryFN/DATA/Drone/PA2_W2_West-South_Sikanni_Road/LiDAR_derivatives'
    output_folder = '/media/irro/All/BlueberryFN/DATA/Drone/PA2_W2_West-South_Sikanni_Road/connected_segment'

    # Process files in parallel with 4 workers
    process_files_in_folder(centerline_folder, raster_folder, output_folder, threshold_distance=200, max_workers=1)

if __name__ == "__main__":
    main()