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
from shapely.ops import linemerge, unary_union

# Merge connected lines into single entities
def merge_connected_lines(gdf):
    merged_lines = unary_union(gdf.geometry)
    merged_geometries = []
    merged_lengths = []

    if merged_lines.geom_type == 'LineString':
        merged_geometries.append(merged_lines)
        merged_lengths.append(merged_lines.length)
    elif merged_lines.geom_type == 'MultiLineString':
        for line in merged_lines.geoms:
            merged_geometries.append(line)
            merged_lengths.append(line.length)
    else:
        raise ValueError(f"Unexpected geometry type: {merged_lines.geom_type}")

    merged_gdf = gpd.GeoDataFrame({'geometry': merged_geometries, 'length': merged_lengths}, crs=gdf.crs)
    return merged_gdf

# Classify trail lengths as short or long
def classify_trail_length(length, length_median=10):
    return 'long_trail' if length >= length_median else 'short_trail'

# Process each chunk of the raster and GeoDataFrame
def process_chunk(gdf_chunk, probability_map, transform, cost_map, threshold_length, high_cost_factor, low_prob_threshold):
    points, line_ids = [], []
    line_probabilities = []
    line_classifications = []

    for idx, row in gdf_chunk.iterrows():
        line = row.geometry
        if line.is_empty or line.geom_type != 'LineString':
            continue
        start_point, end_point = Point(line.coords[0]), Point(line.coords[-1])
        points.extend([start_point, end_point])
        line_ids.extend([idx, idx])

        line_coords = np.array(line.coords)
        line_raster_indices = np.apply_along_axis(lambda coord: transform * (coord[0], coord[1]), 1, line_coords)
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
        classification = classify_trail_length(row['length'], threshold_length)
        line_classifications.append(classification)

    return points, line_ids, line_probabilities, line_classifications

def connect_segments(
    gdf, raster_path, connection_threshold=200, tolerance=5, high_cost_factor=1.5,
    low_prob_threshold=3, long_trail_check_points=20, long_trail_max_connections=5,
    short_trail_check_points=5, short_trail_max_connections=1, chunk_size=1000):

    gdf = merge_connected_lines(gdf)
    print('Initial lines connected')

    with rasterio.open(raster_path) as src:
        probability_map = src.read(1)
        transform = src.transform

    cost_map = np.array(100 - probability_map, dtype=np.float32)
    cost_map = np.where(cost_map > 50, cost_map * high_cost_factor, cost_map)
    cost_map[probability_map < low_prob_threshold] = 1000

    threshold_length = np.percentile(gdf['length'], 70)
    print('Threshold trail length: ', threshold_length)

    all_points, all_line_ids, all_probs, all_classifications = [], [], [], []

    for start_idx in range(0, len(gdf), chunk_size):
        gdf_chunk = gdf.iloc[start_idx:start_idx + chunk_size]
        points, line_ids, line_probs, line_classifications = process_chunk(gdf_chunk, probability_map, transform, cost_map, threshold_length, high_cost_factor, low_prob_threshold)
        all_points.extend(points)
        all_line_ids.extend(line_ids)
        all_probs.extend(line_probs)
        all_classifications.extend(line_classifications)

    # Now do your nearest neighbor, cost path, and connection logic on the accumulated data
    all_coords = np.array([(p.x, p.y) for p in all_points])
    tree = cKDTree(all_coords)
    closest_connections, median_costs, step_costs, lengths, connection_types = [], [], [], [], []

    def point_to_raster_indices(point, transform):
        col, row = ~transform * (point.x, point.y)
        return int(row), int(col)

    long_trail_indices = [i for i, length in enumerate(gdf['length']) if classify_trail_length(length, threshold_length) == 'long_trail']

    def process_trails(indices, check_points, max_connections):
        for i in indices:
            distances, nearest_indices = tree.query(all_coords[i], k=min(len(all_coords), check_points + 1))
            nearest_indices = nearest_indices[1:]  # Skip itself

            connections = 0
            for nearest_idx in nearest_indices:
                if connections >= max_connections:
                    break
                if all_line_ids[i] == all_line_ids[nearest_idx]:
                    continue

                try:
                    start_raster_idx = point_to_raster_indices(Point(all_coords[i]), transform)
                    end_raster_idx = point_to_raster_indices(Point(all_coords[nearest_idx]), transform)

                    if not (0 <= start_raster_idx[0] < cost_map.shape[0] and 0 <= start_raster_idx[1] < cost_map.shape[1]):
                        continue
                    if not (0 <= end_raster_idx[0] < cost_map.shape[0] and 0 <= end_raster_idx[1] < cost_map.shape[1]):
                        continue

                    indices_path, _ = route_through_array(cost_map, start_raster_idx, end_raster_idx, fully_connected=True)

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

                        classification_1 = classify_trail_length(gdf.loc[all_line_ids[i], 'length'], threshold_length)
                        classification_2 = classify_trail_length(gdf.loc[all_line_ids[nearest_idx], 'length'], threshold_length)
                        if classification_1 == 'long_trail' and classification_2 == 'long_trail':
                            connection_types.append('long-long')
                        else:
                            connection_types.append('short-long')

                        connections += 1
                except ValueError as e:
                    print(f"Error while processing connection: {e}")
                    continue

    print('Processing long trails')
    process_trails(long_trail_indices, check_points=long_trail_check_points, max_connections=long_trail_max_connections)
    print('Processing short trails')
    process_trails(range(len(all_coords)), check_points=short_trail_check_points, max_connections=short_trail_max_connections)

    connections_gdf = gpd.GeoDataFrame({
        'geometry': closest_connections,
        'median_cost': median_costs,
        'step_cost': step_costs,
        'length': lengths,
        'connection_type': connection_types
    }, crs=gdf.crs)

    combined_gdf = pd.concat([gdf, connections_gdf], ignore_index=True)
    print('Merged')

    points_gdf = gpd.GeoDataFrame({
        'geometry': all_points,
        'line_id': all_line_ids,
        'classification': all_classifications
    }, crs=gdf.crs)

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

        gdf = gpd.read_file(centerline_path)
        print('Starting connect_segment')

        simplified_gdf = connect_segments(gdf, raster_file, connection_threshold=threshold_distance,
                                          tolerance=5, high_cost_factor=1.5,
                                          low_prob_threshold=3, long_trail_check_points=20,
                                          long_trail_max_connections=5,
                                          short_trail_check_points=5, short_trail_max_connections=1)

        print('Filtering out short lines')
        simplified_gdf = simplified_gdf[simplified_gdf.geometry.length >= 1]

        print('Saving connections')
        output_gpkg = os.path.join(output_folder, f"{base_name}_connected.gpkg")
        simplified_gdf.to_file(output_gpkg, driver='GPKG')
        print(f"Saved simplified output as GeoPackage to {output_gpkg}")
    else:
        print(f"Raster file not found for {centerline_file}")

def process_files_in_folder(centerline_folder, raster_folder, output_folder, threshold_distance=200, max_workers=4):
    os.makedirs(output_folder, exist_ok=True)

    centerline_files = [f for f in os.listdir(centerline_folder) if f.endswith('.gpkg')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, centerline_file, centerline_folder, raster_folder, output_folder,
                            threshold_distance)
            for centerline_file in centerline_files
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")

def main():
    centerline_folder = '/media/irro/All/HumanFootprint/DATA/intermediate/connect'
    raster_folder = '/media/irro/All/HumanFootprint/DATA/Products/Unet/Kirby/DTM10cm'
    output_folder = '/media/irro/All/HumanFootprint/DATA/intermediate/connect/connected_segment'

    process_files_in_folder(centerline_folder, raster_folder, output_folder, threshold_distance=200, max_workers=1)

if __name__ == "__main__":
    main()
