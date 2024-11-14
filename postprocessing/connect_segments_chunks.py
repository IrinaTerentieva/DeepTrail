import geopandas as gpd
from shapely.geometry import Point, LineString, box
from scipy.spatial import cKDTree
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.graph import route_through_array
import pandas as pd
import os
import gc
from shapely.ops import linemerge, unary_union


def merge_connected_lines(gdf):
    """Merge intersecting or connected lines from the GeoDataFrame into single entities."""
    merged_lines = unary_union(gdf.geometry)
    merged_geometries, merged_lengths = [], []

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


def classify_trail_length(length, length_median=10):
    """Classify trail length into 'short' or 'long' based on median length."""
    return 'long_trail' if length >= length_median else 'short_trail'


def crop_gdf_to_tile(gdf, tile_bounds):
    """Crop the GeoDataFrame to match the bounds of the current tile."""
    tile_box = box(*tile_bounds)
    cropped_gdf = gdf[gdf.intersects(tile_box)]
    return cropped_gdf


def connect_segments(gdf, probability_map, transform, high_cost_factor=1.5, low_prob_threshold=3):
    """Connect segments within the current tile using the cost map."""
    gdf = merge_connected_lines(gdf)
    print('Initial lines connected')

    points, line_ids, line_probabilities, line_classifications = [], [], [], []

    threshold_length = np.percentile(gdf['length'], 70)
    print(f"Threshold trail length: {threshold_length}")

    for idx, row in gdf.iterrows():
        line = row.geometry
        if line.is_empty or line.geom_type != 'LineString':
            continue
        start_point, end_point = Point(line.coords[0]), Point(line.coords[-1])
        points.extend([start_point, end_point])
        line_ids.extend([idx, idx])

        line_coords = np.array(line.coords)
        line_raster_indices = np.apply_along_axis(lambda coord: transform * (coord[0], coord[1]), 1, line_coords)

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

    cost_map = np.array(100 - probability_map, dtype=np.float32)
    cost_map = np.where(cost_map > 50, cost_map * high_cost_factor, cost_map)
    cost_map[probability_map < low_prob_threshold] = 1000

    all_coords = np.array([(p.x, p.y) for p in points])
    if all_coords.size == 0:
        print("No points in this tile.")
        return None

    tree = cKDTree(all_coords)
    closest_connections, median_costs, step_costs, lengths, connection_types = [], [], [], [], []

    def point_to_raster_indices(point):
        col, row = ~transform * (point.x, point.y)
        return int(row), int(col)

    def process_trails(indices, check_points, max_connections):
        for i in indices:
            distances, nearest_indices = tree.query(all_coords[i], k=min(len(all_coords), check_points + 1))
            nearest_indices = nearest_indices[1:]  # Skip itself

            connections = 0

            for nearest_idx in nearest_indices:
                if connections >= max_connections:
                    break
                # Skip if trying to connect to the same line
                if line_ids[i] == line_ids[nearest_idx]:
                    continue

                try:
                    start_raster_idx = point_to_raster_indices(Point(all_coords[i]))
                    end_raster_idx = point_to_raster_indices(Point(all_coords[nearest_idx]))

                    # Skip if indices are invalid (e.g., out of bounds)
                    if not (0 <= start_raster_idx[0] < cost_map.shape[0] and 0 <= start_raster_idx[1] < cost_map.shape[
                        1]):
                        print(f"Start point {start_raster_idx} out of bounds.")
                        continue
                    if not (0 <= end_raster_idx[0] < cost_map.shape[0] and 0 <= end_raster_idx[1] < cost_map.shape[1]):
                        print(f"End point {end_raster_idx} out of bounds.")
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
                        print(f"Connection added in tile between points {i} and {nearest_idx}")
                except ValueError as e:
                    print(f"Error while processing connection: {e}")
                    continue

    print('Process long trails')
    long_trail_indices = [i for i, length in enumerate(gdf['length']) if
                          classify_trail_length(length, threshold_length) == 'long_trail']
    process_trails(long_trail_indices, check_points=20, max_connections=5)

    print('Process short trails')
    process_trails(range(len(all_coords)), check_points=5, max_connections=1)

    connections_gdf = gpd.GeoDataFrame({
        'geometry': closest_connections,
        'median_cost': median_costs,
        'step_cost': step_costs,
        'length': lengths,
        'connection_type': connection_types
    }, crs=gdf.crs)

    print(f"Tile connected with {len(closest_connections)} connections.")
    return connections_gdf


def process_in_tiles(raster_path, gdf_path, output_folder, tile_size=1000):
    gdf = gpd.read_file(gdf_path)

    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        transform = src.transform

        tile_index = 1
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                window = Window(j, i, tile_size, tile_size)
                tile_bounds = rasterio.windows.bounds(window, transform)

                # Load tile raster data
                probability_map = src.read(1, window=window)

                # Reset the transform for the current tile
                tile_transform = src.window_transform(window)

                tile_gdf = crop_gdf_to_tile(gdf, tile_bounds)

                if tile_gdf.empty:
                    print(f"Tile {tile_index} has no data in bounds.")
                    tile_index += 1
                    continue

                # Resetting memory and variables before each tile processing
                gc.collect()

                print(f"Processing tile {tile_index}")
                connected_segments = connect_segments(tile_gdf, probability_map, tile_transform)

                if connected_segments is not None and not connected_segments.empty:
                    output_path = os.path.join(output_folder, f"tile_{tile_index}_connected.gpkg")
                    connected_segments.to_file(output_path, driver='GPKG')
                    print(f"Saved connected segments for tile {tile_index}")
                else:
                    print(f"No connections were made for tile {tile_index}")

                tile_index += 1


def main():
    raster_path = '/media/irro/All/HumanFootprint/DATA/connect/DTM_10cm_binning_488_6131_496_6137_CNN9ep_512_358max.tif'
    gdf_path = '/media/irro/All/HumanFootprint/DATA/connect/DTM_10cm_binning_488_6131_496_6137_CNN9ep_512_358max.gpkg'
    output_folder = '/media/irro/All/HumanFootprint/DATA/connect/connected_segment'

    process_in_tiles(raster_path, gdf_path, output_folder, tile_size=2000)


if __name__ == "__main__":
    main()
