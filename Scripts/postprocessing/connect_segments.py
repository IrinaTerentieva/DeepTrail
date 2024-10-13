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
    gdf['buffer'] = gdf.geometry.buffer(tolerance)
    gdf['representative'] = gdf['buffer'].apply(lambda x: unary_union([x]))
    simplified_gdf = gdf.drop_duplicates(subset='representative')
    return simplified_gdf.drop(columns=['buffer', 'representative'])

def connect_segments(gdf, raster_path, plot_endpoints=True, plot_connections=True,
                     connection_threshold=200, max_connections=3, tolerance=5, high_cost_factor=1.5,
                     low_prob_threshold=3):
    with rasterio.open(raster_path) as src:
        probability_map = src.read(1)
        transform = src.transform

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

def process_file(centerline_file, centerline_folder, raster_folder, output_folder, threshold_distance):
    base_name = os.path.splitext(centerline_file)[0]
    raster_file = os.path.join(raster_folder, f"{base_name}.tif")
    centerline_path = os.path.join(centerline_folder, centerline_file)

    if os.path.exists(raster_file):
        print(f"Processing: {centerline_path} with {raster_file}")

        # Load centerline GeoPackage
        gdf = gpd.read_file(centerline_path)
        gdf['length'] = gdf.geometry.length

        # Filter out lines shorter than a certain length
        gdf = gdf[gdf.geometry.length >= 1]

        # Connect the segments using the probability raster
        simplified_gdf = connect_segments(gdf, raster_file, plot_endpoints=False, plot_connections=False,
                                          connection_threshold=threshold_distance, high_cost_factor=1.5,
                                          low_prob_threshold=3)

        # Filter out lines shorter than a certain length
        simplified_gdf = simplified_gdf[simplified_gdf.geometry.length >= 1]

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
            executor.submit(process_file, centerline_file, centerline_folder, raster_folder, output_folder, threshold_distance)
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
    centerline_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm/centerline'
    raster_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm'
    output_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm/connected_segment'

    # Process files in parallel with 4 workers
    process_files_in_folder(centerline_folder, raster_folder, output_folder, threshold_distance=200, max_workers=4)

if __name__ == "__main__":
    main()
