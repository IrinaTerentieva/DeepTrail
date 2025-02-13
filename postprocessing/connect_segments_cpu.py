import os
import cupy as cp
import cuspatial
import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import pandas as pd
from skimage.graph import route_through_array
from scipy.spatial import cKDTree


### ========== COORDINATE TRANSFORM ==========
def safe_transform(transform, x, y, raster_shape, verbose=False):
    """Transform coordinates to raster indices safely with optional debugging."""
    try:
        row, col = rasterio.transform.rowcol(transform, x, y)
        if verbose:
            print(f"🔢 Transformed (x={x:.2f}, y={y:.2f}) → (row={row}, col={col})")
        # Check boundaries
        if row < 0 or row >= raster_shape[0] or col < 0 or col >= raster_shape[1]:
            if verbose:
                print(f"⚠️ Out-of-bounds: (row={row}, col={col}), Raster shape: {raster_shape}")
            return None
        return (row, col)
    except Exception as e:
        if verbose:
            print(f"❌ Coordinate transform failed for ({x:.2f}, {y:.2f}): {e}")
        return None


def transform_trail_to_raster(gdf, transform, raster_shape, raster_crs, verbose=False):
    """Convert trail (GeoDataFrame) coordinates to raster row/col indices with debugging."""
    if verbose:
        print("🔍 Entering transform_trail_to_raster")
        print(f"🗺️ Raster CRS: {raster_crs}")
        print(f"📐 Raster shape: {raster_shape}")
        print(f"🌐 GeoDataFrame CRS: {gdf.crs}")

    if gdf.crs != raster_crs:
        if verbose:
            print("⚠️ CRS mismatch detected! Reprojecting GeoDataFrame...")
        gdf = gdf.to_crs(raster_crs)
        if verbose:
            print(f"✅ Reprojection complete. New CRS: {gdf.crs}")

    raster_trails = []
    for line_idx, line in enumerate(gdf.geometry):
        if line.geom_type == "LineString":
            trail_pixels = []
            if verbose:
                print(f"🛤️ Processing line {line_idx + 1}/{len(gdf)} with {len(line.coords)} points")
            for point_idx, (x, y) in enumerate(line.coords):
                if verbose:
                    print(f"📍 Point {point_idx}: (x={x:.2f}, y={y:.2f}) - Transforming...")
                result = safe_transform(transform, x, y, raster_shape, verbose)
                if result is not None:
                    trail_pixels.append(result)
                else:
                    if verbose:
                        print(f"⚠️ Skipping point ({x}, {y}) due to transform error.")
            if trail_pixels:
                raster_trails.append(trail_pixels)
            else:
                if verbose:
                    print(f"⚠️ No valid points found for line {line_idx}")
    if verbose:
        print(f"🔍 Completed transform_trail_to_raster. Trails count: {len(raster_trails)}")
    return raster_trails


### ========== PLOTTING ==========
def plot_maps(probability_map, cost_map, gdf, transform, start_x=0, start_y=0, window_size=500, verbose=False):
    """Plots the probability and cost maps with the trail network in raster space."""
    print(
        f"\n🔍 GPU Probability Map: Min={cp.nanmin(probability_map):.4f}, Mean={cp.nanmean(probability_map):.4f}, Max={cp.nanmax(probability_map):.4f}")
    print(
        f"🔍 GPU Cost Map: Min={cp.nanmin(cost_map):.4f}, Mean={cp.nanmean(cost_map):.4f}, Max={cp.nanmax(cost_map):.4f}")

    # Ensure the window does not exceed the raster size
    x_max = min(start_x + window_size, probability_map.shape[1])
    y_max = min(start_y + window_size, probability_map.shape[0])
    start_x = max(0, min(start_x, probability_map.shape[1] - window_size))
    start_y = max(0, min(start_y, probability_map.shape[0] - window_size))

    # Convert trails to raster coordinates
    raster_trails = transform_trail_to_raster(gdf, transform, probability_map.shape, gdf.crs, verbose)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Probability Map Plot
    ax1 = axes[0]
    im1 = ax1.imshow(cp.asnumpy(probability_map[start_y:y_max, start_x:x_max]), cmap="Blues", origin="upper")
    ax1.set_title("Probability Map")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    # Cost Map Plot
    ax2 = axes[1]
    im2 = ax2.imshow(cp.asnumpy(cost_map[start_y:y_max, start_x:x_max]), cmap="Reds", origin="upper")
    ax2.set_title("Cost Map")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    # Overlay trails
    for trail in raster_trails:
        if len(trail) > 1:
            trail_arr = np.array(trail)  # Convert list of tuples to array
            row, col = trail_arr[:, 0], trail_arr[:, 1]
            mask = (start_x <= col) & (col < x_max) & (start_y <= row) & (row < y_max)
            col, row = col[mask] - start_x, row[mask] - start_y
            if len(col) > 1 and len(row) > 1:
                ax1.plot(col, row, color='yellow', linewidth=1)
                ax2.plot(col, row, color='black', linewidth=1)
    plt.tight_layout()
    plt.show()


### ========== UTILITIES ==========
def merge_connected_lines(gdf, verbose=False):
    """Merge all intersecting or connected trails into single entities."""
    merged_lines = unary_union(gdf.geometry)
    if merged_lines.geom_type == 'LineString':
        return gpd.GeoDataFrame({'geometry': [merged_lines], 'length': [merged_lines.length]}, crs=gdf.crs)
    elif merged_lines.geom_type == 'MultiLineString':
        return gpd.GeoDataFrame({'geometry': list(merged_lines.geoms),
                                 'length': [line.length for line in merged_lines.geoms]}, crs=gdf.crs)
    else:
        raise ValueError(f"Unexpected geometry type: {merged_lines.geom_type}")


def classify_trail_length(length, threshold_length):
    """Classify trail length as 'long' or 'short' based on a threshold."""
    return 'long_trail' if length >= threshold_length else 'short_trail'


### ========== PROCESS TRAILS ==========
def process_trails(indices, check_points, max_connections, all_coords, raster_indices, line_ids,
                   gdf, cost_map, transform, threshold_length, neighbor_radius=20, verbose=False):
    """
    Processes and connects trail endpoints using a precomputed NumPy array for coordinates
    and cached raster indices. Neighbors are found within a specified radius.
    """
    if verbose:
        print(f"🔍 Entering `process_trails` with {len(indices)} indices")
    connections = []
    connection_data = []

    # Build a KDTree using precomputed NumPy coordinates
    tree = cKDTree(all_coords)

    for i in indices:
        if i >= len(all_coords):
            if verbose:
                print(f"⚠️ Invalid index: {i} (max: {len(all_coords)})")
            continue

        # Use only the radius query
        neighbors = tree.query_ball_point(all_coords[i], r=neighbor_radius)
        # Exclude self
        neighbors = [n for n in neighbors if n != i]
        if not neighbors:
            if verbose:
                print(f"⚠️ No neighbors within {neighbor_radius}m for point {i}")
            continue

        # Sort neighbors by Euclidean distance and limit to 'check_points'
        neighbors = sorted(neighbors, key=lambda n: np.linalg.norm(all_coords[i] - all_coords[n]))
        nearest_indices = neighbors[:check_points]
        if verbose:
            print(f"🔹 Point {i}: Found {len(nearest_indices)} neighbors within {neighbor_radius}m")

        connections_count = 0

        # Optionally, prioritize long-trail connections
        long_trail_neighbors = [
            idx for idx in nearest_indices
            if classify_trail_length(gdf.loc[line_ids[idx], 'length'], threshold_length) == 'long_trail'
        ]
        if long_trail_neighbors:
            nearest_indices = long_trail_neighbors

        for nearest_idx in nearest_indices:
            if connections_count >= max_connections:
                if verbose:
                    print(f"🔴 Max connections ({max_connections}) reached for point {i}")
                break

            if nearest_idx >= len(all_coords):
                if verbose:
                    print(f"⚠️ Invalid neighbor index: {nearest_idx}")
                continue

            if line_ids[i] == line_ids[nearest_idx]:
                if verbose:
                    print(f"⚠️ Skipping self-connection for line {line_ids[i]}")
                continue

            try:
                # Use precomputed raster indices directly
                start_raster_idx = raster_indices[i]
                end_raster_idx = raster_indices[nearest_idx]
                if verbose:
                    print(f"🔍 Start: {start_raster_idx}, End: {end_raster_idx}")

                # Check raster bounds
                if not (0 <= start_raster_idx[0] < cost_map.shape[0] and 0 <= start_raster_idx[1] < cost_map.shape[1]):
                    if verbose:
                        print(f"⚠️ Start point out of bounds: {start_raster_idx}")
                    continue
                if not (0 <= end_raster_idx[0] < cost_map.shape[0] and 0 <= end_raster_idx[1] < cost_map.shape[1]):
                    if verbose:
                        print(f"⚠️ End point out of bounds: {end_raster_idx}")
                    continue

                # Compute route using the cost map (converted to NumPy)
                indices_path, _ = route_through_array(cp.asnumpy(cost_map), start_raster_idx, end_raster_idx,
                                                      fully_connected=True)
                path_costs = [cost_map[row, col] for row, col in indices_path]
                median_cost = np.median(cp.asnumpy(cp.array(path_costs)))
                step_cost = np.mean(cp.asnumpy(cp.array(path_costs)))
                path_coords = [transform * (col, row) for row, col in indices_path]

                if verbose:
                    print(f"🛠️ Path costs - Median: {median_cost}, Mean: {step_cost}")

                if len(path_coords) > 1:
                    path_line = LineString(path_coords)
                    connections.append(path_line)
                    connection_data.append({
                        'geometry': path_line,
                        'median_cost': median_cost,
                        'step_cost': step_cost,
                        'length': path_line.length,
                        'connection_type': (
                            'long-long' if classify_trail_length(gdf.loc[line_ids[i], 'length'],
                                                                 threshold_length) == 'long_trail'
                                           and classify_trail_length(gdf.loc[line_ids[nearest_idx], 'length'],
                                                                     threshold_length) == 'long_trail'
                            else 'short-short' if classify_trail_length(gdf.loc[line_ids[i], 'length'],
                                                                        threshold_length) == 'short_trail'
                                                  and classify_trail_length(gdf.loc[line_ids[nearest_idx], 'length'],
                                                                            threshold_length) == 'short_trail'
                            else 'short-long'
                        )
                    })
                    if verbose:
                        print(f"🔗 Added connection between points {i} and {nearest_idx}")
                    connections_count += 1
                else:
                    if verbose:
                        print(f"⚠️ Empty path for points {i} → {nearest_idx}")

            except ValueError as e:
                if verbose:
                    print(f"❌ Error processing connection: {e}")
            except Exception as e:
                if verbose:
                    print(f"🔥 Unexpected Error: {e}")

    if verbose:
        print(f"✅ `process_trails` completed with {len(connections)} connections.")
    return connections, connection_data


### ========== GPU-BASED CONNECTIVITY PROCESS ==========
def connect_segments(
        gdf, raster_path, connection_threshold=200, high_cost_factor=1.5, low_prob_threshold=0.05,
        long_trail_check_points=10, long_trail_max_connections=2, short_trail_check_points=5,
        short_trail_max_connections=1,
        verbose=False
):
    """
    Connects trail segments based on a probability raster using GPU acceleration.
    Implements precomputation of raster indices and streamlined neighbor search.
    """
    if verbose:
        print("🚀 Starting `connect_segments()`")
    # Merge connected trails
    gdf = merge_connected_lines(gdf, verbose)
    if verbose:
        print(f'✅ Initial trails connected. {len(gdf)} trails remaining.')

    # Load the probability map and obtain transform
    with rasterio.open(raster_path) as src:
        probability_map = cp.asarray(src.read(1))  # Move to GPU
        transform = src.transform

    if verbose:
        print(
            f"📊 Probability Map Loaded: Shape={probability_map.shape}, Min={cp.nanmin(probability_map):.4f}, Max={cp.nanmax(probability_map):.4f}")

    # Compute cost map on GPU
    cost_map = cp.asarray(1 - probability_map, dtype=cp.float32)
    cost_map = cp.where(cost_map > 0.5, cost_map * high_cost_factor, cost_map)
    cost_map[probability_map < low_prob_threshold] = 100  # High cost for low probability
    if verbose:
        print(f"📊 Cost Map Created: Min={cp.nanmin(cost_map):.4f}, Max={cp.nanmax(cost_map):.4f}")

    # Extract trail endpoints
    points = []
    line_ids = []
    for idx, row in gdf.iterrows():
        line = row.geometry
        if line.is_empty or line.geom_type != 'LineString':
            continue
        points.extend([Point(line.coords[0]), Point(line.coords[-1])])
        line_ids.extend([idx, idx])

    if not points:
        if verbose:
            print("⚠️ No valid points found! Exiting.")
        return gdf

    # Convert points to a CuPy array (only once)
    points_array = cp.asarray([(p.x, p.y) for p in points])
    if verbose:
        print(f"📍 {len(points_array)} points extracted from trails.")

    # Convert CuPy array to NumPy once for KDTree and precompute raster indices
    all_coords = cp.asnumpy(points_array)
    raster_indices = np.array([rasterio.transform.rowcol(transform, x, y) for x, y in all_coords])
    if verbose:
        print("✅ Precomputed raster indices for endpoints.")

    # Optionally, create a CuSpatial GeoSeries (if needed for other GPU computations)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    interleaved_coords = cp.empty((x_coords.size * 2,), dtype=cp.float64)
    interleaved_coords[0::2] = x_coords
    interleaved_coords[1::2] = y_coords
    points_gs = cuspatial.GeoSeries.from_points_xy(interleaved_coords)
    if verbose:
        print(f"✅ CuSpatial GeoSeries created with {len(points_gs)} points.")

    # (Optional) Compute pairwise distances using cuspatial if needed...
    # For now, we continue with the KDTree-based neighbor search.

    # Process long trails first
    long_trail_indices = [i for i, length in enumerate(gdf['length']) if
                          classify_trail_length(length, 10) == 'long_trail']

    print(f'🔹 Processing {len(long_trail_indices)} long trails...')
    long_connections, long_connection_data = process_trails(
        long_trail_indices, long_trail_check_points, long_trail_max_connections,
        all_coords, raster_indices, line_ids, gdf, cost_map, transform,
        threshold_length=10, neighbor_radius=50, verbose=verbose
    )

    # (Optional) Process short trails here if needed...

    # Convert connection data into a GeoDataFrame
    connections_gdf = gpd.GeoDataFrame(
        {
            'geometry': [data['geometry'] for data in long_connection_data],
            'median_cost': [data['median_cost'] for data in long_connection_data],
            'step_cost': [data['step_cost'] for data in long_connection_data],
            'length': [data['length'] for data in long_connection_data]
        },
        crs=gdf.crs
    )

    print(f'✅ {len(connections_gdf)} connections added.')

    # Merge results with original trails
    gdf['connection_type'] = 'trail'
    combined_gdf = pd.concat([gdf, connections_gdf], ignore_index=True)
    if verbose:
        print("✅ GPU-based `connect_segments()` completed!")
    return combined_gdf


def normalize_filename(filename):
    """Remove common suffixes and make lowercase for better matching."""
    return filename.lower().replace('_preds_segformer', '')


def process_file(centerline_path, raster_file, output_folder, threshold_distance, verbose=False):
    """Process a single file with GPU acceleration."""
    base_name = os.path.splitext(os.path.basename(centerline_path))[0]
    if os.path.exists(raster_file):
        print(f"🚀 Processing: {centerline_path} with {raster_file}")
        gdf = gpd.read_file(centerline_path)
        print('⚡ Running GPU-accelerated `connect_segments`...')
        simplified_gdf = connect_segments(gdf, raster_file, connection_threshold=threshold_distance, verbose=verbose)
        print('🔹 Filtering out short trails...')
        simplified_gdf = simplified_gdf[simplified_gdf.geometry.length >= 1]
        output_gpkg = os.path.join(output_folder, f"{base_name}_connected.gpkg")
        simplified_gdf.to_file(output_gpkg, driver='GPKG')
        print(f"✅ Saved to {output_gpkg}")
    else:
        print(f"⚠️ Raster file not found for {centerline_path}")


def process_files_in_folder(centerline_folder, raster_folder, output_folder, threshold_distance=200, verbose=False):
    """
    Process files sequentially but use multiprocessing inside each file if desired.
    """
    os.makedirs(output_folder, exist_ok=True)
    centerline_files = {
        normalize_filename(os.path.splitext(f)[0]): os.path.join(centerline_folder, f)
        for f in os.listdir(centerline_folder) if f.endswith('.gpkg')
    }
    raster_files = {
        normalize_filename(os.path.splitext(f)[0]): os.path.join(raster_folder, f)
        for f in os.listdir(raster_folder) if f.endswith('.tif')
    }
    matching_files = [(centerline_files[name], raster_files[name]) for name in centerline_files if name in raster_files]
    missing_centerlines = set(raster_files.keys()) - set(centerline_files.keys())
    missing_rasters = set(centerline_files.keys()) - set(raster_files.keys())
    if missing_centerlines:
        print(f"⚠️ Warning: No centerline found for {len(missing_centerlines)} raster files: {missing_centerlines}")
    if missing_rasters:
        print(f"⚠️ Warning: No raster found for {len(missing_rasters)} centerline files: {missing_rasters}")
    for centerline_path, raster_path in matching_files:
        print(f"\n🔄 Processing: {centerline_path} with {raster_path}")
        process_file(centerline_path, raster_path, output_folder, threshold_distance, verbose=verbose)


### ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    centerline_folder = "/home/irina/HumanFootprint/DATA/Test_Models/segformer/centerline"
    raster_folder = "/home/irina/HumanFootprint/DATA/Test_Models/segformer"
    output_folder = "/home/irina/HumanFootprint/DATA/Test_Models/segformer/connected_segment"
    process_files_in_folder(centerline_folder, raster_folder, output_folder, threshold_distance=200, verbose=True)
