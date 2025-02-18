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
import concurrent.futures
from skimage.measure import block_reduce
from rasterio.enums import Resampling


def pad_cost_map(cost_map_np, pad_width=5, pad_value=1000):
    """
    Pad the cost map with a border of pad_width cells, each having the value pad_value.
    """
    return np.pad(cost_map_np, pad_width, mode='constant', constant_values=pad_value)


# --- Alternative Downsampling with Rasterio ---
def rasterio_downsample(src, scale=5):
    """
    Downsample a raster using Rasterio's built-in resampling (e.g. average).

    Parameters:
      src : rasterio.io.DatasetReader
          The opened rasterio dataset.
      scale : int
          The factor by which to downsample (e.g., 5 means 5x5 pixels become one).

    Returns:
      data : np.ndarray
          The downsampled raster data (as a NumPy array).
      new_transform : affine.Affine
          The adjusted affine transform.
    """
    new_height = src.height // scale
    new_width = src.width // scale
    data = src.read(
        out_shape=(src.count, new_height, new_width),
        resampling=Resampling.average
    )
    # Calculate the new transform (each pixel now covers a larger area)
    new_transform = src.transform * src.transform.scale(
        (src.width / new_width),
        (src.height / new_height)
    )
    return data, new_transform


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
            trail_arr = np.array(trail)
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


def aggregate_block_10th_percentile(block, **kwargs):
    # Compute the 10th percentile of the block values
    return np.percentile(block, 10)


def coarse_raster_aggregation(cost_map_np, block_size=(5, 5)):
    """
    Coarsens the raster by aggregating each block_size pixels using the 10th percentile.

    Parameters:
      cost_map_np : np.ndarray or cupy.ndarray
          The high-resolution cost map.
      block_size : tuple
          The block size for aggregation (default is (5, 5)).

    Returns:
      np.ndarray
          The aggregated, coarser cost map.
    """
    # If the input is a CuPy array, convert it explicitly to NumPy.
    if hasattr(cost_map_np, "get"):
        cost_map_np = cost_map_np.get()
    return block_reduce(cost_map_np, block_size=block_size, func=aggregate_block_10th_percentile)




# Move the lambda out as a global function
def compute_route_from_task(task):
    """
    task is a tuple of length 6, but only the first 4 elements are used
    for route computation.
    """
    return compute_route(task[:-2])

def compute_bearing(pt1, pt2):
    """
    Compute the bearing in degrees from pt1 to pt2.
    pt1 and pt2 are (x, y) tuples.
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    angle = math.degrees(math.atan2(dy, dx))
    # Normalize angle to 0-360 degrees
    return (angle + 360) % 360


def compute_route(args):
    """
    Compute the route between two raster indices.
    Expects args as a tuple:
      (cost_map_np, transform, start_raster_idx, end_raster_idx, src_fid, tgt_fid, direction)
    """
    cost_map_np, transform, start_raster_idx, end_raster_idx, src_fid, tgt_fid, direction = args
    try:
        indices_path, _ = route_through_array(cost_map_np, start_raster_idx, end_raster_idx, fully_connected=True)
    except ValueError:
        return None

    path_costs = [cost_map_np[row, col] for row, col in indices_path]
    median_cost = np.median(np.array(path_costs))
    step_cost = np.mean(np.array(path_costs))
    path_coords = [transform * (col, row) for row, col in indices_path]
    if len(path_coords) <= 1:
        return None

    return (LineString(path_coords), median_cost, step_cost, LineString(path_coords).length, direction)


def compute_route_from_task(task):
    return compute_route(task)



def process_trails_parallel(indices, check_points, max_connections, all_coords, raster_indices, line_ids,
                            gdf, cost_map, transform, threshold_length, neighbor_radius=20, verbose=False):
    """
    Processes and connects trail endpoints using precomputed NumPy arrays and cached raster indices.
    Route computations are executed in parallel.
    """
    if verbose:
        print(f"🔍 Entering parallel process_trails with {len(indices)} indices")
    connections = []
    connection_data = []

    tree = cKDTree(all_coords)
    cost_map_np = cp.asnumpy(cost_map)  # Convert cost map to NumPy once

    tasks = []  # Store tasks for parallel route computation

    for i in indices:
        if i >= len(all_coords):
            if verbose:
                print(f"⚠️ Invalid index: {i} (max: {len(all_coords)})")
            continue

        neighbors = tree.query_ball_point(all_coords[i], r=neighbor_radius)
        neighbors = [n for n in neighbors if n != i]
        if not neighbors:
            if verbose:
                print(f"⚠️ No neighbors within {neighbor_radius}m for point {i}")
            continue

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

            start_raster_idx = raster_indices[i]
            end_raster_idx = raster_indices[nearest_idx]
            if verbose:
                print(f"🔍 Queueing route from {start_raster_idx} to {end_raster_idx}")
            # Append a task tuple: (cost_map_np, transform, start_raster_idx, end_raster_idx, i, nearest_idx)
            tasks.append((cost_map_np, transform, start_raster_idx, end_raster_idx, i, nearest_idx))
            connections_count += 1

    BATCH_SIZE = 100
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[i:i + BATCH_SIZE]
            for result in executor.map(compute_route_from_task, batch, chunksize=20):
                results.append(result)

    # Process the results and add valid connections
    for task, result in zip(tasks, results):
        i, nearest_idx = task[-2], task[-1]
        if result is None:
            continue
        route_line, median_cost, step_cost, length = result
        connections.append(route_line)
        connection_data.append({
            'geometry': route_line,
            'median_cost': median_cost,
            'step_cost': step_cost,
            'length': length,
            'connection_type': (
                'long-long' if (classify_trail_length(gdf.loc[line_ids[i], 'length'], threshold_length) == 'long_trail'
                                and classify_trail_length(gdf.loc[line_ids[nearest_idx], 'length'],
                                                          threshold_length) == 'long_trail')
                else 'short-short' if (
                            classify_trail_length(gdf.loc[line_ids[i], 'length'], threshold_length) == 'short_trail'
                            and classify_trail_length(gdf.loc[line_ids[nearest_idx], 'length'],
                                                      threshold_length) == 'short_trail')
                else 'short-long'
            )
        })
    if verbose:
        print(f"✅ Parallel process_trails completed with {len(connections)} connections.")
    return connections, connection_data


def extract_and_validate_endpoints(gdf, transform, raster_shape, verbose=False):
    """
    Extract endpoints (start and end) from each LineString in the GeoDataFrame and validate
    that their transformed raster indices fall within the raster_shape.
    Returns a tuple of:
      - valid_points: list of shapely Points that are valid.
      - valid_coords: corresponding (x, y) tuples.
      - valid_raster_indices: list of (row, col) indices.
    """
    valid_points = []
    valid_coords = []
    valid_raster_indices = []
    for idx, row in gdf.iterrows():
        line = row.geometry
        if line.is_empty or line.geom_type != 'LineString':
            continue
        for pt in [Point(line.coords[0]), Point(line.coords[-1])]:
            # Get spatial coordinates
            x, y = pt.x, pt.y
            # Transform coordinates to raster indices using rasterio (without safe_transform debugging)
            try:
                row_idx, col_idx = rasterio.transform.rowcol(transform, x, y)
            except Exception as e:
                if verbose:
                    print(f"❌ Transformation failed for point ({x:.2f}, {y:.2f}): {e}")
                continue
            # Check that the indices fall within the raster shape
            if 0 <= row_idx < raster_shape[0] and 0 <= col_idx < raster_shape[1]:
                valid_points.append(pt)
                valid_coords.append((x, y))
                valid_raster_indices.append((row_idx, col_idx))
            else:
                if verbose:
                    print(
                        f"⚠️ Point ({x:.2f}, {y:.2f}) transformed to (row={row_idx}, col={col_idx}) is out-of-bounds.")
    return valid_points, valid_coords, valid_raster_indices


def process_all_endpoint_connections(valid_coords, valid_raster_indices, cost_map, transform, verbose=False):
    """
    For each endpoint in valid_coords, find its closest other endpoint (using KD-tree)
    and queue up a least-cost path task.
    """
    tree = cKDTree(valid_coords)
    tasks = []
    n_points = len(valid_coords)
    for i in range(n_points):
        # Query k=2 nearest neighbors: first is the point itself.
        dists, indices = tree.query(valid_coords[i], k=2)
        if len(indices) < 2:
            if verbose:
                print(f"⚠️ Endpoint {i} has no neighbor")
            continue
        j = indices[1]  # closest neighbor (excluding itself)
        if verbose:
            print(f"🔍 Endpoint {i} closest neighbor is {j} at distance {dists[1]:.2f}")
        # Prepare task: ensure cost_map is a NumPy array
        cost_map_np = cost_map.get() if hasattr(cost_map, "get") else cost_map
        tasks.append((cost_map_np, transform, valid_raster_indices[i], valid_raster_indices[j], i, j))
    return tasks

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
import concurrent.futures
from skimage.measure import block_reduce
from rasterio.enums import Resampling
import math


def pad_cost_map(cost_map_np, pad_width=5, pad_value=1000):
    """
    Pad the cost map with a border of pad_width cells, each having the value pad_value.
    """
    return np.pad(cost_map_np, pad_width, mode='constant', constant_values=pad_value)


# --- Alternative Downsampling with Rasterio ---
def rasterio_downsample(src, scale=5):
    """
    Downsample a raster using Rasterio's built-in resampling (e.g. average).

    Parameters:
      src : rasterio.io.DatasetReader
          The opened rasterio dataset.
      scale : int
          The factor by which to downsample (e.g., 5 means 5x5 pixels become one).

    Returns:
      data : np.ndarray
          The downsampled raster data (as a NumPy array).
      new_transform : affine.Affine
          The adjusted affine transform.
    """
    new_height = src.height // scale
    new_width = src.width // scale
    data = src.read(
        out_shape=(src.count, new_height, new_width),
        resampling=Resampling.average
    )
    # Calculate the new transform (each pixel now covers a larger area)
    new_transform = src.transform * src.transform.scale(
        (src.width / new_width),
        (src.height / new_height)
    )
    return data, new_transform


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
            trail_arr = np.array(trail)
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


def aggregate_block_10th_percentile(block, **kwargs):
    # Compute the 10th percentile of the block values
    return np.percentile(block, 10)


def coarse_raster_aggregation(cost_map_np, block_size=(5, 5)):
    """
    Coarsens the raster by aggregating each block_size pixels using the 10th percentile.

    Parameters:
      cost_map_np : np.ndarray or cupy.ndarray
          The high-resolution cost map.
      block_size : tuple
          The block size for aggregation (default is (5, 5)).

    Returns:
      np.ndarray
          The aggregated, coarser cost map.
    """
    # If the input is a CuPy array, convert it explicitly to NumPy.
    if hasattr(cost_map_np, "get"):
        cost_map_np = cost_map_np.get()
    return block_reduce(cost_map_np, block_size=block_size, func=aggregate_block_10th_percentile)


def compute_route_from_task(task):
    return compute_route(task)



def process_all_endpoint_connections(valid_coords, valid_raster_indices, valid_line_ids, verbose=False):
    """
    For each valid endpoint in valid_coords, find the three closest endpoints (using KD-tree)
    that are NOT from the same trail. For each candidate, compute the general direction (bearing)
    from the source endpoint to the candidate endpoint and queue up a least-cost path task.
    Returns a list of tasks.
    """
    tree = cKDTree(valid_coords)
    tasks = []
    n_points = len(valid_coords)
    for i in range(n_points):
        # Query more neighbors than needed to allow filtering; k = min(n_points, 10)
        k = min(n_points, 10)
        dists, indices = tree.query(valid_coords[i], k=k)
        # Filter out self and endpoints from the same trail.
        candidates = [idx for idx in indices if idx != i and valid_line_ids[idx] != valid_line_ids[i]]
        # Take up to 3 candidates
        selected = candidates[:3]
        if verbose:
            print(f"Endpoint {i} (line {valid_line_ids[i]}) candidates: {selected}")
        for j in selected:
            # Compute general direction (bearing) from endpoint i to candidate j.
            direction = compute_bearing(valid_coords[i], valid_coords[j])
            # Prepare task: ensure cost_map is a NumPy array when processing.
            tasks.append((None, None, valid_raster_indices[i], valid_raster_indices[j], i, j, direction))
            # Note: cost_map and transform will be filled in later in connect_segments.
    return tasks


def connect_segments(
        gdf, raster_path,
        endpoints_output_path='/home/irina/HumanFootprint/DATA/Test_Models/segformer/connected_segment/endpoints.gpkg',
        connection_threshold=200, high_cost_factor=2, low_prob_threshold=0.05,
        verbose=False
):
    """
    Connects trail segments based on a probability raster using GPU acceleration.
    For every validated endpoint (start or end of each trail), finds its three closest endpoints
    (only connecting endpoints that exist within the raster and are not from the same trail)
    and computes the least cost path between them. The general direction (bearing) between endpoints
    is also computed.

    IMPORTANT: The reported fid for each endpoint is adjusted by +1 compared to the extracted value.
    """
    if verbose:
        print("🚀 Starting connect_segments()")
    # Merge connected trails
    gdf = merge_connected_lines(gdf, verbose)
    if verbose:
        print(f"✅ Initial trails connected. {len(gdf)} trails remaining.")

    # Load probability map and transform using Rasterio's resampling downsampling method.
    with rasterio.open(raster_path) as src:
        data, new_transform = rasterio_downsample(src, scale=5)
        prob_map = cp.asarray(data[0])  # use the first band
        transform = new_transform

    # Report probability map percentiles:
    p25 = cp.percentile(prob_map, 25)
    p50 = cp.percentile(prob_map, 50)
    p75 = cp.percentile(prob_map, 75)
    print(f"📊 Probability Map Percentiles: 25th: {p25:.4f}, 50th: {p50:.4f}, 75th: {p75:.4f}")

    if verbose:
        print(
            f"📊 Probability Map Loaded: Shape={prob_map.shape}, Min={cp.nanmin(prob_map):.4f}, Max={cp.nanmax(prob_map):.4f}")

    # Create cost map based on the probability map.
    cost_map = cp.where(
        (prob_map > -0.2) & (prob_map <= 0),
        0.1,
        cp.where(
            (prob_map >= 0.3) | (prob_map <= -0.3),
            10,
            1 + cp.abs(prob_map)
        )
    )

    # Optional: Plot maps for debugging.
    plot_maps(prob_map, cost_map, gdf, transform, start_x=100, start_y=100, window_size=300, verbose=False)

    if verbose:
        print(f"📊 Cost Map Created: Min={cp.nanmin(cost_map):.4f}, Max={cp.nanmax(cost_map):.4f}")

    # Extract and validate endpoints from trails.
    # For each endpoint, record its geometry, spatial coordinate, transformed raster index,
    # and the originating trail's fid (adjusted: idx+1).
    valid_points = []
    valid_coords = []
    valid_raster_indices = []
    valid_line_ids = []
    for idx, row in gdf.iterrows():
        line = row.geometry
        if line.is_empty or line.geom_type != 'LineString':
            continue
        for pt in [Point(line.coords[0]), Point(line.coords[-1])]:
            x, y = pt.x, pt.y
            try:
                row_idx, col_idx = rasterio.transform.rowcol(transform, x, y)
            except Exception as e:
                if verbose:
                    print(f"❌ Transformation failed for point ({x:.2f}, {y:.2f}): {e}")
                continue
            if 0 <= row_idx < prob_map.shape[0] and 0 <= col_idx < prob_map.shape[1]:
                valid_points.append(pt)
                valid_coords.append((x, y))
                valid_raster_indices.append((row_idx, col_idx))
                # Adjust fid by adding 1 so that if the gdf index is 714, reported fid becomes 715.
                valid_line_ids.append(idx + 1)
            else:
                if verbose:
                    print(
                        f"⚠️ Point ({x:.2f}, {y:.2f}) transformed to (row={row_idx}, col={col_idx}) is out-of-bounds.")

    n_points = len(valid_points)
    if verbose:
        print(f"📍 {n_points} valid endpoints extracted from trails.")

    # Compute candidate endpoints.
    # Build a KD-tree and, for each endpoint, find the three closest endpoints (excluding endpoints from the same trail).
    tree = cKDTree(valid_coords)
    candidate_index_dict = {}  # endpoint index -> list of candidate indices (within valid endpoints)
    candidate_fid_dict = {}  # endpoint index -> list of candidate fids (adjusted)
    for i in range(n_points):
        k = min(n_points, 10)
        dists, indices = tree.query(valid_coords[i], k=k)
        # Exclude self and endpoints from the same trail.
        candidates = [idx for idx in indices if idx != i and valid_line_ids[idx] != valid_line_ids[i]]
        candidate_index_dict[i] = candidates[:3]
        candidate_fid_dict[i] = [valid_line_ids[j] for j in candidates[:3]]

    # Save endpoints (with candidate attributes) to a file.
    if endpoints_output_path:
        endpoints_gdf = gpd.GeoDataFrame({
            'geometry': valid_points,
            'line_id': valid_line_ids,
            'closest_ids': [str(candidate_fid_dict.get(i, [])) for i in range(n_points)]
        }, crs=gdf.crs)
        endpoints_gdf.to_file(endpoints_output_path, driver='GPKG')
        print(f"💾 Endpoints saved to {endpoints_output_path}")

    # Build connection tasks from each endpoint to all 3 of its closest endpoints.
    tasks = []
    for i in range(n_points):
        for j in candidate_index_dict.get(i, []):
            direction = compute_bearing(valid_coords[i], valid_coords[j])
            # Task tuple format:
            # (None, None, start_raster_idx, end_raster_idx, src_fid, tgt_fid, direction)
            tasks.append((None, None, valid_raster_indices[i], valid_raster_indices[j], valid_line_ids[i],
                          valid_line_ids[j], direction))
    if verbose:
        print(f"🔹 {len(tasks)} route tasks queued for processing.")

    # Fill in cost_map and transform in each task.
    cost_map_np = cost_map.get() if hasattr(cost_map, "get") else cost_map
    final_tasks = [(cost_map_np, transform) + task[2:] for task in tasks]

    # Execute tasks in parallel.
    BATCH_SIZE = 100
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(0, len(final_tasks), BATCH_SIZE):
            batch = final_tasks[i:i + BATCH_SIZE]
            for result in executor.map(compute_route_from_task, batch, chunksize=20):
                results.append(result)

    # Process the results and collect valid connections.
    connections = []
    connection_data = []
    for task, result in zip(final_tasks, results):
        src_fid, tgt_fid = task[-3], task[-2]
        direction = task[-1]
        if result is None:
            if verbose:
                print(f"⚠️ No valid path between endpoints {src_fid} and {tgt_fid}")
            continue
        route_line, median_cost, step_cost, length, _ = result
        connections.append(route_line)
        connection_data.append({
            'geometry': route_line,
            'median_cost': median_cost,
            'step_cost': step_cost,
            'length': length,
            'direction': direction,
            'connection_type': 'endpoint_connection',
            'src_endpoint': src_fid,
            'tgt_endpoint': tgt_fid
        })

    # Convert connection data to a GeoDataFrame.
    connections_gdf = gpd.GeoDataFrame(
        {
            'geometry': [data['geometry'] for data in connection_data],
            'median_cost': [data['median_cost'] for data in connection_data],
            'step_cost': [data['step_cost'] for data in connection_data],
            'length': [data['length'] for data in connection_data],
            'direction': [data['direction'] for data in connection_data],
            'connection_type': [data['connection_type'] for data in connection_data],
            'src_endpoint': [data['src_endpoint'] for data in connection_data],
            'tgt_endpoint': [data['tgt_endpoint'] for data in connection_data],
        },
        crs=gdf.crs
    )
    if verbose:
        print(f"✅ {len(connections_gdf)} connections added.")

    # Merge connections with original trails.
    gdf['connection_type'] = 'trail'
    combined_gdf = pd.concat([gdf, connections_gdf], ignore_index=True)
    if verbose:
        print("✅ GPU-based connect_segments() completed!")
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
        print("⚡ Running GPU-accelerated connect_segments()...")
        simplified_gdf = connect_segments(gdf, raster_file, connection_threshold=threshold_distance, verbose=verbose)
        print("🔹 Filtering out short trails...")
        simplified_gdf = simplified_gdf[simplified_gdf.geometry.length >= 1]
        output_gpkg = os.path.join(output_folder, f"{base_name}_connected_25_4.gpkg")
        simplified_gdf.to_file(output_gpkg, driver='GPKG')
        print(f"✅ Saved to {output_gpkg}")
    else:
        print(f"⚠️ Raster file not found for {centerline_path}")


def process_files_in_folder(centerline_folder, raster_folder, output_folder, threshold_distance=200, verbose=False):
    """
    Process files sequentially. Files are processed one-by-one.
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
        # For debugging purposes, hardcoded paths can be removed later.
        centerline_path = '/home/irina/HumanFootprint/DATA/Test_Models/segformer/centerline/test4_preds_segformer.gpkg'
        raster_path = "/media/irina/My Book1/Conoco/DATA/Drone_PPC_2024/ndtm/test4.tif"
        process_file(centerline_path, raster_path, output_folder, threshold_distance, verbose=verbose)
        break


### ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    centerline_folder = "/home/irina/HumanFootprint/DATA/Test_Models/segformer/centerline"
    raster_folder = "/home/irina/HumanFootprint/DATA/Test_Models/segformer"
    output_folder = "/home/irina/HumanFootprint/DATA/Test_Models/segformer/connected_segment"
    process_files_in_folder(centerline_folder, raster_folder, output_folder, threshold_distance=200, verbose=True)
