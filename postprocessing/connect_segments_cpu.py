import os
import cupy as cp
import cuspatial
import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from skimage.graph import route_through_array
from scipy.spatial import cKDTree
from skimage.measure import block_reduce
from rasterio.enums import Resampling
import math
from queue import PriorityQueue
import pandas as pd
import rasterio.warp

def pad_cost_map(cost_map_np, pad_width=5, pad_value=1000):
    return np.pad(cost_map_np, pad_width, mode='constant', constant_values=pad_value)

def update_duplicate_point_ids(point_ids):
    seen = {}
    updated_ids = []
    for pid in point_ids:
        if pid in seen:
            new_pid = pid + 5000
            while new_pid in seen:
                new_pid += 5000
            updated_ids.append(new_pid)
            seen[new_pid] = True
        else:
            updated_ids.append(pid)
            seen[pid] = True
    return updated_ids

def rasterio_downsample(src, scale=5):
    new_height = src.height // scale
    new_width = src.width // scale
    data = src.read(out_shape=(src.count, new_height, new_width), resampling=Resampling.average)
    new_transform = src.transform * src.transform.scale((src.width / new_width), (src.height / new_height))
    return data, new_transform

def align_boundaries(vector_gdf, raster_src):
    vector_bounds = vector_gdf.total_bounds
    raster_bounds = raster_src.bounds
    min_x = min(vector_bounds[0], raster_bounds.left)
    min_y = min(vector_bounds[1], raster_bounds.bottom)
    max_x = max(vector_bounds[2], raster_bounds.right)
    max_y = max(vector_bounds[3], raster_bounds.top)
    vector_gdf = vector_gdf.cx[min_x:max_x, min_y:max_y]
    return vector_gdf

def safe_transform(transform, x, y, raster_shape, verbose=False):
    try:
        row, col = rasterio.transform.rowcol(transform, x, y)
        if verbose:
            print(f"🔍 Original (x={x}, y={y}) → Raster (row={row}, col={col})")
        if 0 <= row < raster_shape[0] and 0 <= col < raster_shape[1]:
            return row, col
        else:
            if verbose:
                print("⚠️ Out-of-bounds point detected!")
            return None
    except Exception as e:
        if verbose:
            print(f"❌ Failed to transform ({x}, {y}): {e}")
        return None

def transform_trail_to_raster(gdf, transform, raster_shape, raster_crs, verbose=False):
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    raster_trails = []
    for line in gdf.geometry:
        if line and line.geom_type == "LineString":
            trail_pixels = []
            for (x, y) in line.coords:
                result = safe_transform(transform, x, y, raster_shape, verbose)
                if result is not None:
                    trail_pixels.append(result)
            if trail_pixels:
                raster_trails.append(trail_pixels)
    return raster_trails

def plot_maps(probability_map, cost_map, gdf, transform, start_x=0, start_y=0, window_size=500):
    x_max = min(start_x + window_size, probability_map.shape[1])
    y_max = min(start_y + window_size, probability_map.shape[0])
    raster_trails = transform_trail_to_raster(gdf, transform, probability_map.shape, gdf.crs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axes[0].imshow(cp.asnumpy(probability_map[start_y:y_max, start_x:x_max]), cmap="Blues", origin="upper")
    axes[0].set_xlim(start_x, x_max)
    axes[0].set_ylim(start_y, y_max)
    axes[0].set_title("Probability Map")
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(cp.asnumpy(cost_map[start_y:y_max, start_x:x_max]), cmap="Reds", origin="upper")
    axes[1].set_xlim(start_x, x_max)
    axes[1].set_ylim(start_y, y_max)
    axes[1].set_title("Cost Map")
    fig.colorbar(im2, ax=axes[1])
    for trail in raster_trails:
        if len(trail) > 1:
            trail_arr = np.array(trail)
            row, col = trail_arr[:, 0], trail_arr[:, 1]
            axes[0].plot(col - start_x, row - start_y, color='yellow', linewidth=1)
            axes[1].plot(col - start_x, row - start_y, color='black', linewidth=1)
    plt.tight_layout()
    plt.show()

def custom_astar(cost_map, start, end):
    rows, cols = cost_map.shape
    visited = np.full((rows, cols), False)
    costs = np.full((rows, cols), np.inf)
    parents = np.empty((rows, cols), dtype=object)

    def heuristic(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    pq = PriorityQueue()
    pq.put((heuristic(start, end), start))
    costs[start] = 0

    while not pq.empty():
        _, current = pq.get()
        if visited[current]:
            continue
        visited[current] = True
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return path[::-1]

        neighbors = [
            (current[0] + dr, current[1] + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]
            if 0 <= current[0] + dr < rows and 0 <= current[1] + dc < cols
        ]

        for neighbor in neighbors:
            if visited[neighbor]:
                continue
            dr, dc = neighbor[0] - current[0], neighbor[1] - current[1]
            move_cost = math.sqrt(dr ** 2 + dc ** 2)
            tentative_cost = costs[current] + cost_map[neighbor] * move_cost
            if tentative_cost < costs[neighbor]:
                costs[neighbor] = tentative_cost
                parents[neighbor] = current
                pq.put((tentative_cost + heuristic(neighbor, end), neighbor))
    return None

def compute_route(args):
    """
    Computes the least-cost path between two endpoints using custom_astar,
    then calculates the median cost (median DTM value along the path) and path length.
    Returns a tuple: (LineString, median_cost, length)
    """
    cost_map_np, prob_map_np, transform, start, end, start_id, end_id = args
    print(f"🚀 Calculating path from ID {start_id} to ID {end_id}, coords {start} → {end}")
    path = custom_astar(cost_map_np, start, end)
    if path is None or len(path) < 2:
        print(f"⚠️ Invalid path: insufficient points (start_id={start_id}, end_id={end_id})")
        return None
    path_coords = [transform * (col, row) for row, col in path]
    # Extract probability values along the path
    path_values = [prob_map_np[row, col] for row, col in path]
    path_values = np.array(path_values)
    # Replace NaNs with a default value (e.g., 0)
    path_values = np.nan_to_num(path_values, nan=0)
    median_cost = np.median(path_values)
    print('** Median cost: ', median_cost)
    print('Max value along path: ', np.max(path_values))
    try:
        line = LineString(path_coords)
        length = line.length
    except Exception as e:
        print(f"❌ Failed to create LineString (start_id={start_id}, end_id={end_id}): {e}")
        return None
    return (line, median_cost, length)


def compute_route_from_task(task):
    """
    task is a tuple of length 7, but only the first 5 elements are used for route computation.
    """
    return compute_route(task[:-2])

def process_all_endpoint_connections(valid_coords, valid_raster_indices, new_ids, orig_ids, verbose=False):
    """
    For each valid endpoint, find up to three closest endpoints using simple Euclidean distance.
    Exclude endpoints that come from the same original line (i.e. same orig_id).
    Returns a list of tasks for route computation.
    """
    tree = cKDTree(valid_coords)
    tasks = []
    n_points = len(valid_coords)
    for i in range(n_points):
        k = min(n_points, 10)
        dists, indices = tree.query(valid_coords[i], k=k)
        candidates = [idx for idx, dist in zip(indices, dists)
                      if idx != i and dist > 0.1 and orig_ids[idx] != orig_ids[i]]
        candidates = candidates[:3]
        for j in candidates:
            tasks.append((None, None, valid_raster_indices[i], valid_raster_indices[j],
                          new_ids[i], new_ids[j]))
            if verbose:
                print(f"🔗 Task: connect new_id {new_ids[i]} → {new_ids[j]} (orig_ids: {orig_ids[i]} vs {orig_ids[j]})")
    return tasks

def connect_segments(gdf, dtm_path, chm_path, scale=3):
    # Load and downsample DTM; align boundaries.
    with rasterio.open(dtm_path) as src_dt:
        gdf = align_boundaries(gdf, src_dt)
        data, new_transform = rasterio_downsample(src_dt, scale)
        prob_map = cp.asarray(data[0])
        transform = new_transform
        dtm_shape = data[0].shape

    # Reproject and downscale CHM to match the DTM grid.
    with rasterio.open(chm_path) as src_chm:
        chm_full = src_chm.read(1)
        chm_aligned = np.empty(dtm_shape, dtype=chm_full.dtype)
        rasterio.warp.reproject(
            source=chm_full,
            destination=chm_aligned,
            src_transform=src_chm.transform,
            src_crs=src_chm.crs,
            dst_transform=transform,
            dst_crs=src_dt.crs
        )

    # Create cost map based on the DTM (prob_map)
    cost_map = cp.where(
        (prob_map > -0.25) & (prob_map <= 0),
        0.1,
        cp.where(
            (prob_map >= 0.2) | (prob_map <= -0.35),
            10,
            1 + cp.abs(prob_map)
        )
    )
    # Increase cost where CHM > 1.
    chm_aligned_cp = cp.asarray(chm_aligned)
    cost_map = cp.where(chm_aligned_cp > 1, cost_map * 5, cost_map)
    prob_map = cp.where(chm_aligned_cp > 1, prob_map * 3, prob_map)

    # --- STEP 1: Extract endpoints and their original IDs ---
    valid_points, valid_coords, valid_raster_indices, valid_line_ids = [], [], [], []
    for idx, row in gdf.iterrows():
        line = row.geometry
        if line is None or line.geom_type != "LineString":
            continue
        for pt in [Point(line.coords[0]), Point(line.coords[-1])]:
            x, y = pt.x, pt.y
            res = safe_transform(transform, x, y, prob_map.shape)
            if res:
                valid_points.append(pt)
                valid_coords.append((x, y))
                valid_raster_indices.append(res)
                valid_line_ids.append(idx + 1)
    orig_ids = valid_line_ids.copy()
    # --- STEP 2: Update duplicate point IDs and store as new_ids ---
    new_ids = update_duplicate_point_ids(valid_line_ids)

    endpoints_gdf = gpd.GeoDataFrame({
        'geometry': valid_points,
        'original_id': orig_ids,
        'new_id': new_ids
    }, crs=gdf.crs)
    endpoints_path = os.path.join('/media/irina/My Book1/Conoco/DATA/Products/Trails/segformer/filtered_int',
                                  "endpoints.gpkg")
    endpoints_gdf.to_file(endpoints_path, driver='GPKG')
    print(f"✅ Endpoints saved to: {endpoints_path}")

    # --- STEP 3: Build candidate connections using new_ids and ensuring different orig_ids ---
    tree = cKDTree(valid_coords)
    candidate_dict = {}
    n_points = len(valid_coords)
    for i in range(n_points):
        k = min(n_points, 10)
        dists, indices = tree.query(valid_coords[i], k=k)
        candidates = [new_ids[j] for j, dist in zip(indices, dists)
                      if j != i and dist > 2 and orig_ids[j] != orig_ids[i]]
        candidate_dict[i] = candidates[:3]
    endpoints_gdf['closest_ids'] = [str(candidate_dict.get(i, [])) for i in range(n_points)]
    endpoints_gdf.to_file(endpoints_path, driver='GPKG')
    print(f"✅ Endpoints updated with candidate connections.")

    plot_maps(cp.asnumpy(prob_map), cost_map, gdf, transform, start_x=100, start_y=100)

    # --- STEP 4: Build connection tasks and compute routes ---
    tasks = process_all_endpoint_connections(valid_coords, valid_raster_indices, new_ids, orig_ids)
    cost_map_np = cost_map.get() if hasattr(cost_map, "get") else cost_map
    prob_map_np = prob_map.get() if hasattr(prob_map, "get") else prob_map
    # Build tasks including the probability map for median cost computation.
    final_tasks = [(cost_map_np, prob_map_np, transform) + task[2:] for task in tasks]
    results = [compute_route(task) for task in final_tasks]

    connection_geoms = []
    median_costs = []
    lengths = []
    src_ids = []
    tgt_ids = []
    for task, res in zip(final_tasks, results):
        if res is None:
            continue
        line, med_cost, length = res
        connection_geoms.append(line)
        median_costs.append(med_cost)
        lengths.append(length)
        src_ids.append(task[-3])
        tgt_ids.append(task[-2])

    gdf['connection_type'] = 'trail'
    connection_gdf = gpd.GeoDataFrame({
        'geometry': connection_geoms,
        'median_cost': median_costs,
        'length': lengths,
        'src_endpoint': src_ids,
        'tgt_endpoint': tgt_ids
    }, crs=gdf.crs)
    return pd.concat([gdf, connection_gdf], ignore_index=True)

# Main execution
centerline_path = '/media/irina/My Book1/Conoco/DATA/Products/Trails/segformer/filtered_int/centerline/area_E_polygon0-1_ppc_ncdtm_preds_segformer_filt500_uint8.gpkg'
dtm_path = '/media/irina/My Book1/Conoco/DATA/Drone_PPC_2024/ndtm/area_E_polygon0-1_ppc_ncdtm.tif'
chm_path = '/media/irina/My Book1/Conoco/DATA/Drone_PPC_2024/chm/area_E_polygon0-1_ppc_cCHM.tif'

downscale = 1

gdf = gpd.read_file(centerline_path)

connected = connect_segments(gdf, dtm_path, chm_path, scale = downscale)
output_path = centerline_path.replace('gpkg', 'chm_connected.gpkg')
connected.to_file(output_path, driver='GPKG')
print(f"✅ Connected segments saved to: {output_path}")
