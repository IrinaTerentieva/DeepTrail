import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev


# ---- Step 1: Load Hough-detected trails ----
def load_hough_trails(hough_geojson):
    return gpd.read_file(hough_geojson)


# ---- Step 2: Filter Short & Misaligned Trails ----
def filter_trails(gdf, min_length=10, max_angle_diff=15):
    gdf["length"] = gdf.geometry.length
    gdf = gdf[gdf["length"] > min_length]

    def compute_angle(line):
        coords = list(line.coords)
        dx, dy = coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1]
        return np.degrees(np.arctan2(dy, dx)) % 180

    gdf["angle"] = gdf.geometry.apply(compute_angle)
    median_angle = gdf["angle"].median()
    gdf = gdf[abs(gdf["angle"] - median_angle) < max_angle_diff]

    return gdf.drop(columns=["length", "angle"])


# ---- Step 3: Pair Parallel Tracks (Left & Right Ruts) ----
def pair_parallel_trails(gdf, max_distance=15):
    midpoints = np.array([(line.centroid.x, line.centroid.y) for line in gdf.geometry])
    clustering = DBSCAN(eps=max_distance, min_samples=2).fit(midpoints)
    gdf["pair_group"] = clustering.labels_
    return gdf[gdf["pair_group"] != -1]


# ---- Step 4: Connect Broken Trail Segments ----
from scipy.spatial import KDTree


def connect_trails(gdf, max_gap_distance=10):
    """Connect broken trail segments by finding nearby points."""

    # Extract trail points
    points = [geom.centroid for geom in gdf.geometry]

    # Create KDTree for nearest neighbor search
    tree = KDTree([(p.x, p.y) for p in points])

    # Connect close segments
    connections = []
    for i, p in enumerate(points):
        dist, idx = tree.query([(p.x, p.y)], k=2)  # k=2 to avoid self-matching
        dist = dist[0][1]  # Extract the second closest point (ignoring itself)
        j = idx[0][1]  # Extract the index of the closest neighbor

        if j != i and dist < max_gap_distance:
            connections.append((i, j))

    print(f"[DEBUG] Connected {len(connections)} trail segments.")
    return gdf


# ---- Step 5: Smooth Trails ----
def smooth_trails(gdf, smoothing_factor=0.1):
    smoothed_lines = []
    for line in gdf.geometry:
        coords = np.array(line.coords)
        if len(coords) < 4:
            smoothed_lines.append(line)
            continue
        tck, u = splprep([coords[:, 0], coords[:, 1]], s=smoothing_factor)
        new_coords = np.array(splev(np.linspace(0, 1, 50), tck)).T
        smoothed_lines.append(LineString(new_coords))

    gdf["geometry"] = smoothed_lines
    return gdf


# ---- Step 6: Merge with Original Trails ----
def merge_with_original(original_trails, processed_trails, output_geojson):
    orig_gdf = gpd.read_file(original_trails)
    processed_gdf = processed_trails.copy()
    processed_gdf["processed"] = 1
    merged = orig_gdf.sjoin_nearest(processed_gdf, how="left", distance_col="proximity")
    merged.to_file(output_geojson, driver="GeoJSON")
    return merged


# ---- Full Pipeline Execution ----
def process_trails(hough_geojson, original_trails, output_geojson):
    print("[Step 1] Loading Hough-detected trails...")
    gdf = load_hough_trails(hough_geojson)
    print(len(gdf))

    print("[Step 2] Filtering misaligned & short trails...")
    gdf = filter_trails(gdf)
    print(len(gdf))
    print("[Step 3] Pairing parallel trails...")
    gdf = pair_parallel_trails(gdf)
    print(len(gdf))
    print("[Step 4] Connecting broken segments...")
    gdf = connect_trails(gdf)
    print(len(gdf))
    print("[Step 5] Smoothing trails...")
    gdf = smooth_trails(gdf)

    print("[Step 6] Merging with original trails...")
    final_output = merge_with_original(original_trails, gdf, output_geojson)
    print("[Done] Processed trails saved!")
    return final_output


# ---- Run the Pipeline ----
if __name__ == "__main__":
    hough_geojson = "/media/irina/My Book1/Conoco/DATA/Drone_PPC_2024/ndtm/Processed_Results/hough_trails.geojson"  # Input Hough-transformed trails
    original_trails = "/media/irina/My Book1/Conoco/DATA/Drone_PPC_2024/ndtm/Processed_Results/original_trails.geojson"  # Your original trails dataset
    output_geojson = "/media/irina/My Book1/Conoco/DATA/Drone_PPC_2024/ndtm/Processed_Results/final_trails.geojson"  # Final output with attributes
    process_trails(hough_geojson, original_trails, output_geojson)
