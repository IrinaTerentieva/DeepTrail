import numpy as np
import rasterio
import geopandas as gpd
import cv2
from rasterio.features import shapes
from shapely.geometry import LineString
from skimage.transform import hough_line, hough_line_peaks
from scipy.spatial import KDTree
import os


import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import LineString


# --------- Convert Raster to Vector ---------
def raster_to_vector(raster_path, threshold=40):
    """ Converts trail raster to vector lines """

    with rasterio.open(raster_path) as src:
        image = src.read(1)  # Read the raster band
        transform = src.transform
        crs = src.crs

        # Normalize the data (handle nodata values)
        image = np.where(image < threshold, 0, image)

        # Extract vector shapes
        shapes_list = list(shapes(image, mask=image > 0, transform=transform))

        lines = []
        for shape, value in shapes_list:
            coords = shape.get("coordinates", [])
            if isinstance(coords, list) and all(isinstance(coord, tuple) and len(coord) == 2 for coord in coords):
                lines.append(LineString(coords))

        if not lines:
            print("[ERROR] No valid trail segments were found in the raster. Adjust the threshold!")
        else:
            print(f"[Step 1] Successfully extracted {len(lines)} trail segments.")

        return gpd.GeoDataFrame(geometry=lines, crs=crs)



# --------- Detect Main Trail Orientations ---------
def detect_trail_orientation(gdf):
    """ Uses Hough Transform to detect dominant trail orientations """
    all_lines = np.concatenate([np.array(line.xy).T for line in gdf.geometry])
    edges = np.zeros((5000, 5000))  # Large empty space for Hough Transform
    for x, y in all_lines.astype(int):
        edges[y, x] = 1  # Mark pixels in binary image

    h, theta, d = hough_line(edges)
    hough_lines = []

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        hough_lines.append(angle)

    print(f"[Step 2] Detected {len(hough_lines)} dominant trail orientations.")
    return hough_lines


# --------- Connect Broken Trail Segments ---------
def connect_trails(gdf, max_distance=10):
    """ Connects broken trails by finding nearest segments """
    if gdf.empty:
        print("[ERROR] No trails detected!")
        return gdf

    # Extract line endpoints
    endpoints = np.array([(line.coords[0], line.coords[-1]) for line in gdf.geometry])
    points = np.vstack(endpoints)  # Flatten to point list

    # KDTree for fast nearest-neighbor search
    tree = KDTree(points)

    # Find close neighbors
    new_connections = []
    for i, (start, end) in enumerate(endpoints):
        dist, idx = tree.query(end, k=2)  # Find closest point
        if dist[1] < max_distance:  # Avoid self-matching
            new_connections.append(LineString([end, points[idx[1]]]))

    # Merge connected trails into GeoDataFrame
    connected_gdf = gpd.GeoDataFrame(geometry=gdf.geometry.tolist() + new_connections, crs=gdf.crs)
    print(f"[Step 3] Connected {len(new_connections)} broken segments.")
    return connected_gdf


# --------- Save Output ---------
def save_to_gpkg(output_path, gdf):
    """ Saves the final connected trails to a GeoPackage """
    gdf.to_file(output_path, driver="GPKG")
    print(f"[Step 4] Saved connected trails to {output_path}")


# --------- Run Processing ---------
def process_trails(trail_raster, output_path):
    print("[Step 1] Processing Trail Raster...")
    trails_gdf = raster_to_vector(trail_raster)

    print("[Step 2] Detecting Trail Orientations...")
    detect_trail_orientation(trails_gdf)

    print("[Step 3] Connecting Broken Segments...")
    connected_trails = connect_trails(trails_gdf)

    print("[Step 4] Saving Results...")
    save_to_gpkg(output_path, connected_trails)


# --------- Main Execution ---------
if __name__ == "__main__":
    output_gpkg = "/media/irina/My Book1/Conoco/DATA/Drone_PPC_2024/ndtm/Processed_Results/connected_trails.gpkg"
    trail_raster_path = "/home/irina/HumanFootprint/DATA/Test_Models/Area_C_combined_ncdtm_Human_lesstrails_DTM10_512_byCNN_7ep_preds_v2.1.tif"

    process_trails(trail_raster_path, output_gpkg)
