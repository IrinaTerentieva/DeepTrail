import geopandas as gpd
import os
import json
import sys
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
import rasterio
import cv2
import networkx as nx
import numpy as np
import scipy.ndimage.measurements
import shapely.geometry
from PIL import Image
from skimage import morphology, segmentation
from affine import Affine

# File paths
input = '/media/irro/All/HumanFootprint/DATA/intermediate/1_label.tif'


def find_color(im: Image, rgb: Tuple[int]) -> np.ndarray:
    """Given an RGB image, return an ndarray with 1s where the pixel is the given color."""
    px = np.asarray(im)
    out = np.zeros(im.size, dtype=np.uint8)
    r, g, b = rgb
    out[(px[:, :, 0] == r) & (px[:, :, 1] == g) & (px[:, :, 2] == b)] = 1
    return out


def zhang_suen_node_detection(skel: np.ndarray) -> List[Tuple[int]]:
    """Find nodes based on a skeletonized bitmap.

    (From nefi) Node detection based on criteria put forward in "A fast parallel algorithm
    for thinning digital patterns" by T. Y. Zhang and C. Y. Suen. Pixels p of the skeleton
    are categorized as nodes/non-nodes based on the value of a function A(p) depending on
    the pixel neighborhood of p. Please check the above paper for details.

    A(p1) == 1: The pixel p1 sits at the end of a skeleton line, thus a node
    of degree 1 has been found.
    A(p1) == 2: The pixel p1 sits in the middle of a skeleton line but not at
    a branching point, thus a node of degree 2 has been found. Such nodes are
    ignored and not introduced to the graph.
    A(p1) >= 3: The pixel p1 belongs to a branching point of a skeleton line,
    thus a node of degree >=3 has been found.

    Args:
        *skel* : Skeletonised source image. The skeleton must be exactly 1 pixel wide.

    Returns:
        *nodes* : List of (x, y) coordinates of nodes
    """
    skel = np.pad(skel, 1)
    item = skel.item

    def check_pixel_neighborhood(x, y, skel):
        """
        Check the number of components around a pixel.
        If it is either 1 or more than 3, it is a node.
        """
        p2 = item(x - 1, y)
        p3 = item(x - 1, y + 1)
        p4 = item(x, y + 1)
        p5 = item(x + 1, y + 1)
        p6 = item(x + 1, y)
        p7 = item(x + 1, y - 1)
        p8 = item(x, y - 1)
        p9 = item(x - 1, y - 1)

        # The function A(p1),
        # where p1 is the pixel whose neighborhood is beeing checked
        components = (
                (p2 == 0 and p3 == 1)
                + (p3 == 0 and p4 == 1)
                + (p4 == 0 and p5 == 1)
                + (p5 == 0 and p6 == 1)
                + (p6 == 0 and p7 == 1)
                + (p7 == 0 and p8 == 1)
                + (p8 == 0 and p9 == 1)
                + (p9 == 0 and p2 == 1)
        )
        return (components >= 3) or (components == 1)

    nodes = []
    w, h = skel.shape
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            if item(x, y) != 0 and check_pixel_neighborhood(x, y, skel):
                nodes.append((x - 1, y - 1))
    return nodes


def find_dense_skeleton_nodes(skel: np.ndarray) -> List[Tuple[int, int]]:
    """Find "dense" (2x2 or larger) regions in the skeleton."""
    eroded = morphology.binary_erosion(np.pad(skel, 1), np.ones((2, 2)))[1:-1, 1:-1]

    # Find the centers of mass of connected components
    labeled_array, num_features = scipy.ndimage.measurements.label(eroded)
    centers = scipy.ndimage.measurements.center_of_mass(eroded, labeled_array, [*range(1, num_features + 1)])
    return [(int(x), int(y)) for (x, y) in centers]


def add_dense_nodes(nodes: List[Tuple[int, int]], dense_nodes: List[Tuple[int, int]], min_distance=5) -> List[
    Tuple[int, int]]:
    """Add in new nodes which are distinct from the old ones."""
    keep = []
    min_d2 = min_distance ** 2
    for node in dense_nodes:
        x, y = node
        is_ok = True
        for nx, ny in nodes:
            d2 = (x - nx) ** 2 + (y - ny) ** 2
            if d2 < min_d2:
                is_ok = False
                break
        if is_ok:
            keep.append(node)

    #     print(f'Adding {len(keep)}/{len(dense_nodes)} dense nodes to existing {len(nodes)} nodes.')
    return [*nodes, *keep]


@dataclass
class Path:
    start: Tuple[int, int]
    stop: Tuple[int, int]
    path: List[Tuple[int, int]]


def is_new_path(paths: List[Path], path: Path) -> bool:
    """Is this a new path, or does it overlap signficantly with existing paths?"""
    candidates = [p for p in paths if p.start == path.start and p.stop == path.stop]
    other_points = {coord for p in candidates for coord in p.path[1:-1]}
    interior = set(path.path[1:-1])
    if other_points & interior:
        return False
    return True


def is_valid_self_loop(path: List[Tuple[int, int]], min_self_loop_distance: int) -> bool:
    if len(path) < min_self_loop_distance:
        return False
    # Only the end node can appear twice in a self-loop
    return len([c for c, n in Counter(path).items() if n >= 2]) == 1


def log(message):
    print(message)


def find_paths(skel: np.ndarray, nodes: List[Tuple[int]], min_distance=5) -> List[Path]:
    """Find paths between nodes in the graph using the connectivity in the skeleton.

    This returns a list of edges (pairs of nodes) with the following properties.
        - path: list of coordinates connecting the nodes (including the nodes)
        - d: length of the path

    This will early-out if a path shorter than min_distance is found.

    There may be multiple distinct paths between the same nodes, or a path between a node and itself.
    """
    width, height = skel.shape

    def neighbors(x, y):
        for dy in (-1, 0, 1):
            cy = y + dy
            if cy < 0 or cy >= height:
                continue
            for dx in (-1, 0, 1):
                cx = x + dx
                if (dx != 0 or dy != 0) and 0 <= cx < width and skel[cx, cy]:
                    yield cx, cy

    # each cell points back to its parent
    parents = {n: None for n in nodes}

    def trace_back(node):
        trace = []
        while node:
            trace.append(node)
            node = parents.get(node)
        return trace

    d = {n: 0 for n in nodes}  # used to avoid backtracking

    edges = []
    frontier = [*nodes]
    while frontier:
        next_frontier = []
        for n in frontier:
            x, y = n
            for c in neighbors(x, y):
                if c not in parents:
                    parents[c] = n
                    next_frontier.append(c)
                    d[c] = 1 + d[n]
                else:
                    if d[c] >= d[n]:
                        # we've got a connection! Follow both cells back to trace it out
                        tn = trace_back(n)
                        tc = trace_back(c)
                        tc.reverse()
                        path = [*tc, *tn]
                        endpoints = (path[0], path[-1])
                        start, stop = min(endpoints), max(endpoints)
                        new_path = Path(start, stop, path)
                        # Ignore redundant paths and short self-loops
                        if is_new_path(edges, new_path) and (
                                start != stop or is_valid_self_loop(path, min_distance)
                        ):
                            edges.append(new_path)
                            if len(path) - 1 < min_distance:
                                # This edge will get pruned out anyway, so no need to keep looking.
                                return edges

        frontier = next_frontier
    return edges


def merge_nodes(
        nodes: List[Tuple[int, int]], edges: List[Path], n1: Tuple[int, int], n2: Tuple[int, int]
) -> List[Tuple[int, int]]:
    ends = {n1, n2}
    paths = [e.path for e in edges if {e.start, e.stop} == ends]
    assert paths
    path = min(paths, key=lambda p: len(p))
    idx = len(path) // 2
    new_node = path[idx]
    return [new_node] + [n for n in nodes if n != n1 and n != n2]


def make_graph(nodes: List[Tuple[int, int]], edges: List[Path]) -> nx.MultiGraph:
    g = nx.MultiGraph()
    g.add_nodes_from(nodes)
    for edge in edges:
        g.add_edge(edge.start, edge.stop, path=edge.path, d=len(edge.path) - 1)
    return g


def simplify_paths(g: nx.Graph, tolerance=1) -> nx.Graph:
    for n1, n2, k in g.edges(keys=True):
        g[n1][n2][k]['path'] = shapely.geometry.LineString(g[n1][n2][k]['path']).simplify(tolerance)
    return g


def extract_network(px: np.ndarray, min_distance=8) -> nx.Graph:
    skel = morphology.skeletonize(px)
    #     print(f'Skeleton px={skel.sum()}')
    g = connect_graph(skel, min_distance)
    g = simplify_paths(g)
    return g


def create_circular_mask(shape, center, radius):
    w, h = shape
    cx, cy = center
    X, Y = np.ogrid[:w, :h]
    dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = dist_from_center <= radius
    return mask


# Sum of the min & max of (a, b, c)
def hilo(a, b, c):
    if c < b:
        b, c = c, b
    if b < a:
        a, b = b, a
    if c < b:
        b, c = c, b
    return a + c


def complement(rgb):
    """Return a complementary color; see https://stackoverflow.com/a/40234924/388951"""
    k = hilo(*rgb)
    return tuple(k - u for u in rgb)


def connect_graph(skel: np.ndarray, min_distance: int) -> nx.MultiGraph:
    nodes = zhang_suen_node_detection(skel)
    dense_nodes = find_dense_skeleton_nodes(skel)
    nodes = add_dense_nodes(nodes, dense_nodes)
    edges = find_paths(skel, nodes, min_distance)

    processed_pairs = set()
    while True:
        merged_any = False
        for edge in edges:
            d = len(edge.path) - 1
            n1 = edge.start
            n2 = edge.stop
            node_pair = frozenset([n1, n2])

            if d < min_distance and node_pair not in processed_pairs:
                nodes = merge_nodes(nodes, edges, n1, n2)
                edges = find_paths(skel, nodes, min_distance)
                #                 print(f'Merged {n1} and {n2}, d={d}')
                merged_any = True
                processed_pairs.add(node_pair)
                break

        if not merged_any:
            break  # Exit the loop if no more nodes were merged

    return make_graph(nodes, edges)


from shapely.geometry import LineString, Point
import networkx as nx


def network_to_geojson(g: nx.Graph, transform):
    features = []
    for i, (n1, n2, k) in enumerate(g.edges(keys=True)):
        edge_data = g[n1][n2][k]
        path = edge_data['path']

        # Transform coordinates and ensure correct order
        # Note: Assuming path.coords gives a list of tuples like (x, y)
        transformed_path = [transform * Point(p[1], p[0]).coords[0] for p in path.coords]
        feature = {
            'type': 'Feature',
            'id': f'street-{i}',
            'properties': {
                'street': i,
                'len': edge_data['d'],
                'start_lon': transformed_path[0][0],  # Longitude of start
                'start_lat': transformed_path[0][1],  # Latitude of start
                'stop_lon': transformed_path[-1][0],  # Longitude of stop
                'stop_lat': transformed_path[-1][1],  # Latitude of stop
            },
            'geometry': {
                'type': 'LineString',
                'coordinates': transformed_path
            }
        }
        features.append(feature)

    return {
        'type': 'FeatureCollection',
        'features': features
    }


def save_network_as_shapefile(g, tif_path):
    # Open the TIFF file to get its affine transform and CRS
    with rasterio.open(tif_path) as src:
        transform = src.transform
        crs = src.crs

    # Convert network to GeoJSON with transformed coordinates
    network_geojson = network_to_geojson(g, transform)

    # Convert GeoJSON to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(network_geojson, crs=crs)

    # Save to shapefile
    shapefile_path = tif_path.replace('.tif', '.shp')
    gdf.to_file(shapefile_path)
    print(f'Network shapefile saved to {shapefile_path}')


def extract_network_from_tif(tif_input, threshold=10):
    # Check if input is a file path or a NumPy array
    if isinstance(tif_input, str) and os.path.isfile(tif_input):
        # It's a file path
        with rasterio.open(tif_input) as src:
            binary_image = src.read(1) > threshold
    elif isinstance(tif_input, np.ndarray):
        # It's a NumPy array
        binary_image = tif_input > threshold
    else:
        raise ValueError("Input must be a file path to a .tif file or a NumPy array.")

    print(f'Binary pixels: {binary_image.sum()}')

    # Skeletonize the binary image
    skel = morphology.skeletonize(binary_image)

    # Extract the network from the skeletonized image
    g = connect_graph(skel, min_distance=10)  # Adjust min_distance as needed
    #     print(g)
    g = simplify_paths(g)
    print('after simplify', g)

    return skel, g


def network_to_geojson_new(g: nx.Graph):
    features = []
    for i, (n1, n2) in enumerate(g.edges()):
        edge_data = g[n1][n2]
        path = edge_data.get('path')

        if path:
            # Use the coordinates directly from the LineString
            coordinates = list(path.coords)
            feature = {
                'type': 'Feature',
                'id': f'street-{i}',
                'properties': {
                    'street': i,
                    'len': edge_data.get('d', 0),
                    'start_lon': coordinates[0][0],
                    'start_lat': coordinates[0][1],
                    'stop_lon': coordinates[-1][0],
                    'stop_lat': coordinates[-1][1],
                },
                'geometry': {'type': 'LineString', 'coordinates': coordinates}
            }
            features.append(feature)

    return {'type': 'FeatureCollection', 'features': features}

tif_files = [input]
for tif_path in tif_files:
    with rasterio.open(tif_path) as src:
        print(tif_path)
        transform = src.transform
        crs = src.crs
        print('Start')

        threshold = 30
        skel, g = extract_network_from_tif(tif_path, threshold)
        while len(g.edges) < 10:
            print('Ooops')
            threshold = threshold + 5
            skel, g = extract_network_from_tif(tif_path, threshold)
            print('Thre: ', threshold)

            if threshold > 40:
                continue

        save_network_as_shapefile(g, tif_path)