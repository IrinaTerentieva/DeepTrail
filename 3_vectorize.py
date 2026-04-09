"""
Vectorize a binary trail prediction raster into LineString centerlines.

Steps:
  1. Read the prediction GeoTIFF produced by 2_run_predictions.py.
  2. Threshold to a binary mask.
  3. Optionally drop small connected components (noise).
  4. Skeletonize to a 1-pixel-wide centerline.
  5. Trace skeleton pixels into shapely LineStrings using 8-connectivity.
  6. Merge and (optionally) simplify, then save as a GeoPackage.
"""

import os

import geopandas as gpd
import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig
from rasterio.transform import xy
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union
from skimage.morphology import remove_small_objects, skeletonize


def _derive_prediction_path(cfg: DictConfig) -> str:
    """Reproduce the output path that 2_run_predictions.py writes to."""
    input_tif = cfg.paths_to_predict.input_to_predict_file
    input_name = os.path.splitext(os.path.basename(input_tif))[0]
    input_folder = os.path.dirname(input_tif)
    output_dir = os.path.join(input_folder, cfg.trail_mapping.output_subfolder)
    model_tag = cfg.trail_mapping.model_name
    return os.path.join(output_dir, f"{input_name}_{model_tag}_pred.tif")


def skeleton_to_linestrings(skeleton: np.ndarray, transform) -> list[LineString]:
    """Convert a binary skeleton to a list of pixel-segment LineStrings.

    For every skeleton pixel, we look at four of its eight neighbors
    (right, down, down-right, down-left) to avoid double-counting, and
    emit a 2-point LineString in georeferenced coordinates between the
    pixel centers.
    """
    rows, cols = np.where(skeleton)
    if rows.size == 0:
        return []

    skel = skeleton.astype(bool)
    h, w = skel.shape
    lines: list[LineString] = []

    # Pre-compute geo coordinates for every skeleton pixel
    xs, ys = xy(transform, rows.tolist(), cols.tolist(), offset="center")
    coord_lookup = {(int(r), int(c)): (float(x), float(y))
                    for r, c, x, y in zip(rows, cols, xs, ys)}

    # 4 of 8 neighbor offsets — the other 4 are mirror duplicates
    neighbors = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for (r, c), (x, y) in coord_lookup.items():
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
                x2, y2 = coord_lookup[(rr, cc)]
                lines.append(LineString([(x, y), (x2, y2)]))
    return lines


def vectorize_prediction(
    prediction_path: str,
    output_path: str,
    threshold: int = 50,
    min_pixels: int = 50,
    simplify_tolerance: float = 0.0,
) -> str:
    """Vectorize a binary prediction raster into LineStrings.

    Parameters
    ----------
    prediction_path : str
        Path to the prediction GeoTIFF (uint8, 0..100).
    output_path : str
        Path to write the GeoPackage.
    threshold : int
        Per-pixel confidence threshold (0..100).
    min_pixels : int
        Drop connected components smaller than this many pixels before skeletonization.
    simplify_tolerance : float
        Douglas-Peucker tolerance in CRS units (meters). 0 = no simplification.
    """
    print(f"Reading: {prediction_path}")
    with rasterio.open(prediction_path) as src:
        pred = src.read(1)
        transform = src.transform
        crs = src.crs

    print(f"Thresholding at {threshold} (0..100)...")
    binary = pred >= threshold
    print(f"  positive pixels: {int(binary.sum()):,}")

    if min_pixels > 0:
        print(f"Removing connected components smaller than {min_pixels} px...")
        binary = remove_small_objects(binary, min_size=min_pixels, connectivity=2)
        print(f"  positive pixels after cleanup: {int(binary.sum()):,}")

    print("Skeletonizing...")
    skeleton = skeletonize(binary)
    print(f"  skeleton pixels: {int(skeleton.sum()):,}")

    print("Tracing skeleton -> LineStrings...")
    segments = skeleton_to_linestrings(skeleton, transform)
    print(f"  raw 2-point segments: {len(segments):,}")

    if not segments:
        print("No skeleton found — nothing to vectorize.")
        return output_path

    print("Merging segments...")
    merged = linemerge(unary_union(segments))

    if merged.geom_type == "LineString":
        geoms = [merged]
    else:  # MultiLineString
        geoms = list(merged.geoms)
    print(f"  merged into {len(geoms):,} LineString(s)")

    if simplify_tolerance > 0:
        print(f"Simplifying with tolerance {simplify_tolerance} (CRS units)...")
        geoms = [g.simplify(simplify_tolerance, preserve_topology=False) for g in geoms]

    gdf = gpd.GeoDataFrame(
        {"length_m": [g.length for g in geoms]},
        geometry=geoms,
        crs=crs,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")
    print(f"Vector centerlines saved to: {output_path}")
    return output_path


@hydra.main(config_path="configs", config_name="inference", version_base=None)
def main(cfg: DictConfig):
    vec_cfg = cfg.get("vectorize", {}) or {}
    threshold = int(vec_cfg.get("threshold", 50))
    min_pixels = int(vec_cfg.get("min_pixels", 50))
    simplify_tolerance = float(vec_cfg.get("simplify_tolerance", 0.0))

    input_file = vec_cfg.get("input_file") or _derive_prediction_path(cfg)
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Prediction raster not found at {input_file}. "
            f"Set vectorize.input_file or run 2_run_predictions.py first."
        )

    output_file = vec_cfg.get("output_file")
    if not output_file:
        base, _ = os.path.splitext(input_file)
        output_file = f"{base}_thre{threshold}_centerlines.gpkg"

    vectorize_prediction(
        prediction_path=input_file,
        output_path=output_file,
        threshold=threshold,
        min_pixels=min_pixels,
        simplify_tolerance=simplify_tolerance,
    )


if __name__ == "__main__":
    main()
