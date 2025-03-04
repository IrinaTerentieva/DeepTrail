#!/usr/bin/env python3
import random
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate, scale, translate
from scipy.interpolate import splprep, splev


# --- Spline smoothing function ---
def strong_smooth_geometry(geom, smoothing_factor=50.0, num_points=50, simplify_tolerance=0.8):
    """
    Smooths a geometry using a two-step process:
      1. Simplify the geometry to remove small-scale details.
      2. Apply spline interpolation to produce a smooth curve.

    Assumes the input is a line geometry.
    """
    if geom.is_empty:
        return geom

    simplified = geom.simplify(simplify_tolerance, preserve_topology=True)
    if simplified.geom_type == 'LineString':
        x, y = simplified.xy
        try:
            tck, u = splprep([x, y], s=smoothing_factor)
            unew = np.linspace(0, 1, num_points)
            out = splev(unew, tck)
            return LineString(zip(out[0], out[1]))
        except Exception:
            return simplified
    elif simplified.geom_type == 'MultiLineString':
        smoothed = [strong_smooth_geometry(line, smoothing_factor, num_points, simplify_tolerance)
                    for line in simplified.geoms]
        return MultiLineString(smoothed)
    elif simplified.geom_type == 'Polygon':
        # Convert polygon to its exterior boundary.
        return LineString(simplified.exterior.coords)
    else:
        return simplified


# --- Ensure geometry is a line ---
def ensure_line(geom):
    """
    If the input geometry is a Polygon or MultiPolygon, extracts its exterior boundary
    (or boundaries) and returns a LineString or MultiLineString.
    Otherwise, returns geom unchanged.
    """
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return LineString(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        lines = [LineString(poly.exterior.coords) for poly in geom.geoms]
        return MultiLineString(lines)
    else:
        return geom


# --- Flatten geometry to a list of LineStrings ---
def flatten_geom(geom):
    """
    Returns a list of LineStrings from geom.
    """
    if geom is None:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    elif geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    else:
        return []


# --- Create parallel pair of lines ---
def create_parallel_pair(geom, distance):
    """
    Creates a pair of parallel offset lines from the given line geometry.
    Uses shapely's parallel_offset method to generate offsets on the left and right.
    If the offsets return MultiLineString, flattens them and combines into a MultiLineString.
    Returns a combined MultiLineString if both offsets exist; otherwise, returns the available offset.
    """
    left_line = None
    right_line = None
    try:
        if geom.geom_type == 'LineString':
            left_line = geom.parallel_offset(distance, side='left')
            right_line = geom.parallel_offset(distance, side='right')
        elif geom.geom_type == 'MultiLineString':
            left_lines = []
            right_lines = []
            for line in geom.geoms:
                off_left = line.parallel_offset(distance, side='left')
                off_right = line.parallel_offset(distance, side='right')
                if off_left is not None:
                    if off_left.geom_type == 'MultiLineString':
                        left_lines.extend(list(off_left.geoms))
                    else:
                        left_lines.append(off_left)
                if off_right is not None:
                    if off_right.geom_type == 'MultiLineString':
                        right_lines.extend(list(off_right.geoms))
                    else:
                        right_lines.append(off_right)
            if left_lines:
                left_line = MultiLineString(left_lines) if len(left_lines) > 1 else left_lines[0]
            if right_lines:
                right_line = MultiLineString(right_lines) if len(right_lines) > 1 else right_lines[0]
    except Exception as e:
        print("Error creating parallel pair:", e)
    pair_lines = flatten_geom(left_line) + flatten_geom(right_line)
    if pair_lines:
        if len(pair_lines) == 1:
            return pair_lines[0]
        else:
            return MultiLineString(pair_lines)
    else:
        return geom


# --- Variant generation ---
def generate_variant(geom, variant_type='base'):
    """
    Generates a variant of the given (smoothed) line geometry.

    - For 'base': returns the geometry as-is.
    - For 'augmented': applies a random rotation (0 to 360°)
      and random scaling (0.7 to 1.3) about the centroid.

    Returns a tuple: (variant_geom, applied_rotation, applied_scale)
    """
    if variant_type == 'base':
        return geom, 0, 1
    else:
        angle = random.uniform(0, 360)
        scale_factor = random.uniform(0.7, 1.3)
        variant_geom = rotate(geom, angle, origin='centroid', use_radians=False)
        variant_geom = scale(variant_geom, xfact=scale_factor, yfact=scale_factor, origin='centroid')
        return variant_geom, angle, scale_factor


def main():
    # Input GeoPackage containing trails.
    input_gpkg = "/media/irina/My Book/Surmont/manual/art_trail.gpkg"
    # Output GeoPackage for the generated variants.
    output_gpkg = "/media/irina/My Book/Surmont/manual/art_trail_variants.gpkg"

    # Read input trails.
    gdf = gpd.read_file(input_gpkg)

    # Use 'trail_id' if available; otherwise, use the index.
    if 'trail_id' in gdf.columns:
        unique_ids = gdf['trail_id'].unique()
    else:
        unique_ids = gdf.index

    records = []
    new_id = 1

    for tid in unique_ids:
        # Merge all features for this unique trail.
        if 'trail_id' in gdf.columns:
            trail_features = gdf[gdf['trail_id'] == tid]
        else:
            trail_features = gdf.loc[[tid]]
        merged_geom = unary_union(trail_features.geometry.tolist())
        merged_line = ensure_line(merged_geom)
        # Smooth the merged line.
        smoothed_geom = strong_smooth_geometry(merged_line, smoothing_factor=100.0, num_points=200,
                                               simplify_tolerance=0.8)

        # Create the parallel pair variant.
        # Choose a random offset distance from 0.5 to 1.
        offset_distance = random.uniform(0.5, 0.8)
        pair_geom = create_parallel_pair(smoothed_geom, distance=offset_distance)
        # Save the pair variant with augm_id 0 (they share the same orig_id).
        records.append({
            "orig_id": tid,
            "variant": "parallel_pair",
            "augm_id": 0,
            "new_id": new_id,
            "offset_distance": offset_distance,
            "rotation": 0,
            "scale": 1,
            "geometry": pair_geom
        })
        new_id += 1

        # Generate 10 augmented variants from the parallel pair.
        for v in range(10):
            aug_variant, angle, scale_factor = generate_variant(pair_geom, variant_type='augmented')
            records.append({
                "orig_id": tid,
                "variant": "augmented",
                "augm_id": v + 1,
                "new_id": new_id,
                "rotation": angle,
                "scale": scale_factor,
                "geometry": aug_variant
            })
            new_id += 1

    gdf_out = gpd.GeoDataFrame(records, crs=gdf.crs)
    gdf_out.to_file(output_gpkg, driver="GPKG")
    print(f"Saved {len(gdf_out)} variant features to {output_gpkg}")
    print(f"Number of unique trails in input: {len(unique_ids)}")
    print(f"Number of unique trails in output: {len(gdf_out['orig_id'].unique())}")


if __name__ == "__main__":
    main()
