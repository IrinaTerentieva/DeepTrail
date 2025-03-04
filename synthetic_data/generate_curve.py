#!/usr/bin/env python3
import random
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate, scale
from scipy.interpolate import splprep, splev


# --- Helper: Ensure geometry is a line ---
def ensure_line(geom):
    """
    Converts a Polygon or MultiPolygon into a LineString or MultiLineString
    by extracting exterior boundaries. Leaves other geometries unchanged.
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


# --- Spline smoothing function ---
def strong_smooth_geometry(geom, smoothing_factor=50.0, num_points=50, simplify_tolerance=0.8):
    """
    Smooths a geometry using a two-step process:
      1. Simplify the geometry to remove small-scale details.
      2. Apply spline interpolation to produce a smooth curve.

    Assumes the input is a line geometry (LineString or MultiLineString).
    """
    if geom.is_empty:
        return geom
    simplified = geom.simplify(simplify_tolerance, preserve_topology=True)
    # Ensure we work with line geometry.
    simplified = ensure_line(simplified)
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
    else:
        return simplified


# --- Variant generation ---
def generate_variant(geom, variant_type='base'):
    """
    Generates a variant of the given (smoothed) line geometry.

    - For 'base': returns the geometry as-is.
    - For 'augmented': applies a random rotation (±15°) and scaling (0.9 to 1.1)
      about the centroid.

    Returns a tuple: (variant_geom, applied_rotation, applied_scale)
    """
    if variant_type == 'base':
        return geom, 0, 1
    else:
        angle = random.uniform(-15, 15)
        scale_factor = random.uniform(0.9, 1.1)
        variant_geom = rotate(geom, angle, origin='centroid', use_radians=False)
        variant_geom = scale(variant_geom, xfact=scale_factor, yfact=scale_factor, origin='centroid')
        return variant_geom, angle, scale_factor


def main():
    # Input GeoPackage containing trails.
    input_gpkg = "/media/irina/My Book/Surmont/manual/art_trail.gpkg"
    # Output GeoPackage for the augmented variants.
    output_gpkg = "/media/irina/My Book/Surmont/manual/art_trail_variants.gpkg"

    # Read the input trails.
    gdf = gpd.read_file(input_gpkg)

    # Use 'trail_id' field if available; otherwise, use the index.
    if 'trail_id' in gdf.columns:
        unique_ids = gdf['trail_id'].unique()
    else:
        unique_ids = gdf.index

    records = []
    new_id = 1

    for tid in unique_ids:
        # Merge all features for the given trail.
        if 'trail_id' in gdf.columns:
            trail_features = gdf[gdf['trail_id'] == tid]
        else:
            trail_features = gdf.loc[[tid]]
        merged_geom = unary_union(trail_features.geometry.tolist())
        # Ensure the merged geometry is a line.
        merged_line = ensure_line(merged_geom)
        # Smooth the merged line.
        smoothed_geom = strong_smooth_geometry(merged_line, smoothing_factor=50.0, num_points=50,
                                               simplify_tolerance=0.8)

        # Generate the base variant.
        base_variant, base_angle, base_scale = generate_variant(smoothed_geom, variant_type='base')
        records.append({
            "orig_id": tid,
            "variant": "base",
            "new_id": new_id,
            "rotation": base_angle,
            "scale": base_scale,
            "geometry": base_variant
        })
        new_id += 1

        # Generate 9 augmented variants.
        for v in range(9):
            aug_variant, angle, scale_factor = generate_variant(smoothed_geom, variant_type='augmented')
            records.append({
                "orig_id": tid,
                "variant": f"augmented_{v + 1}",
                "new_id": new_id,
                "rotation": angle,
                "scale": scale_factor,
                "geometry": aug_variant
            })
            new_id += 1

    # Create a GeoDataFrame with the new variants.
    gdf_out = gpd.GeoDataFrame(records, crs=gdf.crs)
    gdf_out.to_file(output_gpkg, driver="GPKG")
    print(f"Saved {len(gdf_out)} variant features to {output_gpkg}")
    print(f"Number of unique trails in input: {len(unique_ids)}")
    print(f"Number of unique trails in output: {len(gdf_out['orig_id'].unique())}")


if __name__ == "__main__":
    main()
