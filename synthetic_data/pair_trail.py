import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import os
import pandas as pd
import random

# --- Settings ---
smooth = True  # Set to True to smooth original lines before offsetting
max_width = 2

# Paths for the input vector and output name
input_trails = 'file:///media/irina/My Book1/Conoco/DATA/FLM/FLM_2022/Surmont_2022_centerline.shp'
name = 'Surmont'

# Define the local output folder and ensure it exists
output_folder = '/home/irina/HumanFootprint/DATA/manual/intermediate'
os.makedirs(output_folder, exist_ok=True)

# Read the original trails file
gdf = gpd.read_file(input_trails)

# Remove duplicate rows with the same geometry.
gdf = gdf[~gdf.geometry.apply(lambda g: g.wkt).duplicated()]
print("After duplicate removal, number of features:", len(gdf))
print("Geometry types:", gdf.geometry.geom_type.unique())

# Assign a unique trail_id to each original feature
gdf['trail_id'] = range(1, len(gdf) + 1)

# Generate a random offset distance (between 1 and max_width meters) for each feature.
gdf['offset_distance'] = gdf.geometry.apply(lambda geom: random.uniform(1, max_width))

# If smoothing is enabled, first smooth the original geometry.
if smooth:
    # Smooth each geometry using a tolerance equal to half its offset distance.
    gdf['geometry'] = gdf.apply(lambda row: row.geometry.simplify(row.offset_distance/2, preserve_topology=True), axis=1)
    # Optionally, fix any invalid geometries.
    gdf['geometry'] = gdf.geometry.apply(lambda g: g if g.is_valid else g.buffer(0))
    print("Original geometries have been smoothed.")

def create_parallel_line(geom, distance, side='left', smooth=False):
    """
    Create an offset (parallel) line from geom using the given distance.
    If smooth is True, smooth the resulting line by simplifying with tolerance=distance/2.
    Validity of the resulting geometry is checked and fixed if necessary.
    """
    try:
        if geom.geom_type == 'LineString':
            line = geom.parallel_offset(distance, side)
            if smooth and line is not None:
                line = line.simplify(distance/2, preserve_topology=True)
            if line is not None and not line.is_valid:
                line = line.buffer(0)
            return line
        elif geom.geom_type == 'MultiLineString':
            offset_lines = []
            for line in geom.geoms:
                off_line = line.parallel_offset(distance, side)
                if off_line is None:
                    continue
                if smooth:
                    if off_line.geom_type == 'MultiLineString':
                        off_line = MultiLineString([l.simplify(distance/2, preserve_topology=True) for l in off_line.geoms])
                    else:
                        off_line = off_line.simplify(distance/2, preserve_topology=True)
                # Check validity and fix if necessary.
                if off_line is not None:
                    if off_line.geom_type == 'MultiLineString':
                        off_line = MultiLineString([l if l.is_valid else l.buffer(0) for l in off_line.geoms])
                    else:
                        if not off_line.is_valid:
                            off_line = off_line.buffer(0)
                if off_line is not None:
                    if off_line.geom_type == 'MultiLineString':
                        offset_lines.extend(list(off_line.geoms))
                    else:
                        offset_lines.append(off_line)
            if offset_lines:
                result = MultiLineString(offset_lines)
                if not result.is_valid:
                    result = result.buffer(0)
                return result
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error offsetting geometry: {e}")
        return None

# Create the synthetic (offset) line using the random offset distance.
# When smooth==True, the original geometries are already smoothed, so we pass smooth=False here.
gdf['paired_line'] = gdf.apply(lambda row: create_parallel_line(row.geometry, row.offset_distance, side='left', smooth=False), axis=1)

# (Optional) Convert the paired lines to WKT for inspection.
gdf['paired_line_wkt'] = gdf['paired_line'].apply(lambda geom: geom.wkt if geom is not None else None)

# ---------------------
# Prepare synthetic (offset) features by exploding the MultiLineStrings.
offset_rows = []
for idx, geom in gdf['paired_line'].dropna().items():
    if geom.geom_type == 'MultiLineString':
        for subgeom in geom.geoms:
            offset_rows.append({'original_index': idx, 'offset_geom': subgeom})
    else:
        offset_rows.append({'original_index': idx, 'offset_geom': geom})

if offset_rows:
    gdf_offsets = gpd.GeoDataFrame(offset_rows, geometry='offset_geom', crs=gdf.crs)
    print("Exploded offset features:", len(gdf_offsets))
else:
    print("No offset geometries to explode.")
    gdf_offsets = gpd.GeoDataFrame(columns=["offset_geom"], crs=gdf.crs)

# Check if the number of synthetic lines is more than double the number of original features.
if len(gdf_offsets) > 2 * len(gdf):
    raise ValueError("Number of synthetic lines exceeds double the number of original features.")

# ---------------------
# Tag the original features with a source attribute.
gdf_orig = gdf.copy()
gdf_orig['source'] = 'original'
gdf_orig = gdf_orig.drop(columns=["paired_line", "paired_line_wkt"], errors="ignore")

# ---------------------
# For synthetic features, merge original attributes (except the geometry) using the original index.
gdf_orig_reset = gdf_orig.reset_index().rename(columns={'index': 'orig_index'})
gdf_orig_reset = gdf_orig_reset.drop(columns=["geometry"], errors='ignore')

gdf_offsets = gdf_offsets.merge(
    gdf_orig_reset,
    left_on="original_index",
    right_on="orig_index",
    how="left"
)

# Rename the synthetic geometry column to "geometry" and set it as the active geometry.
gdf_offsets = gdf_offsets.rename(columns={"offset_geom": "geometry"})
gdf_offsets = gdf_offsets.set_geometry("geometry")
gdf_offsets['source'] = 'synthetic'

# ---------------------
# Combine the original and synthetic features into one GeoDataFrame.
gdf_combined = pd.concat([gdf_orig, gdf_offsets], ignore_index=True)
gdf_combined = gpd.GeoDataFrame(gdf_combined, geometry='geometry', crs=gdf.crs)

# The column "offset_distance" remains in both original and synthetic features,
# so you can see which offset was used for each trail_id.
print("Columns in combined GeoDataFrame:", gdf_combined.columns.tolist())

# Save the combined GeoDataFrame to a GeoPackage.
output_path_combined = os.path.join(output_folder, f"{name}_synthetic_trails.gpkg")
gdf_combined.to_file(output_path_combined, driver="GPKG")
print("Combined file saved to:", output_path_combined)
