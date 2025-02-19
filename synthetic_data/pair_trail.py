import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import os
import pandas as pd

# Define the local output folder and ensure it exists
output_folder = '/home/irina/HumanFootprint/DATA/manual/intermediate'
os.makedirs(output_folder, exist_ok=True)

# Read the original wheel ruts file
gdf = gpd.read_file("file:///home/irina/HumanFootprint/DATA/manual/wheel_ruts.gpkg")

# Debug: Print number of features and geometry types
print("Original number of features:", len(gdf))
print("Geometry types:", gdf.geometry.geom_type.unique())

# Set the offset distance to 1 meter (assuming CRS units are in meters)
offset_distance = 1.0

def create_parallel_line(geom, distance, side='left'):
    try:
        if geom.geom_type == 'LineString':
            return geom.parallel_offset(distance, side)
        elif geom.geom_type == 'MultiLineString':
            offset_lines = []
            # Iterate over each component of the MultiLineString
            for line in geom.geoms:
                offset_line = line.parallel_offset(distance, side)
                if offset_line is None:
                    continue
                # Handle case where offset returns a MultiLineString
                if offset_line.geom_type == 'MultiLineString':
                    offset_lines.extend(list(offset_line.geoms))
                else:
                    offset_lines.append(offset_line)
            if offset_lines:
                return MultiLineString(offset_lines)
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error offsetting geometry: {e}")
        return None

# Apply the offset function to create a synthetic (offset) line for each geometry
gdf['paired_line'] = gdf.geometry.apply(lambda geom: create_parallel_line(geom, offset_distance, side='left'))

# (Optional) Convert the paired lines to WKT for inspection
gdf['paired_line_wkt'] = gdf['paired_line'].apply(lambda geom: geom.wkt if geom is not None else None)

# ---------------------
# Prepare synthetic (offset) features by exploding the MultiLineStrings
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

# ---------------------
# Tag the original features with a source attribute.
# Also drop extra geometry columns (paired_line and paired_line_wkt) that are not needed.
gdf_orig = gdf.copy()
gdf_orig['source'] = 'original'
gdf_orig = gdf_orig.drop(columns=["paired_line", "paired_line_wkt"], errors="ignore")

# ---------------------
# For synthetic features, merge original attributes (except geometry) using the original index.
# Reset the index on the original GeoDataFrame and drop its geometry column to avoid duplicates.
gdf_orig_reset = gdf_orig.reset_index().rename(columns={'index': 'orig_index'})
gdf_orig_reset = gdf_orig_reset.drop(columns=["geometry"], errors='ignore')

# Merge synthetic offsets with original attributes using the original index.
gdf_offsets = gdf_offsets.merge(
    gdf_orig_reset,
    left_on="original_index",
    right_on="orig_index",
    how="left"
)

# Rename the synthetic geometry column to "geometry" and set it as the active geometry.
gdf_offsets = gdf_offsets.rename(columns={"offset_geom": "geometry"})
gdf_offsets = gdf_offsets.set_geometry("geometry")

# Tag synthetic features with a source attribute.
gdf_offsets['source'] = 'synthetic'

# ---------------------
# Combine the original and synthetic features into one GeoDataFrame.
# Note: The original features already have a "geometry" column.
gdf_combined = pd.concat([gdf_orig, gdf_offsets], ignore_index=True)
gdf_combined = gpd.GeoDataFrame(gdf_combined, geometry='geometry', crs=gdf.crs)

# Save the combined GeoDataFrame to a geopackage
output_path_combined = os.path.join(output_folder, "combined_original_synthetic.gpkg")
gdf_combined.to_file(output_path_combined, driver="GPKG")

print("Combined file saved to:", output_path_combined)
