import geopandas as gpd
import glob
import os
import pandas as pd

# Define the folder containing the GeoPackage files
folder = "/media/irina/My Book1/Conoco/DATA/footprint"

# Use glob to list all .gpkg files in the folder
gpkg_files = glob.glob(os.path.join(folder, "*.gpkg"))

# List to hold individual GeoDataFrames
gdf_list = []

# Read each GeoPackage and append its GeoDataFrame to the list
for fp in gpkg_files:
    print(f"Processing {fp}")
    gdf = gpd.read_file(fp)
    gdf_list.append(gdf)

# Concatenate all GeoDataFrames into one
combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

# Optionally, set a CRS if needed
# combined_gdf = combined_gdf.set_crs("EPSG:xxxx")

# Define the output filepath for the combined GeoPackage
output_fp = os.path.join(folder, "/media/irina/My Book1/Conoco/DATA/footprint/manual_conoco.gpkg")

# Save the combined GeoDataFrame to a new GeoPackage
combined_gdf.to_file(output_fp, driver="GPKG")

print("Combined GeoPackage saved to:", output_fp)
