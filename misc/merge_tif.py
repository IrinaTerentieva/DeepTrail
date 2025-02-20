import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.crs import CRS

# Define the folder with your TIFFs and the output filename
input_folder = "/media/irina/My Book/Surmont/nDTM"
output_filename = "/media/irina/My Book/Surmont/Surmont_nDTM10cm.tif"

# Find all TIFF files in the folder
tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
print(f"Found {len(tif_files)} TIFF files.")

# Open all the TIFF files as rasterio datasets
src_files_to_mosaic = []
for fp in tif_files:
    src = rasterio.open(fp)
    src_files_to_mosaic.append(src)

# Merge them into a mosaic
mosaic, out_trans = merge(src_files_to_mosaic)

# Get metadata from the first file and update for the mosaic
out_meta = src_files_to_mosaic[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "crs": CRS.from_epsg(2956)
})

# Save the mosaic to disk
with rasterio.open(output_filename, "w", **out_meta) as dest:
    dest.write(mosaic)

print(f"Mosaic saved to: {output_filename}")
