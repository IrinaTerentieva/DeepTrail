import os
import glob
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from skimage.morphology import remove_small_objects

# Paths
footprint_fp = "/media/irina/My Book1/Conoco/DATA/footprint/manual_conoco.gpkg"
orthos_folder = "/media/irina/My Book1/Conoco/DATA/Orthos/3_Orthomosaics_10cm"
output_folder = os.path.join(orthos_folder, "shadows")
os.makedirs(output_folder, exist_ok=True)

# Read the footprint GeoPackage and combine its geometries.
footprint_gdf = gpd.read_file(footprint_fp)
footprint_geom = footprint_gdf.unary_union

# List all .tif files in the Orthos folder.
tif_files = glob.glob(os.path.join(orthos_folder, "*.tif"))

for tif_fp in tif_files:
    print("Processing:", tif_fp)

    # Open the source raster.
    with rasterio.open(tif_fp) as src:
        # Mask the raster with the footprint geometry. Areas outside become nodata.
        masked_image, masked_transform = mask(src, [footprint_geom], crop=True, nodata=src.nodata)
        # Copy metadata and update dimensions.
        meta = src.meta.copy()
        meta.update({
            "height": masked_image.shape[1],
            "width": masked_image.shape[2],
            "transform": masked_transform
        })

    # Compute shadow index: simple formula (1 minus the mean across bands).
    # (Adjust this formula as needed for your data.)
    mean_band = np.mean(masked_image, axis=0)
    shadow_index = 1 - mean_band
    print("Shadow index computed")

    # Create binary mask:
    # Set pixels with shadow_index values between -1000 and -100 to 0, the rest to 1.
    binary_mask = np.where((shadow_index >= -1000) & (shadow_index <= -100), 0, 1)

    # Remove small connected objects (areas) with fewer than 5 pixels.
    cleaned_mask = remove_small_objects(binary_mask.astype(bool), min_size=5)
    # Convert the boolean mask back to uint8 (0 and 1)
    cleaned_mask = cleaned_mask.astype(np.uint8)
    print("Applied binary threshold and filtered out small areas")

    # Prepare output metadata for a single-band raster.
    out_meta = meta.copy()
    out_meta.update({
        "count": 1,
        "dtype": cleaned_mask.dtype
    })

    # Construct the output filename.
    base_name = os.path.splitext(os.path.basename(tif_fp))[0]
    out_tif = os.path.join(output_folder, f"{base_name}_shadow_mask.tif")

    # Save the binary shadow mask raster.
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(cleaned_mask, 1)

    print("Saved shadow mask to:", out_tif)

print("Processing complete.")
