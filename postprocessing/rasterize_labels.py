import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import os


def rasterize_lines_to_match_original(geopkg_path, original_raster_path, output_raster_path, buffer_size_meters=1.0):
    # Read the GeoPackage file
    gdf = gpd.read_file(geopkg_path)

    # Filter out invalid or empty geometries
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]

    if gdf.empty:
        print(f"No valid geometries in {geopkg_path}. Skipping.")
        return

    # Apply a buffer of 1 meter (2 pixels at 50 cm resolution)
    gdf['geometry'] = gdf.buffer(buffer_size_meters)

    # Open the original raster to match its resolution, extent, and metadata
    with rasterio.open(original_raster_path) as src:
        profile = src.profile  # Copy the metadata of the original raster
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
        original_shape = (height, width)

        # Create the shapes for rasterization (geometry, value)
        shapes = [(geom, 1) for geom in gdf.geometry if geom.is_valid and not geom.is_empty]

        # Rasterize the buffered lines to match the original raster's resolution
        rasterized_lines = rasterize(
            shapes=shapes,
            out_shape=original_shape,
            transform=transform,
            fill=0,
            dtype='uint8'
        )

        # Save the rasterized lines using the original raster's profile
        profile.update(dtype=rasterized_lines.dtype, count=1)

        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(rasterized_lines, 1)

    print(f"Rasterized output saved to {output_raster_path}")


def process_geopackages_and_rasters(geopkg_folder, original_raster_folder, output_folder, buffer_size_meters=1.0):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for geopkg_file in os.listdir(geopkg_folder):
        if geopkg_file.endswith('_connected.gpkg'):
            base_name = os.path.splitext(geopkg_file)[0].replace('_connected', '')
            original_raster_file = os.path.join(original_raster_folder, f"{base_name}.tif")
            geopkg_path = os.path.join(geopkg_folder, geopkg_file)

            if os.path.exists(original_raster_file):
                print(f"Processing: {geopkg_path} with {original_raster_file}")

                # Set the output .tif file path
                output_tif = os.path.join(output_folder, f"{base_name}_connected_rasterized.tif")

                # Rasterize the GeoPackage to match the original raster
                rasterize_lines_to_match_original(geopkg_path, original_raster_file, output_tif, buffer_size_meters)
            else:
                print(f"Original raster file not found for {geopkg_file}")


def main():
    # Paths for the input centerline files, corresponding rasters, and output directory
    geopkg_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm/connected_segment'
    original_raster_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm'
    output_folder = '/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm/connected_raster'

    process_geopackages_and_rasters(geopkg_folder, original_raster_folder, output_folder, buffer_size_meters=0.2)


if __name__ == "__main__":
    main()
