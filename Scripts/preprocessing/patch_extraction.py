import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import yaml
from metaflow import FlowSpec, step


class PatchExtractionFlow(FlowSpec):
    """
    This flow handles patch extraction:
    - Loads points file
    - Buffers points
    - Extracts CHM and rasterized canopy footprint patches
    """

    # Fetch environment from environment variable
    environment = os.getenv('ENVIRONMENT', 'local')  # Default to 'local' if not set
    username = 'irina.terenteva'

    def base_dir(self):
        """
        Dynamically set the base directory depending on environment.
        """
        if self.environment == 'hpc':
            return f'/home/{self.username}/HumanFootprint/'
        else:
            return '/media/irro/All/HumanFootprint/'

    @step
    def start(self):
        """
        Start step: Load the configuration settings from the central config.yaml file.
        """
        # Load the config file
        config_path = os.path.join(self.base_dir(), 'config.yaml')
        print(f"Loading configuration from {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"Configuration loaded with settings:\n{self.config['preprocessing']['patch_extraction']}")
        self.next(self.load_points)

    @step
    def load_points(self):
        """
        Load the points file and prepare for patch extraction.
        Logs key details such as point count and output directories.
        """
        print("Loading points file...")
        points_path = os.path.join(self.base_dir(), self.config['preprocessing']['input_files']['points'])
        self.points = gpd.read_file(points_path)
        print(f"Loaded points file from: {points_path}")

        # Ensure output directory exists
        self.output_dir = os.path.join(self.base_dir(), self.config['preprocessing']['patch_extraction']['patches'])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        # Select random points based on n_random_points from config
        n_random_points = self.config['preprocessing']['patch_extraction']['n_random_points']
        if n_random_points < len(self.points):
            print(f"Selecting {n_random_points} random points from {len(self.points)} total points.")
            self.points = self.points.sample(n=n_random_points, random_state=42)
        else:
            print(f"Using all {len(self.points)} points for patch extraction.")

        self.next(self.extract_patches)

    @step
    def extract_patches(self):
        """
        Extract patches around each random point and save CHM and label raster data.
        """
        # Define buffer size in terms of pixels
        patch_size_pixels = self.config['preprocessing']['patch_extraction']['patch_size_pixels']
        print(f"Processing patches with size: {patch_size_pixels} pixels...")

        # Open rasters only in this step (to avoid pickling issues)
        chm_path = os.path.join(self.base_dir(), self.config['preprocessing']['input_files']['chm'])
        rasterized_canopy_path = os.path.join(self.base_dir(), self.config['preprocessing']['rasterization']['rasterized_canopy_footprint'])
        print(f"Reading CHM from: {chm_path}")
        print(f"Reading rasterized canopy from: {rasterized_canopy_path}")

        with rasterio.open(chm_path) as chm, rasterio.open(rasterized_canopy_path) as cnn_raster:
            # Get the resolution of the raster (meters per pixel)
            resolution_x = abs(chm.transform[0])
            resolution_y = abs(chm.transform[4])

            # Calculate buffer size in meters based on pixel size
            buffer_size_meters_x = patch_size_pixels * resolution_x / 2
            buffer_size_meters_y = patch_size_pixels * resolution_y / 2
            print(f"Buffer size in meters: {buffer_size_meters_x} x {buffer_size_meters_y}")

            # Loop over each point to extract patches
            for ind, point_row in tqdm(self.points.iterrows(), total=self.points.shape[0]):
                # Create a buffered area around the point, adjusted by the resolution
                buffered_area = point_row.geometry.buffer(buffer_size_meters_x).envelope
                buffered_box = box(*buffered_area.bounds)
                chm_box = box(*chm.bounds)

                # Check if the buffered area is within CHM bounds
                if not chm_box.contains(buffered_box):
                    print(f"Skipping point {ind}, out of bounds.")
                    continue

                # Extract CHM data within the buffered area
                try:
                    chm_data, chm_transform = mask(chm, [buffered_area], crop=True)
                    # Keep original CHM values, no filtering out negative values
                except rasterio.errors.WindowError as e:
                    print(f"Skipping point {ind} due to CHM WindowError: {e}")
                    continue

                # Extract CNN raster data within the buffered area
                try:
                    cnn_data, cnn_transform = mask(cnn_raster, [buffered_area], crop=True)
                except rasterio.errors.WindowError as e:
                    print(f"Skipping point {ind} due to CNN WindowError: {e}")
                    continue

                # Handle NoData in labels
                cnn_nodata_value = cnn_raster.nodata
                label_raster = (cnn_data[0] > 0.5).astype(np.uint8)  # Threshold CNN output at 0.5
                if cnn_nodata_value is not None:
                    label_raster[cnn_data[0] == cnn_nodata_value] = 0  # Treat NoData as background (0)

                # Save CHM data
                output_image_path = os.path.join(self.output_dir, f"{ind}_image.tif")
                with rasterio.open(output_image_path, 'w', driver='GTiff', height=chm_data.shape[1],
                                   width=chm_data.shape[2], count=1, dtype=chm_data.dtype,
                                   transform=chm_transform) as out_image:
                    out_image.write(chm_data[0], 1)

                # Save label raster
                output_label_path = os.path.join(self.output_dir, f"{ind}_label.tif")
                with rasterio.open(output_label_path, 'w', driver='GTiff', height=label_raster.shape[0],
                                   width=label_raster.shape[1], count=1, dtype=label_raster.dtype,
                                   transform=cnn_transform) as out_label:
                    out_label.write(label_raster, 1)

        print("Patch extraction complete!")
        self.next(self.end)

    @step
    def end(self):
        """
        Final step: Log completion of the patch extraction process.
        """
        print(f"Patch extraction process completed successfully. Patches saved in: {self.output_dir}")


if __name__ == '__main__':
    PatchExtractionFlow()
