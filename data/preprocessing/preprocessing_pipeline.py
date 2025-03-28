# cd /media/irro/All/LineFootprint/Scripts/preprocessing
# python preprocessing_pipeline.py run

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from metaflow import FlowSpec, step
import yaml
import os
import pandas as pd

class PreprocessingFlow(FlowSpec):
    """
    This class handles preprocessing steps including:
    - Loading input files
    - Buffering centerlines
    - Merging layers
    - Rasterizing the output
    """

    # Fetch environment from environment variable
    environment = os.getenv('ENVIRONMENT', 'local')  # Default to 'local' if not set
    username = 'irina.terenteva'

    def base_dir(self):
        """
        Dynamically set the base directory depending on environment.
        """
        if self.environment == 'hpc':
            return f'/home/{self.username}/LineFootprint/'
        else:
            return '/media/irro/All/LineFootprint/'

    @step
    def start(self):
        """
        Start step: Load the configuration settings from the central config.yaml file.
        """
        # Load the config file dynamically
        self.config_path = os.path.join(self.base_dir(), 'config.yaml')
        print(f"Loading configuration from {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"Configuration loaded with settings:\n{self.config['preprocessing']}")
        self.next(self.load_files)

    @step
    def load_files(self):
        """
        Load the input files (canopy footprint and centerline).
        Logs key details such as CRS and input filenames.
        """
        print("Loading input files...")

        # Load the canopy footprint and centerline
        canopy_path = os.path.join(self.base_dir(), self.config['preprocessing']['input_files']['canopy_footprint'])
        centerline_path = os.path.join(self.base_dir(), self.config['preprocessing']['input_files']['centerline'])

        print(f"Reading canopy footprint from: {canopy_path}")
        print(f"Reading centerline from: {centerline_path}")

        self.canopy_footprint = gpd.read_file(canopy_path)
        self.centerline = gpd.read_file(centerline_path)

        # Extract filenames and log the coordinate reference system
        self.canopy_filename = os.path.basename(canopy_path).replace('.gpkg', '')
        self.centerline_filename = os.path.basename(centerline_path).replace('.gpkg', '')

        print(f"Loaded canopy footprint ({self.canopy_filename}) and centerline ({self.centerline_filename})")
        print(f"Coordinate reference system (CRS): {self.canopy_footprint.crs}")
        self.next(self.buffer_centerline)

    @step
    def buffer_centerline(self):
        """
        Buffer the centerline based on the buffer distance from the configuration.
        Saves and logs the buffered centerline.
        """
        buffer_distance = self.config['preprocessing']['buffer_distance']
        print(f"Buffering centerline by {buffer_distance} meters...")

        # Buffer the centerline
        self.buffered_centerline = self.centerline.buffer(buffer_distance)
        self.buffered_centerline = gpd.GeoDataFrame(geometry=self.buffered_centerline, crs=self.centerline.crs)

        # Save buffered output
        buffer_output_path = os.path.join(self.base_dir(), self.config['preprocessing']['output_dir']['intermediate'],
                                          f"{self.centerline_filename}_buffered_{buffer_distance}m.gpkg")
        print(f"Saving buffered centerline to: {buffer_output_path}")
        self.buffered_centerline.to_file(buffer_output_path, driver="GPKG")

        print(f"Buffered centerline saved at {buffer_output_path}")
        self.next(self.combine_layers)

    @step
    def combine_layers(self):
        """
        Combine the buffered centerline and the canopy footprint.
        Save the combined GeoDataFrame for further steps.
        """
        print("Combining buffered centerline and canopy footprint...")

        # Merge the buffered centerline and canopy footprint
        self.combined_gdf = gpd.GeoDataFrame(
            pd.concat([self.buffered_centerline, self.canopy_footprint], ignore_index=True))

        # Save merged output
        self.merged_filename = f"{self.centerline_filename}_with_{self.canopy_filename}.gpkg"
        merged_output_path = os.path.join(self.base_dir(), self.config['preprocessing']['output_dir']['intermediate'],
                                          self.merged_filename)
        print(f"Saving combined layers to: {merged_output_path}")
        self.combined_gdf.to_file(merged_output_path, driver="GPKG")

        print(f"Combined GeoDataFrame saved at {merged_output_path}")
        self.next(self.rasterize_combined)

    @step
    def rasterize_combined(self):
        """
        Rasterize the combined GeoDataFrame.
        Raster properties such as pixel size are derived from the configuration file.
        """
        print("Rasterizing combined footprint...")

        # Get bounds and pixel size from config
        bounds = self.canopy_footprint.total_bounds
        pixel_size = self.config['preprocessing']['rasterization']['pixel_size']
        print(f"Rasterizing with pixel size {pixel_size}...")

        # Compute transform and raster shape
        transform = rasterio.transform.from_bounds(*bounds,
                                                   width=int((bounds[2] - bounds[0]) / pixel_size),
                                                   height=int((bounds[3] - bounds[1]) / pixel_size))
        raster_shape = (int((bounds[3] - bounds[1]) / pixel_size), int((bounds[2] - bounds[0]) / pixel_size))

        # Rasterize combined GeoDataFrame
        rasterized = rasterize(
            [(geom, 1) for geom in self.combined_gdf.geometry],
            out_shape=raster_shape,
            transform=transform,
            fill=0,
            dtype=rasterio.uint8
        )

        # Save rasterized output
        self.output_raster_path = os.path.join(self.base_dir(), self.config['preprocessing']['output_dir']['intermediate'],
                                               f"{self.merged_filename[:-5]}.tif")
        print(f"Saving raster to: {self.output_raster_path}")
        with rasterio.open(self.output_raster_path, 'w', driver='GTiff', height=raster_shape[0], width=raster_shape[1],
                           count=1, dtype=rasterio.uint8, transform=transform) as dst:
            dst.write(rasterized, 1)

        self.config['preprocessing']['rasterization']['rasterized_canopy_footprint'] = self.output_raster_path

        # Update the config file with the new raster path
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config, f)

        print(f"Raster saved at {self.output_raster_path}")
        self.next(self.end)

    @step
    def end(self):
        """
        Final step: Indicate that preprocessing has completed and print the output raster path.
        """
        print(f"Rasterized output is ready for the next steps: {self.output_raster_path}")
        print("Preprocessing pipeline completed successfully.")

if __name__ == '__main__':
    PreprocessingFlow()

