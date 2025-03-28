# cd /media/irro/All/LineFootprint/Scripts/preprocessing
# python check_patches.py run

import os
import yaml
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from metaflow import FlowSpec, step

class VisualizePairsFlow(FlowSpec):
    """
    This flow visualizes image-label pairs, checks data value distributions, and analyzes label balance.
    It is designed to use the paths from `config.yaml`.
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
        Load configuration from `config.yaml` and set up paths for images and labels.
        """
        config_path = os.path.join(self.base_dir(), 'config.yaml')
        print(f"Loading configuration from {config_path}")

        # Load the config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set directories for data (joined with base_dir)
        self.patches_dir = os.path.join(self.base_dir(), self.config['preprocessing']['patch_extraction']['patches'])
        self.output_dir = os.path.join(self.patches_dir, 'Figures')  # Figures in patches directory

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory for figures: {self.output_dir}")

        print(f"Patches Directory: {self.patches_dir}")
        print(f"Figures Output Directory: {self.output_dir}")

        self.next(self.visualize_pairs)

    @step
    def visualize_pairs(self):
        """
        Visualize random image-label pairs and check data distributions.
        """
        patch_files = [f for f in os.listdir(self.patches_dir) if f.endswith('_image.tif')]
        num_files = len(patch_files)
        print(f"Found {num_files} patches.")

        # Visualize and analyze the first 5 image-label pairs
        for patch_file in patch_files[:5]:
            image_path = os.path.join(self.patches_dir, patch_file)
            label_path = image_path.replace('_image.tif', '_label.tif')

            # Read image and label
            with rasterio.open(image_path) as img_src, rasterio.open(label_path) as label_src:
                image = img_src.read(1)
                label = label_src.read(1)

                # Plot image-label pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.set_title(f'Image: {os.path.basename(image_path)}')
                ax1.imshow(image, cmap='gray')
                ax2.set_title(f'Label: {os.path.basename(label_path)}')
                ax2.imshow(label, cmap='gray')
                plt.tight_layout()

                # Save figure
                fig_name = f"visualized_{os.path.basename(image_path).replace('.tif', '')}.png"
                plt.savefig(os.path.join(self.output_dir, fig_name))
                print(f"Saved visualization: {fig_name}")
                plt.close()

            # Print distribution statistics
            print(f"Image Stats (Mean, Min, Max): {np.mean(image):.2f}, {np.min(image):.2f}, {np.max(image):.2f}")
            print(f"Label Distribution: {Counter(label.flatten())}")

        self.next(self.check_balance)

    @step
    def check_balance(self):
        """
        Check label balance and pixel distributions for all patches.
        """
        patch_files = [f for f in os.listdir(self.patches_dir) if f.endswith('_label.tif')]
        label_counts = Counter()
        image_distributions = []

        for label_file in tqdm(patch_files, desc="Analyzing Labels"):
            label_path = os.path.join(self.patches_dir, label_file)
            image_path = label_path.replace('_label.tif', '_image.tif')

            # Read label and image
            with rasterio.open(label_path) as label_src, rasterio.open(image_path) as img_src:
                label = label_src.read(1)
                image = img_src.read(1)

                # Update label balance
                label_counts.update(Counter(label.flatten()))

                # Collect image distributions (for up to 10 images)
                if len(image_distributions) < 10:
                    image_distributions.append(image.flatten())

        # Plot distributions of pixel values
        plt.figure(figsize=(10, 6))
        for dist in image_distributions:
            plt.hist(dist, bins=50, alpha=0.5, label="Pixel Values")
        plt.title("Distribution of Pixel Values in Patches")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Save distribution plot
        dist_fig_name = "pixel_value_distribution.png"
        plt.savefig(os.path.join(self.output_dir, dist_fig_name))
        print(f"Saved distribution plot: {dist_fig_name}")
        plt.close()

        # Print overall label balance
        print(f"Label balance across all patches: {label_counts}")

        self.next(self.end)

    @step
    def end(self):
        """
        End of the flow. Report completion.
        """
        print("Visualization and analysis completed successfully.")


if __name__ == '__main__':
    VisualizePairsFlow()



