import os
import sys

sys.path.append('/media/irro/All/HumanFootprint/Models/custom_unet')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import warnings
import numpy as np
import rasterio
import tensorflow as tf
from tqdm import tqdm
from metaflow import FlowSpec, step, Parameter
from custom_unet import custom_unet
from utils import adjust_window, pad_to_shape, normalize_image, calculate_statistics
from rasterio.windows import Window

# Suppress TensorFlow and CUDA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class UNetPredictionFlow(FlowSpec):
    """
    This flow handles the prediction of a large image using a sliding window approach:
    - Loads model
    - Reads large image
    - Predicts output using sliding window
    - Saves output as a GeoTIFF with improved efficiency by batching predictions.
    """

    environment = 'local'
    username = 'irina.terenteva'
    batch_size = Parameter('batch_size', default=8, help="Batch size for predicting patches.")
    verbose = Parameter('verbose', default=False, help="Set to True to print debugging information like shapes.")

    def base_dir(self):
        """
        Dynamically set the base directory depending on the environment.
        """
        if self.environment == 'hpc':
            return f'/home/{self.username}/HumanFootprint/'
        else:
            return '/media/irro/All/HumanFootprint/'

    def print_verbose(self, message):
        """
        Print debugging information only if verbosity is enabled.
        """
        if self.verbose:
            print(message)

    @step
    def start(self):
        """
        Start step: Load the configuration settings from the central config.yaml file.
        Log the start time and initial settings.
        """
        # Load the config file
        config_path = os.path.join(self.base_dir(), 'config.yaml')
        print(f"Loading configuration from {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Ensure proper formatting of paths based on config
        self.model_path = os.path.join(self.base_dir(), self.config['prediction_params']['model_path'])
        self.input_image_path = os.path.join(self.base_dir(), self.config['prediction_params']['test_image_path'])
        self.output_dir = os.path.join(self.base_dir(), self.config['prediction_params']['output_dir'])
        self.patch_size = self.config['prediction_params']['patch_size']
        self.overlap_size = self.config['prediction_params']['overlap_size']

        print(f"\n******\nModel Path: {self.model_path}")
        print(f"Input Image Path: {self.input_image_path}")
        print(f"Output Directory: {self.output_dir}")

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        self.next(self.load_model)

    @step
    def load_model(self):
        """
        Load the trained UNet model for prediction.
        """
        config = self.config['model']['custom_unet']

        input_shape = (config['patch_size'], config['patch_size'], config['num_bands'])
        print(f"Loading model from: {self.model_path}")

        self.model = custom_unet(
            input_shape=input_shape,
            filters=config['filters'],
            use_batch_norm=config['use_batch_norm'],
            dropout=config['dropout'],
            num_classes=config['num_classes'],
            output_activation=config['output_activation'],
            num_layers=config['num_layers']
        )
        self.model.load_weights(self.model_path)
        print("Model loaded successfully.")

        self.next(self.predict_and_save)

    def preprocess_image(self, patch):
        """
        Apply the same preprocessing steps used in training to the input image patch.
        This includes resizing and normalization.
        """
        # Normalize the image patch as done in the data generator
        patch = normalize_image(patch)
        patch = patch.reshape(1, patch.shape[0], patch.shape[1], patch.shape[2])

        # Print shape for debugging
        self.print_verbose(f"Preprocessed patch shape: {patch.shape}")
        return patch

    @step
    def predict_and_save(self):
        """
        Perform sliding window prediction on the large input image and save incrementally to disk.
        Using batches for improved performance.
        """
        stride = self.patch_size - self.overlap_size

        # Extract the base name of the input image
        base_name = os.path.splitext(os.path.basename(self.input_image_path))[0]
        model_name = os.path.basename(self.config['prediction_params']['model_path'])[:-3]
        output_path = os.path.join(self.output_dir, f"{base_name}_{model_name}.tif")
        print('Save to output path: ', output_path)

        # Open the input image and the output file for saving predictions
        with rasterio.open(self.input_image_path) as src:
            original_height, original_width = src.shape
            profile = src.profile.copy()
            profile.update(dtype='uint8', compress='LZW', nodata=0)  # Set nodata value to 0

            # Batch accumulation
            patches, coords = [], []

            with rasterio.open(output_path, 'w', **profile) as dst:
                print(f"Starting sliding window prediction on image of size {original_height}x{original_width}...")

                # Process each patch and write predictions in batches
                for i in tqdm(range(0, original_height, stride)):
                    for j in range(0, original_width, stride):
                        window_width = min(self.patch_size, original_width - j)
                        window_height = min(self.patch_size, original_height - i)

                        # Ensure window is within image boundaries
                        window = Window(j, i, window_width, window_height)

                        # Read the patch
                        patch = src.read(window=window)
                        patch = np.moveaxis(patch, 0, -1)  # Change to HWC format

                        # Pad if necessary
                        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                            patch = pad_to_shape(patch, (self.patch_size, self.patch_size))

                        # Preprocess the patch (resize and normalize)
                        patch_norm = self.preprocess_image(patch)

                        patches.append(patch_norm)
                        coords.append((i, j, window, patch.shape[0], patch.shape[1]))

                        # If we reached the batch size, process the batch
                        if len(patches) == self.batch_size:
                            self.process_batch_and_save(patches, coords, dst)
                            patches, coords = [], []

                # Process any remaining patches
                if patches:
                    self.process_batch_and_save(patches, coords, dst)

                print(f"Sliding window prediction completed. Predictions saved to {output_path}.")

        self.next(self.end)

    def process_batch_and_save(self, patches, coords, dst):
        """
        Process a batch of patches and save the predictions to the output file.
        """
        patches = np.concatenate(patches, axis=0)
        preds = self.model.predict(patches)

        for idx, pred_patch in enumerate(preds):
            pred_patch = (pred_patch * 100).squeeze().astype(np.uint8)
            i, j, window, height, width = coords[idx]

            # Adjust the window size to ensure it does not go out of bounds
            window_width = min(self.patch_size, dst.width - j)
            window_height = min(self.patch_size, dst.height - i)

            # Unpad the patch back to its original size
            pred_patch = pred_patch[:window_height, :window_width]

            # Write the prediction directly to the output file
            dst.write(pred_patch[np.newaxis, :, :], window=Window(j, i, window_width, window_height))

    @step
    def end(self):
        """
        Final step to log the completion of the prediction flow.
        """
        print("UNet prediction flow completed successfully.")


if __name__ == '__main__':
    UNetPredictionFlow()
