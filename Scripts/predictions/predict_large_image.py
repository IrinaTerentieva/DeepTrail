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
from metaflow import FlowSpec, step
from custom_unet import custom_unet
from utils import adjust_window, pad_to_shape, normalize_image, predict_patch_optimized
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
    - Saves output as a GeoTIFF
    """

    environment = 'local'
    username = 'irina.terenteva'

    def base_dir(self):
        """
        Dynamically set the base directory depending on the environment.
        """
        if self.environment == 'hpc':
            return f'/home/{self.username}/HumanFootprint/'
        else:
            return '/media/irro/All/HumanFootprint/'

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

        print(f"Model Path: {self.model_path}")
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

        self.next(self.predict)

    @step
    def predict(self):
        """
        Perform sliding window prediction on the large input image.
        """
        stride = self.patch_size - self.overlap_size

        # Perform sliding window prediction
        with rasterio.open(self.input_image_path) as src:
            original_height, original_width = src.shape
            pred_accumulator = np.zeros((original_height, original_width), dtype=np.uint8)
            counts = np.zeros((original_height, original_width), dtype=np.uint16)

            print(f"Starting sliding window prediction on image of size {original_height}x{original_width}...")
            for i in tqdm(range(0, original_height, stride)):
                for j in range(0, original_width, stride):
                    window = Window(j, i, self.patch_size, self.patch_size)
                    window = adjust_window(window, original_width, original_height)

                    # Read the patch
                    patch = src.read(window=window)
                    patch = np.moveaxis(patch, 0, -1)  # Change to HWC format

                    # Pad if necessary
                    if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                        patch = pad_to_shape(patch, (self.patch_size, self.patch_size))

                    # Predict the patch
                    pred_patch = predict_patch_optimized(self.model, patch)

                    # Unpad the patch back to its original size
                    col_off, row_off, width, height = map(int, window.flatten())
                    pred_patch = pred_patch[:height, :width]

                    # Accumulate predictions
                    pred_accumulator[row_off:row_off + height, col_off:col_off + width] = np.maximum(
                        pred_accumulator[row_off:row_off + height, col_off:col_off + width], pred_patch)
                    counts[row_off:row_off + height, col_off:col_off + width] += 1

            # Final predicted image
            self.final_pred = pred_accumulator
            print(f"Sliding window prediction completed.")

        self.next(self.save_output)

    @step
    def save_output(self):
        """
        Save the predicted output as a GeoTIFF.
        """
        output_path = os.path.join(self.output_dir, 'predicted_image.tif')

        with rasterio.open(self.input_image_path) as src:
            profile = src.profile.copy()
            profile.update(dtype='uint8', compress='LZW')

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(self.final_pred[np.newaxis, :, :])  # Save single-band prediction

        print(f"Predicted output saved to: {output_path}")

        self.next(self.end)

    @step
    def end(self):
        """
        Final step to log the completion of the prediction flow.
        """
        print("UNet prediction flow completed successfully.")


if __name__ == '__main__':
    UNetPredictionFlow()
