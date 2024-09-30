import yaml
import os
import warnings
import rasterio
import numpy as np
import tensorflow as tf
from rasterio.windows import Window
from tqdm import tqdm
import sys
sys.path.append('/media/irro/All/HumanFootprint/Models/custom_unet')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metaflow import FlowSpec, step
from custom_unet import custom_unet
from utils import process_block, adjust_window, pad_to_shape, predict_patch_optimized, save_predictions

# Suppress TensorFlow and CUDA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Load YAML config
def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class UNetPredictionFlow(FlowSpec):

    environment = 'local'  # Can be 'hpc' if running on HPC
    username = 'irina.terenteva'

    @property
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
        Load the configuration and set up paths.
        """
        self.config_path = os.path.join(self.base_dir, 'config.yaml')
        print(f"Loading configuration from: {self.config_path}")

        # Load YAML configuration
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.test_image_path = os.path.join(self.base_dir, self.config['prediction_params']['test_image_path'])
        self.output_dir = os.path.join(self.base_dir, self.config['prediction_params']['output_dir'])
        self.model_path = os.path.join(self.base_dir, self.config['prediction_params']['model_path'])
        self.patch_size = self.config['prediction_params']['patch_size']
        self.overlap_size = self.config['prediction_params']['overlap_size']
        self.max_height = self.config['project']['max_height']
        self.next(self.load_model)

    @step
    def load_model(self):
        """
        Load the UNet model with the saved weights.
        """
        config = self.config

        # Initialize the UNet model
        input_shape = (config['model']['custom_unet']['patch_size'], config['model']['custom_unet']['patch_size'], 1)
        self.model = custom_unet(input_shape=input_shape, filters=config['model']['custom_unet']['filters'],
                                 use_batch_norm=config['model']['custom_unet']['use_batch_norm'],
                                 dropout=config['model']['custom_unet']['dropout'],
                                 num_classes=config['model']['custom_unet']['num_classes'],
                                 output_activation=config['model']['custom_unet']['output_activation'])

        # Load model weights
        print(f"Loading model weights from: {self.model_path}")
        self.model.load_weights(self.model_path)
        self.next(self.predict)

    @step
    def predict(self):
        """
        Perform sliding window prediction on the test image and save the result.
        """
        config = self.config

        print(f"Starting prediction on {self.test_image_path} using a sliding window.")

        # Perform prediction with a sliding window
        self.output_path = self.run_prediction_sliding_window(
            model=self.model,
            input_tif_path=self.test_image_path,
            patch_size=self.patch_size,
            overlap_size=self.overlap_size
        )

        self.next(self.save_prediction)

    def run_prediction_sliding_window(self, model, input_tif_path, patch_size, overlap_size):
        """
        Predict with sliding window and optimized memory usage for large TIF files.
        """
        stride = patch_size - overlap_size

        with rasterio.open(input_tif_path) as src:
            original_height, original_width = src.shape
            pred_accumulator = np.zeros((original_height, original_width), dtype=np.uint8)
            counts = np.zeros((original_height, original_width), dtype=np.uint16)

            for i in tqdm(range(0, original_height, stride)):
                for j in range(0, original_width, stride):
                    window = Window(j, i, patch_size, patch_size)
                    window = adjust_window(window, original_width, original_height)
                    patch = src.read(window=window)
                    patch = np.moveaxis(patch, 0, -1)

                    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                        patch = pad_to_shape(patch, (patch_size, patch_size))

                    pred_patch = predict_patch_optimized(model, patch)
                    col_off, row_off, width, height = map(int, window.flatten())
                    pred_patch = pred_patch[:height, :width]
                    pred_accumulator[row_off:row_off+height, col_off:col_off+width] = np.maximum(pred_accumulator[row_off:row_off+height, col_off:col_off+width], pred_patch)
                    counts[row_off:row_off+height, col_off:col_off+width] += 1

            final_pred = pred_accumulator

            # Save predicted image
            output_path = input_tif_path[:-4] + f'_predicted_{overlap_size}.tif'
            profile = src.profile.copy()
            profile.update(dtype='uint8', nodata=0, compress='LZW', tiled=True, blockxsize=64, blockysize=64)

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(final_pred[np.newaxis, :, :])

        return output_path

    @step
    def save_prediction(self):
        """
        Save the prediction output as a GeoTIFF.
        """
        output_path = os.path.join(self.output_dir, 'final_predictions.tif')
        print(f"Prediction saved successfully at {self.output_path}")
        self.next(self.end)

    @step
    def end(self):
        """
        End the prediction flow.
        """
        print("Prediction flow completed successfully!")

if __name__ == '__main__':
    UNetPredictionFlow()
