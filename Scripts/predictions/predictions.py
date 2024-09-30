import yaml
import os
import warnings
import rasterio
import numpy as np
import tensorflow as tf
import wandb
import sys
sys.path.append('/media/irro/All/HumanFootprint/Models/custom_unet')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metaflow import FlowSpec, step
from custom_unet import custom_unet
from utils import sliding_window_prediction, save_predictions

# cd /media/irro/All/HumanFootprint/Scripts/predictions
# python predictions.py run

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
        self.stride = self.config['prediction_params']['overlap_size']
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

        # Perform sliding window prediction
        self.full_prediction = sliding_window_prediction(
            image_path=self.test_image_path,
            model=self.model,
            patch_size=self.patch_size,
            stride=self.stride,
            max_height=self.max_height
        )

        self.next(self.save_prediction)

    @step
    def save_prediction(self):
        """
        Save the prediction output as a GeoTIFF.
        """
        output_path = os.path.join(self.output_dir, 'predictions.tif')

        # Save the predictions
        print(f"Saving predictions to {output_path}")
        save_predictions(
            full_prediction=self.full_prediction,
            output_path=output_path,
            reference_image_path=self.test_image_path,
            scale_factor=100,
            config=self.config
        )

        print(f"Prediction saved successfully at {output_path}")
        self.next(self.end)

    @step
    def end(self):
        """
        End the prediction flow.
        """
        print("Prediction flow completed successfully!")


if __name__ == '__main__':
    UNetPredictionFlow()
