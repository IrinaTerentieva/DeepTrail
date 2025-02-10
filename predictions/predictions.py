import os
import sys
import gc
gc.collect()

sys.path.append('/home/irina/HumanFootprint/Models/custom_unet')
sys.path.append('/home/irina.terenteva/HumanFootprint/Models/custom_unet')
sys.path.append('/home/irina.terenteva/HumanFootprint/Models')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import warnings
import numpy as np
import rasterio
# import tensorflow as tf
from tqdm import tqdm
from metaflow import FlowSpec, step, Parameter
from custom_unet import custom_unet
from utils import adjust_window, pad_to_shape, normalize_image, calculate_statistics, sliding_window_prediction_tf, save_predictions_tf
from rasterio.windows import Window

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disables GPU usage

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
    verbose = Parameter('verbose', default=False, help="Set to True to print debugging information like shapes.")

    def base_dir(self):
        """
        Dynamically set the base directory depending on the environment.
        """
        if self.environment == 'hpc':
            return f'/home/{self.username}/HumanFootprint/'
        else:
            return '/home/irina/HumanFootprint/'

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
        print(f"Loading configuration from {config_path}", flush=True)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Ensure proper formatting of paths based on config
        self.model_path = os.path.join(self.base_dir(), self.config['prediction_params']['model_path'])

        self.input_image_path = self.config['prediction_params']['test_image_path']


        # self.input_image_path = '/media/irro/All/RecoveryStatus/DATA/raw/nDTM/LideaSouth_nDTM_2022_30cm.tif'
        # self.input_image_path = '/media/irina/My Book/Recovery/DATA/raw/nDTM/Lidea2_2024_25PPm_ncdtm_50cm.tif'
        # self.input_image_path = '/media/irro/All/HumanFootprint/DATA/nDTM/WAranch_ndtm/jul10_L1_WAlranch_ndtm_merged.tif'
        # self.input_image_path = '/media/irina/My Book/Blueberry/3_LiDAR_derivatives/PA2-W2(West)-SouthSikanniRoad-SiteB_ndtm.tif'

        # self.input_image_path = '/media/irina/My Book1/Blueberry/DEMs/GrizzlyCreekSite_ncdtm_PD5.tif'
        # self.input_image_path = '/media/irina/My Book1/Blueberry/DEMs/PA2-W2(West)-SouthSikanniRoad_ncdtm_PD300.tif'
        # self.input_image_path = '/media/irina/My Book1/Blueberry/DEMs/PA1-PinkMtnRanchRestored_ncdtm_PD300.tif'
        # self.input_image_path = '/media/irina/My Book1/Blueberry/DEMs/PA2-E1(East)-AtticCreekRoadSite_ncdtm_PD5.tif'
        # self.input_image_path = '/media/irina/My Book1/Blueberry/DEMs/PA2-E2(East)-BeattonRiver-SiteB_ncdtm_PD5.tif'
        # self.input_image_path = '/media/irina/My Book1/Blueberry/DEMs/PA2-E2(East)-BeattonRiver-SiteA_ncdtm_PD5.tif'
        # self.input_image_path = '/media/irina/My Book1/Blueberry/DEMs/PA2-W2(West)-RestoredWellpadAccess_ncdtm_PD5.tif'


        self.output_dir = os.path.join(self.base_dir(), self.config['prediction_params']['output_dir'])
        self.patch_size = self.config['prediction_params']['patch_size']
        self.overlap_size = self.config['prediction_params']['overlap_size']
        self.invert_image = self.config['prediction_params']['inverted']

        print(f"\n******\nModel Path: {self.model_path}", flush=True)
        print(f"Input Image Path: {self.input_image_path}", flush=True)
        print(f"Output Directory: {self.output_dir}", flush=True)

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}", flush=True)

        self.next(self.load_model)

    @step
    def load_model(self):
        """
        Load the trained UNet model for prediction.
        """
        config = self.config['model']['custom_unet']

        input_shape = (config['patch_size'], config['patch_size'], config['num_bands'])
        print(f"Loading model from: {self.model_path}", flush=True)

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
        print("Model loaded successfully.", flush=True)

        self.next(self.predict_and_save)

    @step
    def predict_and_save(self):
        """
        Perform sliding window prediction on the large input image and save incrementally to disk.
        """
        stride = self.patch_size - self.overlap_size

        # Extract the base name of the input image
        base_name = os.path.splitext(os.path.basename(self.input_image_path))[0]
        model_name = os.path.basename(self.config['prediction_params']['model_path'])[:-3]
        output_path = os.path.join(self.output_dir, f"{base_name}_{model_name}.tif")
        print('Save to output path: ', output_path, flush=True)

        # Run sliding window prediction and save incrementally
        sliding_window_prediction_tf(
            image_path=self.input_image_path,
            model=self.model,
            patch_size=self.patch_size,
            stride=stride,
            output_path=output_path,
            reference_image_path=self.input_image_path,  # Pass the same image as reference for metadata
            inverted = self.invert_image
        )

        del self.model
        gc.collect()

        self.next(self.end)

    @step
    def end(self):
        """
        Final step to log the completion of the prediction flow
        """
        print("UNet prediction flow completed successfully.", flush=True)


if __name__ == '__main__':
    UNetPredictionFlow()
