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


# import os
# import numpy as np
# import argparse
# import rasterio
# from rasterio.windows import Window
# from keras_unet.models import custom_unet
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from skimage.morphology import skeletonize, dilation, square
#
# # Constants
# BLOCK_SIZE = 256
#
# # Helper Functions
# def adjust_window(window, max_width, max_height):
#     col_off, row_off, width, height = window.flatten()
#     if col_off + width > max_width:
#         width = max_width - col_off
#     if row_off + height > max_height:
#         height = max_height - row_off
#     return Window(col_off, row_off, width, height)
#
# def pad_to_shape(array, target_shape):
#     diff_height = target_shape[0] - array.shape[0]
#     diff_width = target_shape[1] - array.shape[1]
#     padded_array = np.pad(array, ((0, diff_height), (0, diff_width), (0, 0)), 'constant')
#     return padded_array
#
# def normalize_image(image):
#     min_val = np.min(image)
#     max_val = np.max(image)
#     normalized_image = (image - min_val) / (max_val - min_val)
#     return normalized_image
#
# def predict_patch_optimized(model, patch, vis=False):
#     if patch.max() != patch.min():
#         patch_norm = normalize_image(patch)
#     else:
#         patch_norm = patch
#
#     num_bands = patch.shape[2] if len(patch.shape) == 3 else 1
#     patch_norm = patch_norm.reshape(1, patch_norm.shape[0], patch_norm.shape[1], num_bands)
#
#     pred_patch = model.predict(patch_norm)
#
#     if vis:
#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.imshow(patch_norm.squeeze(), cmap='gray', vmin=0, vmax=1)
#         plt.title('Original Patch')
#         plt.axis('off')
#
#         plt.subplot(1, 2, 2)
#         plt.imshow(pred_patch.squeeze(), cmap='gray', vmin=0, vmax=1)
#         plt.title('Predicted Patch')
#         plt.axis('off')
#         plt.show()
#
#     pred_patch = (pred_patch * 100).squeeze().astype(np.uint8)
#
#     return pred_patch
#
# def process_block(src, x_start, x_end, y_start, y_end, patch_size, stride, model, vis=False):
#     pred_accumulator = np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint8)
#     counts = np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint16)
#
#     for i in tqdm(range(y_start, y_end, stride), desc=f"Processing y range {y_start} to {y_end}"):
#         for j in range(x_start, x_end, stride):
#             window = Window(j, i, patch_size, patch_size)
#             window = adjust_window(window, x_end, y_end)
#             patch = src.read(window=window)
#             patch = np.moveaxis(patch, 0, -1)
#
#             if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
#                 patch = pad_to_shape(patch, (patch_size, patch_size))
#
#             pred_patch = predict_patch_optimized(model, patch, vis)
#
#             col_off, row_off, width, height = map(int, window.flatten())
#             pred_patch = pred_patch[:height, :width]
#             accumulator_indices = (slice(row_off - y_start, row_off + height - y_start),
#                                    slice(col_off - x_start, col_off + width - x_start))
#             pred_accumulator[accumulator_indices] = np.maximum(pred_accumulator[accumulator_indices], pred_patch)
#             counts[accumulator_indices] += 1
#
#     return pred_accumulator
#
# def plot_predictions(original_patch, prediction):
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     axs[0].imshow(original_patch.squeeze(), cmap='gray')
#     axs[0].set_title('Original Patch')
#     im = axs[1].imshow(prediction, cmap='jet', vmin=0, vmax=100)
#     axs[1].set_title('Predicted Probabilities')
#     fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
#     plt.show()
#
# def predict_tif_optimized(model, path, patch_size, overlap_size, vis, model_name):
#     stride = patch_size - overlap_size
#     with rasterio.open(path) as src:
#         original_height, original_width = src.shape
#         pred_accumulator = np.zeros((original_height, original_width), dtype=np.uint8)
#         counts = np.zeros((original_height, original_width), dtype=np.uint16)
#
#         for i in tqdm(range(0, original_height, stride)):
#             for j in range(0, original_width, stride):
#                 window = Window(j, i, patch_size, patch_size)
#                 window = adjust_window(window, original_width, original_height)
#                 patch = src.read(window=window)
#                 patch = np.moveaxis(patch, 0, -1)
#
#                 if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
#                     patch = pad_to_shape(patch, (patch_size, patch_size))
#
#                 pred_patch = predict_patch_optimized(model, patch, vis)
#                 col_off, row_off, width, height = map(int, window.flatten())
#                 pred_patch = pred_patch[:height, :width]
#                 pred_accumulator[row_off:row_off+height, col_off:col_off+width] = np.maximum(pred_accumulator[row_off:row_off+height, col_off:col_off+width], pred_patch)
#                 counts[row_off:row_off+height, col_off:col_off+width] += 1
#
#         final_pred = pred_accumulator
#
#         output_path = path[:-4] + f'_{model_name}_{overlap_size}.tif'
#         profile = src.profile.copy()
#         profile.update(dtype='uint8', nodata=0, compress='LZW', tiled=True, blockxsize=BLOCK_SIZE, blockysize=BLOCK_SIZE)
#
#         with rasterio.open(output_path, 'w', **profile) as dst:
#             dst.write(final_pred[np.newaxis, :, :])
#
#     return output_path
#
# # Main Function
# def main(args):
#     model = load_model(args.model_name, custom_objects={'iou': iou, 'iou_thresholded': iou_thresholded})
#
#     # Extract and print the input shape
#     input_shape = model.layers[0].input_shape
#     print("Input shape of the model:", input_shape)
#
#     patch_size = input_shape[0][1]
#     print("Image size:", patch_size)
#
#     output_path = predict_tif_optimized(
#         model,
#         args.input_tif,
#         patch_size,
#         args.overlap_size,
#         args.visualize
#     )
#
#     print(f'Saved final prediction to: {output_path}')
