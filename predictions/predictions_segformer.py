import os
import yaml
import torch
import rasterio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from metaflow import FlowSpec, step

def sliding_window_prediction(image_path, model, output_dir, patch_size, stride):
    with rasterio.open(image_path) as src:
        print('Working with:', image_path)
        image = src.read(1)
        image = np.nan_to_num(image, nan=0.0)
        image_nodata = src.nodata
        if image_nodata is not None:
            image[image == image_nodata] = 0  # Handle nodata

        # Normalize image
        image = normalize_nDTM(image)
        print(f"Normalized Image - Min: {np.min(image):.4f}, Mean: {np.mean(image):.4f}, Max: {np.max(image):.4f}")

        height, width = image.shape
        accumulation = np.zeros((height, width), dtype=np.float32)
        count_array = np.zeros((height, width), dtype=np.int32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sliding window loop
        for i in tqdm(range(0, height - patch_size + 1, stride), desc="Sliding Window Row"):
            for j in tqdm(range(0, width - patch_size + 1, stride), desc="Sliding Window Col", leave=False):
                patch = image[i:i + patch_size, j:j + patch_size]

                # Skip empty patches
                if np.all(patch == 0):
                    continue

                tensor_patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                with torch.no_grad():
                    outputs = model(pixel_values=tensor_patch)
                    logits = outputs.logits
                    probabilities = torch.sigmoid(logits)  # Corrected to sigmoid for binary segmentation
                    upsampled_probs = torch.nn.functional.interpolate(
                        probabilities, size=(patch_size, patch_size), mode="bilinear", align_corners=False
                    )
                    prob_values = upsampled_probs.squeeze(0).squeeze(0).cpu().numpy()

                # Accumulate predictions and counts
                accumulation[i:i + patch_size, j:j + patch_size] += prob_values
                count_array[i:i + patch_size, j:j + patch_size] += 1

        # Compute final prediction by averaging overlapping areas
        full_prediction = accumulation / np.maximum(count_array, 1)

        # Save output
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_prediction.tif'
        output_path = os.path.join(output_dir, output_filename)
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=rasterio.float32,
                crs=src.crs,
                transform=src.transform
        ) as dst:
            dst.write(full_prediction, 1)

        print(f"Prediction saved to {output_path}")

def normalize_nDTM(image):
    """
    Normalize image to the [0, 1] range.
    - Replace all negative values with 0.
    - Clip values to [-1, 1] and normalize.
    """

    # Clip values to [-1, 1]
    image = np.clip(image, -0.1, 0.1)

    min_positive_val = -0.1
    max_val = 0.1

    # Normalize to [0, 1]
    image = (image - min_positive_val) / (max_val - min_positive_val)
    return image

class TrailsPredictionFlow(FlowSpec):

    environment = 'local'  # Can be overridden when running the script with --environment
    username = 'irina.terenteva'

    @property
    def base_dir(self):
        """
        Dynamically set the base directory depending on environment.
        """
        if self.environment == 'hpc':
            return f'/home/{self.username}/HumanFootprint/'
        else:
            return '/home/irina/HumanFootprint'

    @step
    def start(self):
        print(f"Starting Seismic Line Prediction with {self.environment} environment")

        base_dir = self.base_dir
        self.config_path = os.path.join(base_dir, 'config_segformer.yaml')
        print(f"Loading configuration from {self.config_path}")

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load parameters from YAML
        prediction_params = self.config['prediction_params']
        self.test_image_path = os.path.join(base_dir, prediction_params['test_image_path'])
        self.output_dir = os.path.join(base_dir, prediction_params['output_dir'])
        self.model_path = os.path.join(base_dir, prediction_params['model_path'])

        self.patch_size = prediction_params['patch_size']
        self.overlap_size = prediction_params['overlap_size']
        self.stride = self.patch_size - self.overlap_size

        print(f"\n*********\nTest Image Path: {self.test_image_path}")
        print(f"Model Path: {self.model_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Patch Size: {self.patch_size}")
        print(f"Stride: {self.stride}")
        print(f"Overlap Size: {self.overlap_size}")

        self.next(self.predict)

    @step
    def predict(self):
        print(f"Predicting with model: {self.model_path}")
        print(f"Test image path: {self.test_image_path}")
        print(f"Output directory: {self.output_dir}")

        # Load model configuration for mit-b3 from YAML
        model_config = self.config['models']['mit-b3']

        config = SegformerConfig(
            num_labels=1,
            depths=model_config['depths'],
            hidden_sizes=model_config['hidden_sizes'],
            decoder_hidden_size=model_config['decoder_hidden_size'],
            aspp_ratios=model_config.get('aspp_ratios', [1, 6, 12, 18]),
            aspp_out_channels=model_config.get('aspp_out_channels', 256),
            attention_probs_dropout_prob=model_config.get('attention_dropout_prob', 0.1),
        )

        # Initialize model
        segformer_model = SegformerForSemanticSegmentation(config)

        # Modify first layer for grayscale input
        def modify_first_conv_layer(model):
            model.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(
                in_channels=1,
                out_channels=model.segformer.encoder.patch_embeddings[0].proj.out_channels,
                kernel_size=model.segformer.encoder.patch_embeddings[0].proj.kernel_size,
                stride=model.segformer.encoder.patch_embeddings[0].proj.stride,
                padding=model.segformer.encoder.patch_embeddings[0].proj.padding,
                bias=False
            )
            return model

        segformer_model = modify_first_conv_layer(segformer_model)

        # Load model weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        segformer_model.load_state_dict(torch.load(self.model_path, map_location=device))
        segformer_model.eval()
        segformer_model.to(device)

        # Call sliding window prediction
        sliding_window_prediction(self.test_image_path, segformer_model, self.output_dir,
                                       self.patch_size, self.stride)

        self.next(self.end)



    @step
    def end(self):
        print("Prediction flow completed.")


if __name__ == '__main__':
    TrailsPredictionFlow()
