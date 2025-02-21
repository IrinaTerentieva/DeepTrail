import os
import sys
import torch
import torch.nn.functional as F
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation

# Add the parent directory (where utils_segformer.py is located) to the system path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_segformer import normalize_nDTM  # Assumes you have a normalization function


def load_image(image_path):
    """
    Loads a single-band TIFF image, replaces nodata with zero,
    and applies normalization.
    """
    with rasterio.open(image_path) as src:
        image = src.read(1).astype(np.float32)
        print(image)
        if src.nodata is not None:
            image[image == src.nodata] = 0
        transform = src.transform
        crs = src.crs
        profile = src.profile
    # Apply your normalization function
    image = normalize_nDTM(image)
    return image, transform, crs, profile


def main():
    # --- Set Paths and Parameters Directly ---
    # Base directory for your project
    base_dir = "/home/irina/HumanFootprint"

    # Folder containing the saved model. Make sure this folder contains both the weights and config.
    # Example: "SegFormer_HumanFootprint_dataset_1024_nDTM10cm_arch_mit-b2_lr_0.0001_batch_4_epoch_25_valLoss_0.0595"
    model_save_folder = '/home/irina/HumanFootprint/Models/best_segformer_epoch_49_val_loss_0.0582/'

    # Path to the test image (should be a relatively small TIFF file)
    test_image_path = "/home/irina/HumanFootprint/DATA/Training_CNN/synth_tracks_1024px_10cm/64_image.tif"

    # Output directory where predictions will be saved
    output_dir = os.path.join(base_dir, "DATA", "Test_Models")

    # --- Load the Model ---
    print(f"Loading model from: {model_save_folder}")

    # --- Setup Model ---
    def modify_first_conv_layer(model):
        # Replace the first convolution to accept single-channel input.
        model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=1,
            out_channels=model.segformer.encoder.patch_embeddings[0].proj.out_channels,
            kernel_size=model.segformer.encoder.patch_embeddings[0].proj.kernel_size,
            stride=model.segformer.encoder.patch_embeddings[0].proj.stride,
            padding=model.segformer.encoder.patch_embeddings[0].proj.padding,
            bias=False
        )
        return model

    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    import yaml
    import torch.nn as nn

    base_dir = "/home/irina/HumanFootprint"
    config_path = os.path.join(base_dir, 'config_segformer.yaml')
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    architecture = config['experiments']['architectures'][0]
    print(architecture)
    config_model = config['models'][architecture]

    segformer_config = SegformerConfig(
            num_labels=config_model['num_labels'],
            hidden_sizes=config_model['hidden_sizes'],
            depths=config_model['depths'],
            decoder_hidden_size=config_model['decoder_hidden_size'],
            aspp_ratios=config_model.get('aspp_ratios', None) if config_model.get('use_aspp', False) else None,
            aspp_out_channels=config_model.get('aspp_out_channels', None) if config_model.get('use_aspp',
                                                                                              False) else None,
            attention_dropout_prob=config_model.get('attention_dropout_prob', 0.1)
        )
    model = SegformerForSemanticSegmentation(segformer_config)
    model = modify_first_conv_layer(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # --- Load and Preprocess the Test Image ---
    print(f"Loading test image from: {test_image_path}")
    image, transform, crs, profile = load_image(test_image_path)
    print(image)

    # Convert image to a tensor with shape [1, 1, H, W] (assuming a single-channel image)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

    # --- Run Inference ---
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        # Convert logits to probabilities with sigmoid
        pred_prob = torch.sigmoid(outputs.logits)
        print(pred_prob)
        # For binary segmentation, use a threshold of 0.5 to get a binary mask
        pred_mask = (pred_prob > 0.1).float()
        # Convert to numpy array and remove extra dimensions
        pred_mask_np = pred_mask.squeeze().cpu().numpy()

    # --- Save Predictions ---
    os.makedirs(output_dir, exist_ok=True)

    # Save as a PNG image for quick visualization.
    png_path = os.path.join(output_dir, "prediction.png")
    plt.imsave(png_path, pred_mask_np, cmap="gray")
    print(f"Prediction PNG saved at: {png_path}")

    # Save as a GeoTIFF to preserve geospatial metadata.
    # Make a copy of the original profile and update its fields.
    profile_out = profile.copy()
    profile_out.update({
        "dtype": "uint8",
        "count": 1,
        "height": pred_mask_np.shape[0],
        "width": pred_mask_np.shape[1],
    })
    # Override any existing nodata value with a valid one for uint8.
    profile_out["nodata"] = 0

    tif_path = os.path.join(output_dir, "prediction.tif")
    with rasterio.open(tif_path, "w", **profile_out) as dst:
        # Multiply by 255 to convert the binary mask to an 8-bit image.
        dst.write((pred_mask_np * 255).astype("uint8"), 1)
    print(f"Prediction GeoTIFF saved at: {tif_path}")


if __name__ == "__main__":
    main()
