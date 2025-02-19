import os
import yaml
import torch
import rasterio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

def normalize_nDTM(image):
    """Normalize image to the [0, 1] range."""
    min_val, max_val = -2, 2
    image = np.clip(image, min_val, max_val)
    return (image - min_val) / (max_val - min_val)

def sliding_window_prediction(image_path, model, output_dir, patch_size, stride):
    """
    Perform sliding window prediction on a single image.
    """
    with rasterio.open(image_path) as src:
        print(f"\n[INFO] Processing: {image_path}")
        image = src.read(1)
        image = np.nan_to_num(image, nan=0.0)
        image_nodata = src.nodata
        if image_nodata is not None:
            image[image == image_nodata] = 0  # Handle nodata values

        # Normalize input DTM
        image = normalize_nDTM(image)

        height, width = image.shape
        accumulation = np.zeros((height, width), dtype=np.float32)
        count_array = np.zeros((height, width), dtype=np.int32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sliding window inference
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
                    probabilities = torch.sigmoid(logits)  # Binary segmentation
                    upsampled_probs = torch.nn.functional.interpolate(
                        probabilities, size=(patch_size, patch_size), mode="bilinear", align_corners=False
                    )
                    prob_values = upsampled_probs.squeeze(0).squeeze(0).cpu().numpy()

                # Accumulate predictions
                accumulation[i:i + patch_size, j:j + patch_size] += prob_values
                count_array[i:i + patch_size, j:j + patch_size] += 1

        # Compute final prediction by averaging overlapping areas
        full_prediction = accumulation / np.maximum(count_array, 1)

        # Save output
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_preds_segformer.tif'
        output_path = os.path.join(output_dir, output_filename)
        from rasterio.windows import Window

        # Write predictions in chunks.
        chunk_size = 512
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
            for row in range(0, height, chunk_size):
                for col in range(0, width, chunk_size):
                    window = Window(col, row, min(chunk_size, width - col), min(chunk_size, height - row))
                    data_chunk = full_prediction[row:row + window.height, col:col + window.width]
                    dst.write(data_chunk, 1, window=window)

        print(f"[INFO] Prediction saved to {output_path}")
        return output_path

def load_configuration():
    # Set your base directory here.
    base_dir = '/home/irina/HumanFootprint'
    config_path = os.path.join(base_dir, 'config_segformer.yaml')
    print(f"[INFO] Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config, base_dir

def load_model(config, base_dir):
    # Load parameters from YAML
    prediction_params = config['prediction_params']
    model_path = os.path.join(base_dir, prediction_params['model_path'])
    print(f"[INFO] Loading Model from: {model_path}")

    model_config = config['models']['mit-b3']
    config_obj = SegformerConfig(
        num_labels=1,
        depths=model_config['depths'],
        hidden_sizes=model_config['hidden_sizes'],
        decoder_hidden_size=model_config['decoder_hidden_size'],
        aspp_ratios=model_config.get('aspp_ratios', [1, 6, 12, 18]),
        aspp_out_channels=model_config.get('aspp_out_channels', 256),
        attention_probs_dropout_prob=model_config.get('attention_dropout_prob', 0.1),
    )

    model = SegformerForSemanticSegmentation(config_obj)

    # Modify first conv layer to accept single channel input
    model.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(
        in_channels=1,
        out_channels=model.segformer.encoder.patch_embeddings[0].proj.out_channels,
        kernel_size=model.segformer.encoder.patch_embeddings[0].proj.kernel_size,
        stride=model.segformer.encoder.patch_embeddings[0].proj.stride,
        padding=model.segformer.encoder.patch_embeddings[0].proj.padding,
        bias=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def run_prediction():
    config, base_dir = load_configuration()

    prediction_params = config['prediction_params']
    test_image_path = os.path.join(base_dir, prediction_params['test_image_path'])
    output_dir = os.path.join(base_dir, prediction_params['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    patch_size = prediction_params['patch_size']
    overlap_size = prediction_params['overlap_size']
    stride = patch_size - overlap_size

    print("\n*********\nConfiguration:")
    print(f"Test Image Path: {test_image_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Patch Size: {patch_size}")
    print(f"Stride: {stride}")
    print(f"Overlap Size: {overlap_size}")

    model = load_model(config, base_dir)

    # If test_image_path is a directory, list all tif files; otherwise, use the file.
    if os.path.isdir(test_image_path):
        image_files = [os.path.join(test_image_path, f) for f in os.listdir(test_image_path) if f.endswith('.tif')]
    else:
        image_files = [test_image_path]

    print(f"\n[INFO] Found {len(image_files)} images for prediction")
    for img_path in image_files:
        sliding_window_prediction(img_path, model, output_dir, patch_size, stride)

def main():
    print("[INFO] Starting Segformer Prediction Flow")
    run_prediction()
    print("[INFO] Prediction flow completed.")

if __name__ == '__main__':
    main()
