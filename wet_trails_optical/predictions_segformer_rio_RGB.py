import os
import tempfile
import numpy as np
import torch
import rioxarray
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import gc
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import yaml

# Set your temporary directory for memory mapping.
tempfile.tempdir = "/media/irina/My Book/tmp"


def normalize_RGB(image):
    """Normalize RGB image to the [0, 1] range."""
    return image / 255.0  # Assumes image is already clipped to [0, 255]


def sliding_window_prediction(image_path, model, output_dir, patch_size, stride, plot_comparison=False):
    """
    Perform sliding window prediction on a large RGB image using rioxarray,
    reading one patch at a time to avoid high memory usage.
    Includes additional debug prints of patch shapes/statistics.
    Optionally, if plot_comparison=True and a ground truth mask file is available,
    plots the full prediction, binary prediction, and ground truth side by side.
    """
    print(f"\n[INFO] Processing: {image_path}")

    # Open the raster lazily.
    ds = rioxarray.open_rasterio(image_path, masked=True)
    ds = ds.fillna(0)  # Replaces all NaN values with 0

    height = ds.rio.height
    width = ds.rio.width
    print(f"[DEBUG] Image dimensions: height={height}, width={width}")

    # Create memory-mapped arrays for accumulation and count.
    temp_dir = tempfile.gettempdir()
    accum_path = os.path.join(temp_dir, "accumulation.dat")
    count_path = os.path.join(temp_dir, "count_array.dat")
    accumulation = np.memmap(accum_path, dtype=np.float32, mode='w+', shape=(height, width))
    accumulation[:] = 0.0
    count_array = np.memmap(count_path, dtype=np.int32, mode='w+', shape=(height, width))
    count_array[:] = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sliding window: iterate over rows and columns.
    for i in tqdm(range(0, height - patch_size + 1, stride), desc="Sliding Window Rows"):
        for j in tqdm(range(0, width - patch_size + 1, stride), desc="Sliding Window Cols", leave=False):
            try:
                patch = ds.isel(
                    band=slice(0, 3),  # use first 3 bands (RGB)
                    y=slice(i, i + patch_size),
                    x=slice(j, j + patch_size)
                ).values.astype(np.float32)  # Expected shape: (3, patch_size, patch_size)
            except Exception as e:
                print(f"Error reading window at row {i}, col {j}: {e}")
                continue

            # # Debug: print patch statistics.
            # print(
            #     f"[DEBUG] Patch at (row {i}, col {j}) shape: {patch.shape}, min: {np.min(patch)}, max: {np.max(patch)}, mean: {np.mean(patch)}")

            patch = normalize_RGB(patch)
            # Skip patch if it's completely black.
            if np.all(patch == 0):
                continue

            tensor_patch = torch.from_numpy(patch).unsqueeze(0).float().to(device)  # shape: (1, 3, H, W)

            with torch.no_grad():
                outputs = model(pixel_values=tensor_patch)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits)
                upsampled_probs = torch.nn.functional.interpolate(
                    probabilities, size=(patch_size, patch_size),
                    mode="bilinear", align_corners=False
                )
                prob_values = upsampled_probs.squeeze(0).squeeze(0).cpu().numpy()
                # Debug: Print prediction stats for the patch.
                # print(
                #     f"[DEBUG] Probabilities for patch at (row {i}, col {j}): min: {np.min(prob_values)}, max: {np.max(prob_values)}, mean: {np.mean(prob_values)}")

            accumulation[i:i + patch_size, j:j + patch_size] += prob_values
            count_array[i:i + patch_size, j:j + patch_size] += 1

            del tensor_patch, outputs, logits, probabilities, upsampled_probs, prob_values
            torch.cuda.empty_cache()
        gc.collect()

    accumulation.flush()
    count_array.flush()

    # Compute the final probability map.
    full_prediction = accumulation / np.maximum(count_array, 1)
    print(
        f"[DEBUG] Full prediction stats: min: {np.min(full_prediction)}, max: {np.max(full_prediction)}, mean: {np.mean(full_prediction)}")

    # Post-process: thresholding and removal of small connected components.
    threshold_val = 0.1
    bin_mask = (full_prediction >= threshold_val).astype(np.uint8)
    labeled, num_features = ndimage.label(bin_mask)
    component_sizes = ndimage.sum(bin_mask, labeled, index=range(1, num_features + 1))
    too_small = [idx for idx, size in enumerate(component_sizes, start=1) if size < 10]
    if too_small:
        bin_mask[np.isin(labeled, too_small)] = 0
    full_prediction[bin_mask == 0] = 0.0

    scaled_prediction = (full_prediction * 100).astype(np.uint8)
    # print(f"[DEBUG] Scaled prediction unique values: {np.unique(scaled_prediction)}")

    # Prepare output path.
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_wetformer_ep23.tif'
    output_path = os.path.join(output_dir, output_filename)

    output_ds = ds.isel(band=0).copy(deep=True)
    if "long_name" in output_ds.attrs:
        del output_ds.attrs["long_name"]
    # Remove or reset nodata attributes to avoid conversion issues.
    if "nodata" in output_ds.attrs:
        output_ds.attrs["nodata"] = 0
    if "encoded_nodata" in output_ds.attrs:
        del output_ds.attrs["encoded_nodata"]
    output_ds.rio.write_nodata(0, inplace=True)

    output_ds.values = scaled_prediction
    output_ds.rio.to_raster(output_path, compress="lzw", dtype="uint8")

    del accumulation, count_array, full_prediction, bin_mask, labeled, scaled_prediction
    gc.collect()

    print(f"[INFO] Prediction saved to {output_path}")

    return output_path


def load_configuration():
    base_dir = '/home/irina/HumanFootprint/'
    config_path = os.path.join(base_dir, 'config_segformer.yaml')
    print(f"[INFO] Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config, base_dir


def load_model(config, base_dir):
    prediction_params = config['prediction_params']
    model_path = os.path.join(base_dir, prediction_params['model_path'])
    model_path = '/home/irina/HumanFootprint/HF_models/WetFormer_HumanFootprint_dataset_1024_nDTM10cm_arch_mit-b2_lr_0.001_batch_4_epoch_23/pytorch_model_weights.pth'
    print(f"[INFO] Loading Model from: {model_path}")

    model_config = config['models']['mit-b2']
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
    # Modify first conv layer for 3-channel input.
    model.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(
        in_channels=3,
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

    ### **********************
    test_image_path = '/media/irina/data/tmp_10cm/Area_E_polygon0-1_2cm.tif'

    output_dir = os.path.join(base_dir, prediction_params['output_dir'])
    output_dir = '/media/irina/My Book1/Conoco/DATA/Products/WetTrails'
    os.makedirs(output_dir, exist_ok=True)
    patch_size = prediction_params['patch_size']
    overlap_size = prediction_params['overlap_size']
    stride = patch_size - overlap_size
    model = load_model(config, base_dir)

    if os.path.isdir(test_image_path):
        image_files = [os.path.join(test_image_path, f) for f in os.listdir(test_image_path) if f.endswith('.tif')]
    else:
        image_files = [test_image_path]

    print(f"\n[INFO] Found {len(image_files)} images for prediction")
    for img_path in image_files:
        sliding_window_prediction(img_path, model, output_dir, patch_size, stride, plot_comparison=True)


def main():
    print("[INFO] Starting Segformer Prediction Flow")
    run_prediction()
    print("[INFO] Prediction flow completed.")


if __name__ == '__main__':
    main()
