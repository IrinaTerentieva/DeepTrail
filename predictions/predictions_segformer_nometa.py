import os
import yaml
import torch
import rasterio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import gc
import tempfile
import scipy.ndimage as ndimage

def normalize_nDTM(image, std = 0.1):
    """Normalize image to the [0, 1] range."""
    min_val, max_val = -std, std
    image = np.clip(image, min_val, max_val)
    return (image - min_val) / (max_val - min_val)

def sliding_window_prediction(image_path, model, output_dir, patch_size, stride):
    """
    Perform sliding window prediction on a single image using memmap for accumulation
    to reduce peak RAM usage. Remove connected components < 200 pixels after threshold=0.3.
    Then multiply final probabilities by 100, convert to uint8, and save with compression.
    """
    with rasterio.open(image_path) as src:
        print(f"\n[INFO] Processing: {image_path}")
        image = src.read(1)
        image = np.nan_to_num(image, nan=0.0)

        # If the source has a nodata value that is invalid for uint8, you can ignore it in the output
        image_nodata = src.nodata
        if image_nodata is not None:
            # We'll just set those pixels to 0 in the input for the sake of model inference
            image[image == image_nodata] = 0

        # Normalize input DTM
        image = normalize_nDTM(image)

        height, width = image.shape

        # Create memory-mapped arrays for accumulation and count
        temp_dir = tempfile.gettempdir()
        accum_path = os.path.join(temp_dir, "accumulation.dat")
        count_path = os.path.join(temp_dir, "count_array.dat")

        accumulation = np.memmap(accum_path, dtype=np.float32, mode='w+', shape=(height, width))
        accumulation[:] = 0.0
        count_array = np.memmap(count_path, dtype=np.int32, mode='w+', shape=(height, width))
        count_array[:] = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sliding window
        for i in tqdm(range(0, height - patch_size + 1, stride), desc="Sliding Window Row"):
            for j in tqdm(range(0, width - patch_size + 1, stride), desc="Sliding Window Col", leave=False):
                patch = image[i:i + patch_size, j:j + patch_size]
                if np.all(patch == 0):
                    continue

                tensor_patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    outputs = model(pixel_values=tensor_patch)
                    logits = outputs.logits
                    probabilities = torch.sigmoid(logits)
                    upsampled_probs = torch.nn.functional.interpolate(
                        probabilities, size=(patch_size, patch_size),
                        mode="bilinear", align_corners=False
                    )
                    prob_values = upsampled_probs.squeeze(0).squeeze(0).cpu().numpy()

                accumulation[i:i + patch_size, j:j + patch_size] += prob_values
                count_array[i:i + patch_size, j:j + patch_size] += 1

                # Cleanup
                del tensor_patch, outputs, logits, probabilities, upsampled_probs, prob_values
                torch.cuda.empty_cache()
            gc.collect()

        accumulation.flush()
        count_array.flush()

        # Compute final probability map
        full_prediction = accumulation / np.maximum(count_array, 1)

        # Threshold at 0.3, remove connected areas < 200 pixels
        threshold = 0.3
        bin_mask = (full_prediction >= threshold).astype(np.uint8)

        labeled, num_features = ndimage.label(bin_mask)
        component_sizes = ndimage.sum(bin_mask, labeled, index=range(1, num_features + 1))
        too_small = [idx for idx, size in enumerate(component_sizes, start=1) if size < 200]

        if too_small:
            too_small_set = set(too_small)
            bin_mask[np.isin(labeled, list(too_small_set))] = 0

        full_prediction[bin_mask == 0] = 0.0

        # Multiply by 100 and convert to uint8
        scaled_prediction = (full_prediction * 100).astype(np.uint8)

        # Prepare output path
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_preds_segformer.tif'
        output_path = os.path.join(output_dir, output_filename)

        # Build updated profile for uint8 output
        new_profile = src.profile.copy()
        new_profile.update({
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw'
        })
        # Option A: Remove nodata so you don't get the invalid range error
        new_profile.pop('nodata', None)

        # Option B: Alternatively, specify a valid uint8 nodata value (e.g. 255)
        # new_profile['nodata'] = 255

        from rasterio.windows import Window
        chunk_size = 1024

        with rasterio.open(output_path, 'w', **new_profile) as dst:
            for row in range(0, height, chunk_size):
                for col in range(0, width, chunk_size):
                    window = Window(
                        col, row,
                        min(chunk_size, width - col),
                        min(chunk_size, height - row)
                    )
                    data_chunk = scaled_prediction[row:row + window.height, col:col + window.width]
                    dst.write(data_chunk, 1, window=window)

        # Cleanup
        del accumulation, count_array, full_prediction, bin_mask, labeled, scaled_prediction
        gc.collect()

        print(f"[INFO] Prediction saved to {output_path}")
        return output_path

def load_configuration():
    base_dir = '/home/irina/HumanFootprint'
    config_path = os.path.join(base_dir, 'config_segformer.yaml')
    print(f"[INFO] Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config, base_dir

def load_model(config, base_dir):
    prediction_params = config['prediction_params']
    model_path = os.path.join(base_dir, prediction_params['model_path'])

    model_path = '/home/irina/HumanFootprint/Models/SegFormer_HumanFootprint_dataset_1024_nDTM10cm_arch_mit-b2_lr_0.01_batch_4_epoch_60/pytorch_model_weights.pth'
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

    # Modify first conv layer to accept single-channel input
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
    # Overwrite with your actual test image path if needed
    test_image_path = '/media/irina/My Book1/Conoco/DATA/Drone_PPC_2024/ndtm/Area_C_combined_ncdtm.tif'
    # test_image_path = '/media/irina/My Book/Surmont/nDTM_synth_trails/nDTM_10cm_trails_v2/502_6223_nDTM_blended_synth_trails_v5.tif'

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

    # If test_image_path is a directory, list all tif files; otherwise, use that single file
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
