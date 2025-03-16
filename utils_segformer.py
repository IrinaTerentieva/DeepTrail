import os  # For file and directory handling
import torch  # For PyTorch tensors and neural network functionalities
import torch.nn.functional as F  # For loss functions like softmax and cross-entropy
import torch.nn as nn  # For building custom neural network layers
import rasterio  # For reading and writing raster data
import numpy as np  # For numerical operations
from torch.utils.data import Dataset  # For creating custom PyTorch datasets
from tqdm import tqdm  # For progress tracking
import gc  # For garbage collection
import wandb  # For logging to Weights & Biases
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
import numpy as np
import torch
import rasterio
from scipy.ndimage import label

class TrailsDataset(Dataset):
    """
    A PyTorch Dataset class for handling seismic line image and mask data.

    Args:
        data_dir (str): Path to the directory containing image and mask files.
        threshold (int, optional): Threshold for binarizing the mask. Defaults to 20.
        min_area (int, optional): Minimum area of connected regions to keep. Defaults to 300.
        transform (albumentations.Compose, optional): Transformation pipeline for data augmentation.
    """

    def __init__(self, data_dir, std = 0.5, threshold=0, min_area=100, skip_threshold = 0.01, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.threshold = threshold  # Default threshold for binarizing the mask
        self.min_area = min_area  # Default minimum area for filtering
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('_image.tif')]
        self.skip_threshold = skip_threshold
        self.std = std

        all_image_files = [f for f in os.listdir(data_dir) if f.endswith('_image.tif')]
        self.valid_files = []
        for image_file in all_image_files:
            label_file = image_file.replace('_image.tif', '_label.tif')
            label_path = os.path.join(data_dir, label_file)
            if not os.path.exists(label_path):
                continue
            try:
                with rasterio.open(label_path) as src:
                    mask = src.read(1).astype(np.uint8)
            except Exception as e:
                print(f"Error reading label file {label_path}: {e}")
                continue

            mask = np.where(mask > self.threshold, 1, 0).astype(np.uint8)
            mask = self._filter_small_regions(mask)
            ratio = np.sum(mask) / mask.size
            if ratio >= self.skip_threshold:
                self.valid_files.append(image_file)
            else:
                print(f"Skipping {image_file} because label area ratio ({ratio:.4f}) is below threshold {skip_threshold}")


    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        image_file = self.valid_files[idx]
        label_file = image_file.replace('_image.tif', '_label.tif')
        image_path = os.path.join(self.data_dir, image_file)
        label_path = os.path.join(self.data_dir, label_file)

        # Open the image and mask, handle nodata by setting it to 0
        with rasterio.open(image_path) as img_src:
            image = img_src.read(1).astype(np.float32)
            image_nodata = img_src.nodata
            if image_nodata is not None:
                image[image == image_nodata] = 0  # Replace nodata with 0

            image = normalize_nDTM(image, self.std)

        with rasterio.open(label_path) as label_src:
            mask = label_src.read(1).astype(np.uint8)  # Ensure mask is binary (0, 1)
            mask_nodata = label_src.nodata
            if mask_nodata is not None:
                mask[mask == mask_nodata] = 0  # Replace nodata with 0

            # Binarize mask based on threshold
            mask = np.where(mask > self.threshold, 1, 0).astype(np.uint8)

            # Filter out small regions
            mask = self._filter_small_regions(mask)

        # Check the image and mask shape
        if image.shape != mask.shape:
            print(f"Before AUGM >> Size mismatch: {image_path}, Image size: {image.shape}, Mask size: {mask.shape}")

        # Ensure the image is a tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).unsqueeze(0).float()

        # Ensure the mask is a tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return image, mask

    def _filter_small_regions(self, mask):
        """
        Filter out connected regions in the mask smaller than the minimum area.

        Args:
            mask (np.ndarray): Binary mask to filter.

        Returns:
            np.ndarray: Filtered binary mask.
        """
        # Label connected regions
        labeled_mask, num_features = label(mask)
        sizes = np.bincount(labeled_mask.ravel())  # Count the size of each region

        # Create a mask to retain regions larger than or equal to min_area
        large_regions = np.zeros_like(mask, dtype=np.uint8)
        for region_id, size in enumerate(sizes):
            if size >= self.min_area and region_id != 0:  # Ignore background region (id=0)
                large_regions[labeled_mask == region_id] = 1  # Retain large region

        return large_regions


# ------------------- Image nDTM Normalization -------------------
def normalize_nDTM(image, std):
    """
    Normalize image to the [0, 1] range.
    - Replace all negative values with 0.
    - Use the next smallest positive value as the minimum for normalization.
    """
    # Replace NaNs with 0
    image = np.nan_to_num(image, nan=0.0)

    # Clip negative values to -10

    min_positive_val = -std
    max_val = std

    image = np.clip(image, min_positive_val, max_val)

    # Normalize to [0, 1] using the next positive value
    image = (image - min_positive_val) / (max_val - min_positive_val)
    return image

def compute_class_weights(loader, num_classes):
    """
    Compute class weights dynamically based on pixel frequencies in the dataset.
    """
    class_counts = Counter()
    total_pixels = 0

    for _, mask in loader:
        mask_np = mask.cpu().numpy()
        unique, counts = np.unique(mask_np, return_counts=True)
        class_counts.update(dict(zip(unique, counts)))
        total_pixels += mask_np.size

    # Calculate class weights inversely proportional to class frequencies
    weights = []
    for i in range(num_classes):
        class_frequency = class_counts.get(i, 0) / total_pixels
        weight = 1.0 / (class_frequency + 1e-6)  # Avoid division by zero
        weights.append(weight)

    return torch.tensor(weights)

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate IoU and Dice scores for predictions and ground truth.
    """
    predictions = (predictions > threshold).float()
    targets = targets.float()
    print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")

    intersection = (predictions * targets).sum(dim=(1, 2))
    union = predictions.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection
    dice = 2 * intersection / (predictions.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + 1e-6)

    iou = intersection / (union + 1e-6)

    return iou.mean().item(), dice.mean().item()

def validate_data(loader):
    """
    Validate images and masks for spatial alignment, size consistency, and value ranges.
    """
    for idx, (image, mask) in enumerate(loader):
        unique_img_values = torch.unique(image)
        unique_mask_values = torch.unique(mask)

        # Check size consistency
        if image.shape[2:] != mask.shape[1:]:
            raise ValueError(f"Image and mask size mismatch at index {idx}: "
                             f"Image shape {image.shape}, Mask shape {mask.shape}")

        # Check value ranges
        if not (0 <= image.min() and image.max() <= 1):
            print(f"image {idx} unique values: {unique_img_values}")
            raise ValueError(f"Image values out of range at index {idx}.")
        if not (mask.min() >= 0 and mask.max() <= 1):
            print(f"image {idx} unique values: {unique_mask_values}")
            raise ValueError(f"Mask values out of range at index {idx}.")

        # Check spatial alignment (optional visualization)
        if idx < 5:  # Only visualize the first few pairs
            print(f"Visualizing image and mask pair at index {idx}")
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image[0].cpu().numpy().transpose(1, 2, 0))
            plt.title("Image")
            plt.subplot(1, 2, 2)
            plt.imshow(mask[0].cpu().numpy(), cmap='gray')
            plt.title("Mask")
            plt.show()



# ------------------- Augmentation -------------------

def build_augmentations(config):
    """
    Build data augmentation transformations based on the config file.
    """
    aug_config = config['augmentations']
    augmentations = []

    if aug_config.get('brightness_contrast', False):
        augmentations.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
    if aug_config.get('elastic_transform', False):
        augmentations.append(A.ElasticTransform(p=0.5))
    if aug_config.get('horizontal_flip', False):
        augmentations.append(A.HorizontalFlip(p=0.5))
    if aug_config.get('random_rotate90', False):
        augmentations.append(A.RandomRotate90(p=1.0))
    if aug_config.get('rotate', False):
        augmentations.append(A.Rotate(limit=360, p=0.5))
    if aug_config.get('vertical_flip', False):
        augmentations.append(A.VerticalFlip(p=0.5))
    if aug_config.get('gaussian_noise', False):
        augmentations.append(A.GaussNoise(p=0.5))
    if aug_config.get('scale', False):
        augmentations.append(A.RandomScale(scale_limit=0.1, p=0.5))

    # Convert to tensor as the final transformation
    augmentations.append(ToTensorV2())

    return A.Compose(augmentations)

# ------------------- Predictions -------------------

def sliding_window_prediction(image_path, model, patch_size, stride, max_height):
    """
    Perform sliding window prediction on an image with averaging for overlapping pixels.

    Args:
        image_path (str): Path to the image.
        model (torch.nn.Module): The trained model to use for prediction.
        patch_size (int): Size of the patch for the sliding window.
        stride (int): Stride for the sliding window.
        max_height (float): Maximum height for normalization.

    Returns:
        np.ndarray: Full prediction array with averaged probabilities.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with rasterio.open(image_path) as src:
        # Read and normalize the image
        image = src.read(1).astype(np.float32)
        nodata_value = src.nodata
        if nodata_value is not None:
            image[image == nodata_value] = 0

        # image = np.clip(image, 0, max_height)
        # image = image / max_height

        image = normalize_nDTM(image)
        height, width = image.shape

        # Initialize accumulation and count arrays
        full_prediction = np.zeros((height, width), dtype=np.float32)
        count_array = np.zeros((height, width), dtype=np.int32)  # Count how many times a pixel is covered

        print("Starting sliding window prediction...")

        for i in tqdm(range(0, height - patch_size + 1, stride), desc="Sliding Window Rows"):
            for j in tqdm(range(0, width - patch_size + 1, stride), desc="Sliding Window Cols", leave=False):
                patch = image[i:i + patch_size, j:j + patch_size]
                tensor_patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                with torch.no_grad():
                    outputs = model(pixel_values=tensor_patch)
                    logits = outputs.logits
                    probabilities = torch.sigmoid(logits)

                    upsampled_probs = torch.nn.functional.interpolate(
                        probabilities, size=(patch_size, patch_size), mode="bilinear",
                        align_corners=False
                    )
                    prob_values = upsampled_probs.squeeze(0).squeeze(0).cpu().numpy()

                # Accumulate the probabilities and update the count array
                full_prediction[i:i + patch_size, j:j + patch_size] += prob_values
                count_array[i:i + patch_size, j:j + patch_size] += 1

        print('Preds shape: ', full_prediction.shape)
        # Divide accumulated probabilities by count array to get the mean
        full_prediction = full_prediction / np.maximum(count_array, 1)  # Avoid division by zero
        print('Predictions completed')
    return full_prediction

def save_predictions(full_prediction, output_path, reference_image_path, scale_factor=100, config=None, chunk_size=64, save_batch_every=None):
    """
    Save the predicted result as a GeoTIFF in chunks or batches.

    Args:
        full_prediction (numpy.ndarray): Prediction array to save.
        output_path (str): Path where the GeoTIFF should be saved.
        reference_image_path (str): Path to the reference image (to get CRS and transform).
        scale_factor (int): Scaling factor to convert prediction to uint16.
        chunk_size (int): Size of chunks to save at a time.
        save_batch_every (int): Number of rows after which to save the batch (optional, defaults to None).
    """
    print('Saving starts')

    torch.cuda.empty_cache()
    gc.collect()

    # Scale prediction and convert to uint16 format
    full_prediction = (full_prediction * scale_factor).astype(np.uint16)

    # Load the reference image to get metadata
    with rasterio.open(reference_image_path) as src:
        height, width = full_prediction.shape
        crs = src.crs
        transform = src.transform

    # Open the output GeoTIFF file for writing
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=rasterio.uint16,
            crs=crs,
            transform=transform,
            blockxsize=chunk_size,
            blockysize=chunk_size,
            tiled=True
    ) as dst:
        print('Start saving in chunks: ')
        # Save predictions in chunks
        for i in tqdm(range(0, height, chunk_size), desc="Saving chunks"):
            for j in range(0, width, chunk_size):
                window_height = min(chunk_size, height - i)
                window_width = min(chunk_size, width - j)

                chunk = full_prediction[i:i + window_height, j:j + window_width]
                dst.write(chunk, 1, window=rasterio.windows.Window(j, i, window_width, window_height))

            # If save_batch_every is set, save batches periodically
            if save_batch_every and i % save_batch_every == 0:
                batch_output_path = output_path.replace('.tif', f'_batch_{i // save_batch_every}.tif')
                save_batch(full_prediction[:i + chunk_size, :], batch_output_path, reference_image_path)

    print(f"Predictions saved to {output_path}")

def save_batch(batch_data, batch_output_path, reference_image_path):
    """
    Save a batch of predictions to a temporary file.

    Args:
        batch_data (numpy.ndarray): Batch data to save.
        batch_output_path (str): Path to save the batch file.
        reference_image_path (str): Path to the reference image for metadata.
    """
    with rasterio.open(reference_image_path) as src:
        height, width = batch_data.shape
        crs = src.crs
        transform = src.transform

    print("Starts saving in batches")
    with rasterio.open(
            batch_output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=rasterio.uint16,
            crs=crs,
            transform=transform
    ) as dst:
        dst.write(batch_data, 1)

    print(f"Batch saved to {batch_output_path}")




# ------------------- Loss Functions -------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply softmax to convert logits to probabilities for multiclass segmentation
        probs = F.softmax(inputs, dim=1)
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)  # Calculate pt
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Dice Loss class
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid if not already in [0, 1] range
        inputs = torch.sigmoid(inputs)

        # Flatten the tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Calculate Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


# ------------------- Training  -------------------

# EarlyStopping class definition
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# ------------------- Image Checking  -------------------

import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_statistics(image, mask):
    """
    Calculate statistics for image and mask.
    """
    image_stats = {
        "min": np.min(image),
        "max": np.max(image),
        "mean": np.mean(image),
        "std": np.std(image)
    }
    unique, counts = np.unique(mask, return_counts=True)
    mask_stats = dict(zip(unique, counts))

    return image_stats, mask_stats

def plot_histograms(image, mask, axs, idx):
    """
    Plot histograms for image and mask.
    """
    axs[0].hist(image.ravel(), bins=50, alpha=0.5, label=f'Image {idx + 1}')
    axs[0].set_title('Image Pixel Distributions')
    axs[0].set_xlabel('Pixel Values')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].hist(mask.ravel(), bins=[-0.5, 0.5, 1.5], alpha=0.5, label=f'Mask {idx + 1}')
    axs[1].set_title('Binary Mask Distributions')
    axs[1].set_xlabel('Mask Values (0 = background, 1 = line)')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

def save_image_mask_pair(image, mask, idx, figures_dir):
    """
    Save image-mask pair after augmentation to figures directory.
    """
    fig_pair, axs_pair = plt.subplots(1, 2, figsize=(10, 5))
    axs_pair[0].imshow(image, cmap='gray')
    axs_pair[0].set_title(f'Image {idx + 1}')
    axs_pair[1].imshow(mask, cmap='gray')
    axs_pair[1].set_title(f'Mask {idx + 1}')

    pair_figure_path = os.path.join(figures_dir, f'dataloader_pair_{idx + 1}.png')
    plt.savefig(pair_figure_path)
    plt.close()

    return pair_figure_path


import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import rasterio
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import gc
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
from scipy.ndimage import label
import matplotlib.pyplot as plt


# ------------------- Helper Function -------------------
def normalize_nDTM(image, std=0.5):
    """
    Normalize the image to the [0, 1] range.
    - Replace NaNs with 0.
    - Clip values to the range [-std, std].
    - Normalize to [0, 1].
    """
    image = np.nan_to_num(image, nan=0.0)
    min_val = -std
    max_val = std
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)
    return image


# ------------------- TrailsDataset Class -------------------
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import rasterio
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import gc
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
from scipy.ndimage import label
import matplotlib.pyplot as plt


def normalize_RGB(image):
    """
    Normalize an RGB image to the [0, 1] range.
    Assumes the input is in the range [0, 255] and in HWC format.
    """
    image = np.clip(image, 0, 255)
    return image / 255.0


class WetTrailsDataset(Dataset):
    """
    A PyTorch Dataset for handling RGB image and mask data.
    It reads '_image.tif' files (assumed to be RGB) and corresponding '_label.tif' files.
    Default augmentations are applied if none are provided.

    Args:
        data_dir (str): Directory containing image and label files.
        threshold (int, optional): Threshold to binarize the mask. Default is 0.
        min_area (int, optional): Minimum area (in pixels) for connected regions to keep. Default is 100.
        skip_threshold (float, optional): Minimum label area ratio required to keep a sample. Default is 0.01.
        transform (albumentations.Compose, optional): Augmentation pipeline.
    """

    def __init__(self, data_dir, threshold=0, min_area=100, skip_threshold=0.01, transform=None):
        self.data_dir = data_dir
        self.threshold = threshold
        self.min_area = min_area
        self.skip_threshold = skip_threshold

        # If no transformation pipeline is provided, build the default augmentations.
        if transform is None:
            self.transform = self.build_default_augmentations()
        else:
            self.transform = transform

        # Collect valid image files with a corresponding label file and sufficient mask area.
        all_image_files = [f for f in os.listdir(data_dir) if f.endswith('_image.tif')]
        self.valid_files = []
        for image_file in all_image_files:
            label_file = image_file.replace('_image.tif', '_label.tif')
            label_path = os.path.join(data_dir, label_file)
            if not os.path.exists(label_path):
                continue
            try:
                with rasterio.open(label_path) as src:
                    mask = src.read(1).astype(np.uint8)
            except Exception as e:
                print(f"Error reading label file {label_path}: {e}")
                continue

            mask = np.where(mask > self.threshold, 1, 0).astype(np.uint8)
            mask = self._filter_small_regions(mask)
            ratio = np.sum(mask) / mask.size
            if ratio >= self.skip_threshold:
                self.valid_files.append(image_file)
            else:
                print(
                    f"Skipping {image_file} because label area ratio ({ratio:.4f}) is below threshold {self.skip_threshold}")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        image_file = self.valid_files[idx]
        label_file = image_file.replace('_image.tif', '_label.tif')
        image_path = os.path.join(self.data_dir, image_file)
        label_path = os.path.join(self.data_dir, label_file)

        # Read the RGB image.
        with rasterio.open(image_path) as img_src:
            # Read all channels (typically shape: (channels, height, width)).
            image = img_src.read().astype(np.float32)
            nodata = img_src.nodata
            if nodata is not None:
                image[image == nodata] = 0
            # Convert from CHW to HWC for Albumentations.
            if image.shape[0] > 1:
                image = np.transpose(image, (1, 2, 0))
            else:
                image = np.expand_dims(image[0], axis=-1)

            # If the image has more than 3 channels (e.g. RGBA), keep only the first 3 (RGB).
            if image.shape[2] > 3:
                image = image[:, :, :3]
            image = normalize_RGB(image)

        # Read the label/mask (assumed single-channel).
        with rasterio.open(label_path) as label_src:
            mask = label_src.read(1).astype(np.uint8)
            nodata = label_src.nodata
            if nodata is not None:
                mask[mask == nodata] = 0
            mask = np.where(mask > self.threshold, 1, 0).astype(np.uint8)
            mask = self._filter_small_regions(mask)

        # Check shape consistency: image should have shape (H, W, C) and mask (H, W).
        if image.shape[:2] != mask.shape:
            print(f"Size mismatch: {image_path}, Image shape: {image.shape}, Mask shape: {mask.shape}")

        # Apply augmentations if provided.
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # If no augmentation, convert image from HWC to CHW.
            if image.ndim == 3:
                image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
            else:
                image = torch.from_numpy(image).unsqueeze(0).float()
            mask = torch.from_numpy(mask).long()

        return image, mask

    def _filter_small_regions(self, mask):
        """
        Filter out connected regions in the mask smaller than the specified minimum area.
        """
        labeled_mask, num_features = label(mask)
        sizes = np.bincount(labeled_mask.ravel())
        large_regions = np.zeros_like(mask, dtype=np.uint8)
        for region_id, size in enumerate(sizes):
            if region_id != 0 and size >= self.min_area:
                large_regions[labeled_mask == region_id] = 1
        return large_regions

    @staticmethod
    def build_default_augmentations():
        """
        Build a default augmentation pipeline using Albumentations.
        Default augmentations include:
          - Random brightness/contrast adjustments
          - Color jitter (Hue, Saturation, Value changes)
          - Horizontal and vertical flips
          - Random rotation by 90° (i.e. 90, 180, or 270 degrees)
          - Conversion to PyTorch tensors
        """
        augmentations = []
        augmentations.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
        augmentations.append(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5))
        augmentations.append(A.HorizontalFlip(p=0.5))
        augmentations.append(A.VerticalFlip(p=0.5))
        augmentations.append(A.RandomRotate90(p=0.5))
        augmentations.append(ToTensorV2())
        return A.Compose(augmentations)