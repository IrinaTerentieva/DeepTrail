import os  # For file and directory handling
# import torch  # For PyTorch tensors and neural network functionalities
# import torch.nn.functional as F  # For loss functions like softmax and cross-entropy
# import torch.nn as nn  # For building custom neural network layers
import rasterio  # For reading and writing raster data
import numpy as np  # For numerical operations
# from torch.utils.data import Dataset  # For creating custom PyTorch datasets
from tqdm import tqdm  # For progress tracking
import gc  # For garbage collection
import wandb  # For logging to Weights & Biases
import albumentations as A
# from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import tensorflow as tf


# ------------------- Data Loader -------------------

class TrailsDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_list, mask_list, batch_size=32, image_size=(512, 512), shuffle=True, augment=False):
        self.image_list = image_list
        self.mask_list = mask_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.image_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_list) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_list[i] for i in indices]
        batch_mask_paths = [self.mask_list[i] for i in indices]

        X = np.zeros((self.batch_size, *self.image_size, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.image_size, 1), dtype=np.float32)

        for i, (img_path, mask_path) in enumerate(zip(batch_image_paths, batch_mask_paths)):
            img, mask = self.load_data(img_path, mask_path)
            if self.augment:
                img, mask = self.augment_image(img, mask)
            X[i] = img
            y[i] = mask

        return X, y

    def load_data(self, img_path, mask_path):
        """Load and preprocess image and mask."""
        with rasterio.open(img_path) as src:
            img = src.read(1)
            img = np.expand_dims(img, axis=-1)
            img = tf.image.resize(img, self.image_size)
            img = normalize_image(img)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            mask = np.expand_dims(mask, axis=-1)
            mask = tf.image.resize(mask, self.image_size)

        return img, mask

    def augment_image(self, image, mask):
        """Apply augmentations to image and mask."""
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, mask

def load_data(patch_dir):
    all_files = [f for f in os.listdir(patch_dir) if f.endswith('.tif')]
    train_images = [os.path.join(patch_dir, f) for f in all_files if 'img' in f]
    train_labels = [os.path.join(patch_dir, f.replace('img', 'lab')) for f in train_images if os.path.isfile(os.path.join(patch_dir, f.replace('img', 'lab')))]
    return train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

import tensorflow as tf

def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    return iou(y_true, y_pred, smooth)

def plot_predictions(images, ground_truths, predictions, num=5):
    """Plot images, ground truths, and predictions side by side."""
    # Set num to the minimum between the actual batch size and the desired number of images
    num = min(images.shape[0], num)

    fig, axs = plt.subplots(num, 3, figsize=(15, 5 * num))

    for i in range(num):
        # Plot original image
        axs[i, 0].imshow(images[i, :, :, 0], cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Image')

        # Plot ground truth mask
        axs[i, 1].imshow(ground_truths[i, :, :, 0], cmap='gray')
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Ground Truth Mask')

        # Plot predicted mask
        axs[i, 2].imshow(predictions[i, :, :, 0], cmap='gray')
        axs[i, 2].axis('off')
        axs[i, 2].set_title('Predicted Mask')

    plt.tight_layout()
    plt.show()

# class TrailsDataset(Dataset):
#     """
#     A PyTorch Dataset class for handling seismic line image and mask data.
#
#     Args:
#         data_dir (str): Path to the directory containing image and mask files.
#         max_height (float): Maximum height value to clip and normalize the image.
#         transform (albumentations.Compose, optional): Transformation pipeline for data augmentation.
#     """
#     def __init__(self, data_dir, max_height, transform=None):
#         self.data_dir = data_dir
#         self.max_height = max_height  # Set max_height here
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(data_dir) if f.endswith('_image.tif')]
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         image_file = self.image_files[idx]
#         label_file = image_file.replace('_image.tif', '_label.tif')
#         image_path = os.path.join(self.data_dir, image_file)
#         label_path = os.path.join(self.data_dir, label_file)
#
#         # Open the image and mask, handle nodata by setting it to 0
#         with rasterio.open(image_path) as img_src:
#             image = img_src.read(1).astype(np.float32)
#             image_nodata = img_src.nodata
#             if image_nodata is not None:
#                 image[image == image_nodata] = 0  # Replace nodata with 0
#
#             # Clip CHM values to [0, max_height] and normalize to [0, 1]
#             image = normalize_image(image)
#             # image = np.clip(image, 0, self.max_height)
#             # image = image / self.max_height  # Normalize to [0, 1]
#
#         with rasterio.open(label_path) as label_src:
#             mask = label_src.read(1).astype(np.uint8)  # Ensure mask is binary (0, 1)
#             mask_nodata = label_src.nodata
#             if mask_nodata is not None:
#                 mask[mask == mask_nodata] = 0  # Replace nodata with 0
#
#         # Check the image and mask shape
#         if image.shape != mask.shape:
#             print(f"Before AUGM >> Size mismatch: {img_path}, Image size: {image.shape}, Mask size: {mask.shape}")
#
#         # Ensure the image is a tensor
#         if not isinstance(image, torch.Tensor):
#             image = torch.from_numpy(image).unsqueeze(0).float()
#
#         # Ensure the mask is a tensor
#         if not isinstance(mask, torch.Tensor):
#             mask = torch.from_numpy(mask).long()
#
#         return image, mask


# ------------------- Image Normalization -------------------

def normalize_image(image, max_height=25.0):
    """
    Normalize the image by clipping and scaling.

    Args:
        image (np.array): Image array to normalize.
        max_height (float): Maximum height value for clipping.

    Returns:
        torch.Tensor: Normalized image.
    """
    image = np.clip(image, 0, max_height)
    image = image / max_height
    return image


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

        image = np.clip(image, 0, max_height)
        image = image / max_height

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

        # Divide accumulated probabilities by count array to get the mean
        full_prediction = full_prediction / np.maximum(count_array, 1)  # Avoid division by zero
        print('Probabilities shape: ', probabilities.shape)

    return full_prediction

# ------------------- Save Predictions -------------------

def save_predictions(full_prediction, output_path, reference_image_path, scale_factor=100, config=None):
    """
    Save the predicted result as a GeoTIFF using chunks.

    Args:
        full_prediction (numpy.ndarray): Prediction array to save.
        output_path (str): Path where the GeoTIFF should be saved.
        reference_image_path (str): Path to the reference image (to get CRS and transform).
        scale_factor (int): Scaling factor to convert prediction to uint16.
    """
    torch.cuda.empty_cache()
    gc.collect()

    if not wandb.run:
        wandb.init(project=config['project']['project_name'])

    print('Saving starts ')
    full_prediction = (full_prediction * scale_factor).astype(np.uint16)  # Convert to uint16 by scaling

    with rasterio.open(reference_image_path) as src:
        height, width = full_prediction.shape
        crs = src.crs
        transform = src.transform

    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=rasterio.uint16,  # or whatever dtype you need
            crs=src.crs,
            transform=src.transform,
            blockxsize=64,  # Reduce chunk size
            blockysize=64,
            tiled=True
    ) as dst:
        for i in tqdm(range(0, height, 64), desc="Saving chunks"):
            for j in range(0, width, 64):
                window_height = min(64, height - i)
                window_width = min(64, width - j)

                chunk = full_prediction[i:i + window_height, j:j + window_width]
                dst.write(chunk, 1, window=rasterio.windows.Window(j, i, window_width, window_height))

    print(f"Predictions saved to {output_path}")


# ------------------- Loss Functions -------------------

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         # Apply softmax to convert logits to probabilities for multiclass segmentation
#         probs = F.softmax(inputs, dim=1)
#         CE_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-CE_loss)  # Calculate pt
#         F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
#
#         if self.reduction == 'mean':
#             return F_loss.mean()
#         elif self.reduction == 'sum':
#             return F_loss.sum()
#         else:
#             return F_loss
#
# # Dice Loss class
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#
#     def forward(self, inputs, targets):
#         # Apply sigmoid if not already in [0, 1] range
#         inputs = torch.sigmoid(inputs)
#
#         # Flatten the tensors
#         inputs = inputs.reshape(-1)
#         targets = targets.reshape(-1)
#
#         # Calculate Dice coefficient
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
#
#         return 1 - dice


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

