import sys
sys.path.append('/media/irro/All/HumanFootprint/Models/custom_unet')
from sklearn.model_selection import train_test_split
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tensorflow as tf
from custom_unet import custom_unet

# Suppress TensorFlow and CUDA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ---------------------- #
# Parameters and Settings #
# ---------------------- #

# Environment and paths
environment = os.getenv('ENVIRONMENT', 'local')
username = 'irina.terenteva'

if environment == 'hpc':
    patch_dir = f'/home/{username}/HumanFootprint/'
else:
    patch_dir = '/media/irro/All/HumanFootprint/'


# model_path = '/media/irro/All/HumanFootprint/Models/Weights/50cm/Human_DTMnorm50_512_byCNN_5ep.h5'
model_path = '/media/irro/All/HumanFootprint/Models/Weights/best/Human_DTM10cm_512_byCNN_9ep_good.h5'

# Model settings
size = 1024

patch_dir = f'/media/irro/All/HumanFootprint/DATA/TrainingCNN/UNet_patches{size}_nDTM10cm'

input_shape = (size, size, 1)
filters = 32
use_batch_norm = True
dropout = 0.3
num_classes = 1
output_activation = 'sigmoid'
num_layers = 4

# Batch size and image size
batch_size = 2
image_size = (size, size)

# ---------------------------- #
# General Configurations       #
# ---------------------------- #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ---------------------------- #
# Data Preprocessing Functions #
# ---------------------------- #

def normalize_image(image):
    """Normalize image to [0, 1] range and replace NaN with 0."""
    # Replace NaN values with 0 before normalization
    image = np.nan_to_num(image, nan=0.0)

    # Calculate min and max while ignoring NaNs (already replaced by 0s)
    min_val = np.min(image)
    max_val = np.max(image)

    # Print min and max values
    print('Min value: ', min_val)
    print('Max value: ', max_val)

    # Avoid division by zero in case the image has no variation
    if max_val - min_val == 0:
        return image  # No normalization needed if all values are the same

    # Normalize the image to [0, 1] range
    return (image - min_val) / (max_val - min_val)


def remove_small_objects(mask, min_size=200):
    """Remove small objects from mask."""
    from scipy.ndimage import label
    labeled_mask, num_features = label(mask)
    small_objects_mask = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        component = np.where(labeled_mask == i, 1, 0)
        if np.sum(component) < min_size:
            small_objects_mask += component
    return mask - small_objects_mask

def plot_images_masks(images, masks):
    """Plot images and masks side by side."""
    num = min(images.shape[0], 5)
    fig, axs = plt.subplots(num, 2, figsize=(10, 5 * num))
    for i in range(num):
        axs[i, 0].imshow(images[i, :, :, 0], cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Image')
        axs[i, 1].imshow(masks[i, :, :, 0], cmap='gray')
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Mask')
    plt.tight_layout()
    plt.show()

def print_range_and_mean(data, data_type="Image"):
    """Print range and mean for non-zero and non-NaN values."""
    non_zero_data = data[data > 0]
    non_zero_data = non_zero_data[~np.isnan(non_zero_data)]
    if len(non_zero_data) > 0:
        data_min = np.min(non_zero_data)
        data_max = np.max(non_zero_data)
        data_mean = np.mean(non_zero_data)
        print(f"{data_type} - Range: [{data_min}, {data_max}], Mean: {data_mean}")
    else:
        print(f"{data_type} - No valid data (all zeros or NoData).")

# ---------------------- #
# Data Generator         #
# ---------------------- #

class DataGenerator(tf.keras.utils.Sequence):
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

# ---------------------- #
# Model Setup and Loading#
# ---------------------- #

print(f"Loading model from: {model_path}")
themodel = custom_unet(
    input_shape,
    filters=filters,
    use_batch_norm=use_batch_norm,
    dropout=dropout,
    num_classes=num_classes,
    output_activation=output_activation,
    num_layers=num_layers
)
themodel.load_weights(model_path)
# themodel.summary()

# ---------------------- #
# Plot and Predictions   #
# ---------------------- #

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


# ---------------------- #
# Load Data and Predict  #
# ---------------------- #

# Getting all files ending with .tif
all_files = [f for f in os.listdir(patch_dir) if f.endswith('.tif')]
train_images = [os.path.join(patch_dir, f) for f in all_files if 'image' in f]
train_labels = [os.path.join(patch_dir, f.replace('image', 'label')) for f in train_images if os.path.isfile(os.path.join(patch_dir, f.replace('image', 'label')))]

print(f"Total training images: {len(train_images)}")
print(f"Total training labels: {len(train_labels)}")

# Split dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

train_gen = DataGenerator(train_images, train_labels, batch_size=batch_size, image_size=image_size, shuffle=True, augment=False)
val_gen = DataGenerator(val_images, val_labels, batch_size=batch_size, image_size=image_size, shuffle=False)

# Test Predictions
X_val_batch, y_val_batch = val_gen[0]
preds_val_batch = (themodel.predict(X_val_batch) > 0.1).astype(np.uint8)

# Plot predictions
plot_predictions(X_val_batch, y_val_batch, preds_val_batch)
print('Size: ', size)