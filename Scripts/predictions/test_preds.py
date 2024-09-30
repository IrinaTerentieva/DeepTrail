import os
import yaml
import rasterio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('/media/irro/All/HumanFootprint/Models/custom_unet')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import yaml
import warnings
import rasterio
import numpy as np
import tensorflow as tf
from custom_unet import custom_unet
from utils import plot_predictions, TrailsDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Suppress TensorFlow and CUDA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ------------------- #
# Utility Functions   #
# ------------------- #

def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def normalize_image(image):
    """
    Normalize image to the [0, 1] range.
    - Replace all negative values with 0.
    - Use the next smallest positive value as the minimum for normalization.
    """
    # Replace NaNs with 0
    image = np.nan_to_num(image, nan=0.0)

    # Clip negative values to 0
    image = np.clip(image, 0, np.max(image))

    # Find the next smallest positive value after 0
    positive_values = image[image > 0]
    if len(positive_values) == 0:
        min_positive_val = 0.01  # Default in case no positive value exists
    else:
        min_positive_val = np.min(positive_values)

    # Normalize the image using the next smallest positive value
    max_val = np.max(image)

    print(f"Min positive value: {min_positive_val}, Max value: {max_val}")

    # Avoid division by zero if all values are the same
    if max_val - min_positive_val == 0:
        return image  # No normalization needed if all values are the same

    # Normalize to [0, 1] using the next positive value
    return (image - min_positive_val) / (max_val - min_positive_val)


def plot_predictions(images, ground_truths, predictions, num=5):
    """Plot images, ground truths, and predictions side by side with normalization."""
    # Set num to the minimum between the actual batch size and the desired number of images
    num = min(images.shape[0], num)

    fig, axs = plt.subplots(num, 3, figsize=(15, 5 * num))

    for i in range(num):
        # Normalize images and predictions using the provided normalize_image function
        normalized_image = normalize_image(images[i, :, :, 0])
        normalized_prediction = normalize_image(predictions[i, :, :, 0])

        # Plot original image
        axs[i, 0].imshow(normalized_image, cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Image')

        # Plot ground truth mask
        axs[i, 1].imshow(ground_truths[i, :, :, 0], cmap='gray')
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Ground Truth Mask')

        # Plot predicted mask
        axs[i, 2].imshow(normalized_prediction, cmap='gray')
        axs[i, 2].axis('off')
        axs[i, 2].set_title('Predicted Mask')

    plt.tight_layout()
    plt.show()


def calculate_statistics(image, mask):
    """Calculate and print statistics for image and mask."""
    image_stats = {
        "min": np.min(image),
        "max": np.max(image),
        "mean": np.mean(image),
        "std": np.std(image)
    }

    mask_stats = {
        "min": np.min(mask),
        "max": np.max(mask),
        "mean": np.mean(mask),
        "std": np.std(mask)
    }

    unique, counts = np.unique(mask, return_counts=True)
    mask_stats = dict(zip(unique, counts))

    print('Image stats: ', image_stats)
    print('Mask stats: ', mask_stats)

    return image_stats, mask_stats

# ------------------- #
# Data Generator Class#
# ------------------- #

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

# ------------------- #
# Model Testing       #
# ------------------- #

def test_model_on_patches(model, patches_dir, num_patches=5):
    all_files = [f for f in os.listdir(patches_dir) if f.endswith('.tif')]
    train_images = [os.path.join(patches_dir, f) for f in all_files if 'image' in f]
    train_labels = [os.path.join(patches_dir, f.replace('image', 'label')) for f in train_images if os.path.isfile(os.path.join(patches_dir, f.replace('image', 'label')))]

    print(f"Total training images: {len(train_images)}")

    # Create validation generator
    val_gen = DataGenerator(train_images, train_labels, batch_size=5, augment=False)

    # Test Predictions
    X_val_batch, y_val_batch = val_gen[0]
    preds_val_batch = (model.predict(X_val_batch) > 0.1).astype(np.uint8)
    calculate_statistics(X_val_batch, preds_val_batch)
    plot_predictions(X_val_batch, y_val_batch, preds_val_batch, num_patches)


# ------------------- #
# Main Script         #
# ------------------- #

if __name__ == '__main__':
    # Load configuration
    config_path = '/media/irro/All/HumanFootprint/config.yaml'
    config = load_yaml_config(config_path)

    # Extract model configuration
    model_config = config['model']['custom_unet']
    input_shape = (model_config['patch_size'], model_config['patch_size'], model_config['num_bands'])
    filters = model_config['filters']
    use_batch_norm = model_config['use_batch_norm']
    dropout = model_config['dropout']
    num_classes = model_config['num_classes']
    output_activation = model_config['output_activation']
    num_layers = model_config['num_layers']

    # Load the TensorFlow model
    model_path = os.path.join('/media/irro/All/HumanFootprint/Models/Weights/best/Human_DTM10cm_512_byCNN_9ep_good.h5')
    print(f"Loading model from: {model_path}")
    themodel = custom_unet(
        input_shape=input_shape,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        num_classes=num_classes,
        output_activation=output_activation,
        num_layers=num_layers
    )
    themodel.load_weights(model_path)

    # Test the model on patches
    patches_dir = os.path.join('/media/irro/All/HumanFootprint/', config['preprocessing']['patch_extraction']['patches'])
    test_model_on_patches(themodel, patches_dir, num_patches=2)
