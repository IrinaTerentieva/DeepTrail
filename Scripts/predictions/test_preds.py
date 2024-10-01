import os
import sys
sys.path.append('/media/irro/All/HumanFootprint/Models/custom_unet')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import yaml
import warnings
import numpy as np
from custom_unet import custom_unet
from utils import plot_predictions, TrailsDataGenerator, normalize_image, calculate_statistics
import gc
import tensorflow as tf

# Suppress TensorFlow and CUDA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def test_model_on_patches(model, patches_dir, num_patches=5):
    all_files = [f for f in os.listdir(patches_dir) if f.endswith('.tif')]
    train_images = [os.path.join(patches_dir, f) for f in all_files if 'image' in f]
    train_labels = [os.path.join(patches_dir, f.replace('image', 'label')) for f in train_images if os.path.isfile(os.path.join(patches_dir, f.replace('image', 'label')))]

    print(f"Total training images: {len(train_images)}")

    # Create validation generator
    val_gen = TrailsDataGenerator(train_images, train_labels, batch_size=10, augment=False)
    # Test Predictions
    X_val_batch, y_val_batch = val_gen[0]
    preds_val_batch = model.predict(X_val_batch, verbose=0)
    stats = calculate_statistics(X_val_batch, preds_val_batch)
    plot = plot_predictions(X_val_batch, y_val_batch, preds_val_batch, num_patches)

# ------------------- #
# Main Script         #
# ------------------- #

if __name__ == '__main__':
    # Load configuration
    config_path = '/media/irro/All/HumanFootprint/config.yaml'
    config = load_yaml_config(config_path)
    patch_size_pixels = config['preprocessing']['patch_extraction']['patch_size_pixels']
    config['preprocessing']['patch_extraction']['patches'] = config['preprocessing']['patch_extraction'][
        'patches'].format(patch_size_pixels=patch_size_pixels)

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