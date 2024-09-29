# cd /media/irro/All/HumanFootprint/Scripts/training
# python U-Net_HumanFootprint.py run

import yaml
import os
import warnings
from metaflow import FlowSpec, step, batch, Parameter
import numpy as np
import rasterio
import tensorflow as tf
import wandb
import sys
sys.path.append('/media/irro/All/HumanFootprint/Models/custom_unet')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import wandb
import tensorflow as tf
from metaflow import FlowSpec, step, batch, Parameter
from custom_unet import custom_unet
from utils import TrailsDataGenerator, iou, iou_thresholded, load_data

# Load YAML config
def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

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


class UNetFlow(FlowSpec):

    config_file = Parameter('config_file', help="Path to YAML configuration file", default='config.yaml')

    @step
    def start(self):
        with open(self.config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        self.patch_dir = self.config['preprocessing']['patch_extraction']['patches']
        self.next(self.load_data)

    @step
    def load_data(self):
        self.train_images, self.val_images, self.train_labels, self.val_labels = load_data(self.patch_dir)
        self.next(self.train_model)

    @batch(gpu=1)
    @step
    def train_model(self):
        config = self.config
        wandb.init(project=config['project']['project_name'])

        input_shape = (config['model']['custom_unet']['patch_size'], config['model']['custom_unet']['patch_size'], 1)
        model = custom_unet(input_shape=input_shape, filters=config['model']['custom_unet']['filters'],
                            use_batch_norm=config['model']['custom_unet']['use_batch_norm'],
                            dropout=config['model']['custom_unet']['dropout'],
                            num_classes=config['model']['custom_unet']['num_classes'],
                            output_activation=config['model']['custom_unet']['output_activation'])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
                      loss='binary_crossentropy', metrics=[iou, iou_thresholded])

        # early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'], verbose=1)
        # model_checkpoint = ModelCheckpoint(config['prediction_params']['model_path'], save_best_only=True, verbose=1)

        train_gen = TrailsDataGenerator(self.train_images, self.train_labels, batch_size=config['training']['batch_size'], augment=True)
        val_gen = TrailsDataGenerator(self.val_images, self.val_labels, batch_size=config['training']['batch_size'])

        model.fit(train_gen, validation_data=val_gen, epochs=config['training']['num_epochs'])
                  # ,                  callbacks=[early_stopping, model_checkpoint])

        wandb.finish()
        self.next(self.end)

    @step
    def end(self):
        print("UNet training completed!")

if __name__ == '__main__':
    UNetFlow()