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
from utils import TrailsDataGenerator, iou, iou_thresholded, load_data, plot_predictions

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

class UNetFlow(FlowSpec):

    environment = 'local'
    # environment = 'hpc'
    username = 'irina.terenteva'

    @property
    def base_dir(self):
        """
        Dynamically set the base directory depending on environment.
        """
        if self.environment == 'hpc':
            return f'/home/{self.username}/HumanFootprint/'
        else:
            return '/media/irro/All/HumanFootprint/'

    @step
    def start(self):

        self.config_path = os.path.join(self.base_dir, 'config.yaml')
        print(f"Loading configuration from: {self.config_path}")

        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.patch_dir = os.path.join(self.base_dir, self.config['preprocessing']['patch_extraction']['patches'])
        self.next(self.load_data)

    @step
    def load_data(self):
        print('Directory with patches: ', self.patch_dir)
        self.train_images, self.val_images, self.train_labels, self.val_labels = load_data(self.patch_dir)
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train the UNet model using the custom_unet architecture.
        """
        config = self.config

        # Initialize WandB
        wandb.init(project=config['project']['project_name'])

        # Set up model input shape
        input_shape = (config['model']['custom_unet']['patch_size'], config['model']['custom_unet']['patch_size'], 1)

        # Initialize the UNet model
        model = custom_unet(input_shape=input_shape, filters=config['model']['custom_unet']['filters'],
                            use_batch_norm=config['model']['custom_unet']['use_batch_norm'],
                            dropout=config['model']['custom_unet']['dropout'],
                            num_classes=config['model']['custom_unet']['num_classes'],
                            output_activation=config['model']['custom_unet']['output_activation'])

        # Compile the model
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=config['training']['learning_rate']),
                      loss='binary_crossentropy', metrics=[iou, iou_thresholded])

        # Set up dynamic model checkpoint path
        checkpoint_dir = os.path.join(self.base_dir, 'Models', 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        dynamic_model_path = os.path.join(checkpoint_dir, 'unet_epoch_{epoch:02d}_valloss_{val_loss:.2f}.h5')

        # Create data generators for training and validation
        train_gen = TrailsDataGenerator(self.train_images, self.train_labels,
                                        batch_size=config['training']['batch_size'], augment=True)
        val_gen = TrailsDataGenerator(self.val_images, self.val_labels, batch_size=config['training']['batch_size'])

        # Manually implement early stopping and model checkpointing
        best_val_loss = np.inf
        patience = config['training']['early_stopping_patience']
        patience_counter = 0

        for epoch in range(config['training']['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
            print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

            # Training step
            history = model.fit(train_gen, validation_data=val_gen, epochs=config['training']['num_epochs'])

            train_loss = history.history['loss'][0]
            val_loss = history.history['val_loss'][0]
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

            # Validation step: model.evaluate returns a list [val_loss, other metrics...]
            val_results = model.evaluate(val_gen, verbose=0)
            val_loss = val_results[0]  # Extract the validation loss (first element of the list)
            print(f"Validation Loss: {val_loss}")

            # Check if validation loss improved
            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model.")
                best_val_loss = val_loss
                model.save(dynamic_model_path.format(epoch=epoch, val_loss=val_loss))
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

            # Early stopping condition
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Finish the WandB logging
        wandb.finish()
        self.next(self.end)

    @step
    def end(self):
        """
        Final step to indicate the end of training.
        """
        print("UNet training completed!")

if __name__ == '__main__':
    UNetFlow()