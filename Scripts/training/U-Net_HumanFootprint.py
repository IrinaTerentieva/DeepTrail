import os
import sys
import shutil
import yaml
import warnings
import numpy as np
import tensorflow as tf
import wandb
from metaflow import FlowSpec, step, current

sys.path.append('/media/irro/All/HumanFootprint/Models/custom_unet')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_unet import custom_unet
from utils import TrailsDataGenerator, iou, iou_thresholded, load_data, plot_predictions

# Suppress TensorFlow and CUDA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class UNetFlow(FlowSpec):
    environment = 'local'
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
        """
        Start step: Load configuration settings from the central config.yaml file and initialize WandB.
        """
        # Load YAML configuration
        self.config_path = os.path.join(self.base_dir, 'config.yaml')
        print(f"Loading configuration from: {self.config_path}")
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Patch size and directory settings
        self.patch_size_pixels = self.config['preprocessing']['patch_extraction']['patch_size_pixels']
        self.patch_dir = os.path.join(self.base_dir, self.config['preprocessing']['patch_extraction']['patches'].format(
            patch_size_pixels=self.patch_size_pixels))

        # Initialize WandB for the specific experiment
        wandb_key_path = os.path.join(self.base_dir, 'Scripts', 'training', 'wandb_key.txt')
        with open(wandb_key_path, 'r') as f:
            wandb_key = f.read().strip()
        wandb.login(key=wandb_key)

        # Initialize WandB and upload config + scripts
        self.wandb_run = wandb.init(project=self.config['project']['project_name'], config=self.config)

        # Log basic config info to WandB
        wandb.config.update({
            "architecture": "UNet",
            "filters": self.config['model']['custom_unet']['filters'],
            "num_layers": self.config['model']['custom_unet']['num_layers'],
            "dropout": self.config['model']['custom_unet']['dropout'],
            "learning_rate": self.config['training']['learning_rate'],
            "batch_size": self.config['training']['batch_size'],
            "epochs": self.config['training']['num_epochs'],
        })

        # Save config and scripts in WandB
        wandb.save(self.config_path)
        wandb.save('/media/irro/All/HumanFootprint/Scripts/training/U-Net_HumanFootprint.py')
        wandb.save('/media/irro/All/HumanFootprint/Scripts/training/utils.py')

        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Load the dataset, logging the directories and dataset properties.
        """
        # Re-initialize WandB to ensure logging works in this step
        wandb.init(project=self.config['project']['project_name'], resume=True)

        print('Directory with patches: ', self.patch_dir)
        self.train_images, self.val_images, self.train_labels, self.val_labels = load_data(self.patch_dir)

        # Log dataset information to WandB
        wandb.config.update({
            "train_dataset_size": len(self.train_images),
            "val_dataset_size": len(self.val_images),
            "train_dataset_path": self.patch_dir,
            "val_dataset_path": self.patch_dir
        })

        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train the UNet model using the custom_unet architecture.
        """
        # Re-initialize WandB to ensure logging works in this step
        wandb.init(project=self.config['project']['project_name'], resume=True)

        config = self.config

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

        # Build a meaningful model name
        model_name = f"UNet-patch{self.patch_size_pixels}-lr{config['training']['learning_rate']}-epoch{{epoch}}-val_loss{{val_loss:.2f}}"

        # Set up dynamic model checkpoint path
        checkpoint_dir = os.path.join(self.base_dir, 'Models', 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        dynamic_model_path = os.path.join(checkpoint_dir, f"{model_name}")

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
            history = model.fit(train_gen, validation_data=val_gen, epochs=1)

            train_loss = history.history['loss'][0]
            val_loss = history.history['val_loss'][0]
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

            # Validation step: model.evaluate returns a list [val_loss, other metrics...]
            val_results = model.evaluate(val_gen, verbose=0)
            val_loss = val_results[0]  # Extract the validation loss (first element of the list)
            print(f"Validation Loss: {val_loss}")

            # Plot and log predictions after each epoch
            val_images, val_masks = next(iter(val_gen))
            val_preds = model.predict(val_images)

            # Check if validation loss improved
            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model.")
                best_val_loss = val_loss

                # Save model architecture and weights in universal format
                model_save_path = dynamic_model_path.format(epoch=epoch + 1, val_loss=val_loss)
                model.save_weights(f"{model_save_path}.weights.h5")  # Save weights
                with open(os.path.join(checkpoint_dir, f'{model_name}.json'), 'w') as json_file:
                    json_file.write(model.to_json())  # Save architecture

                # Save config, scripts, and metadata
                self.save_metadata(checkpoint_dir, epoch, model_name)

                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

            # Early stopping condition
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        self.next(self.end)

    def save_metadata(self, model_folder, epoch, model_name):
        """
        Save the config, training script, and metadata once in the same folder as the model.
        """
        # Save the YAML config file
        shutil.copy(self.config_path, os.path.join(model_folder, f"{model_name}_config.yaml"))

        # Save the Python script
        shutil.copy(__file__, os.path.join(model_folder, f"{model_name}_training.py"))

        # Save WandB run information
        wandb_run_info = {
            "wandb_run_id": wandb.run.id,
            "wandb_run_name": wandb.run.name,
            "wandb_project": wandb.run.project,
            "wandb_url": wandb.run.url
        }
        with open(os.path.join(model_folder, f"wandb_run_metadata_epoch_{epoch + 1}.yaml"), 'w') as f:
            yaml.dump(wandb_run_info, f)

        # Save Metaflow flow information
        metaflow_info = {
            "run_id": current.run_id,
            "step_name": current.step_name,
            "task_id": current.task_id
        }
        with open(os.path.join(model_folder, f"metaflow_metadata_epoch_{epoch + 1}.yaml"), 'w') as f:
            yaml.dump(metaflow_info, f)

        print(f"Metadata saved in: {model_folder}")

    @step
    def end(self):
        """
        Final step to indicate the end of training.
        """
        print("UNet training completed!")
        wandb.finish()


if __name__ == '__main__':
    UNetFlow()
