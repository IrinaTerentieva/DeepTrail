from metaflow import FlowSpec, step
import torch
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import wandb
import yaml
from tqdm import tqdm
import sys
import os
import torch.nn as nn

# Add parent directory (where utils.py is located) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import TrailsDataset, FocalLoss, DiceLoss, EarlyStopping, build_augmentations, calculate_statistics, plot_histograms, save_image_mask_pair

# Set environment for CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

print(torch.cuda.memory_summary())

# ---------------------------
# cd /media/irro/All/HumanFootprint/Scripts/training
# python Training_HumanFootprint.py run
# ---------------------------

class SeismicLineSegmentationFlow(FlowSpec):
    # Fetch environment from environment variable
    # environment = 'local'
    environment = 'hpc'
    username = 'irina.terenteva'

    @property
    def base_dir(self):
        """
        Dynamically set the base directory depending on environment.
        """
        if self.environment == 'hpc':
            return f'/home/{self.username}/LineFootprint/'
        else:
            return '/media/irro/All/LineFootprint/'

    @step
    def start(self):
        """
        Load the configuration file, initialize directories and experiment combinations (learning rate, dataset, architecture).
        """

        self.config_path = os.path.join(self.base_dir, 'config.yaml')
        print(f"Loading configuration from: {self.config_path}")

        # Load the config file
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract experiment settings from YAML
        learning_rates = self.config['experiments']['learning_rates']
        datasets = self.config['experiments']['datasets']
        architectures = self.config['experiments']['architectures']

        # Create combinations of experiments (learning_rate, dataset, architecture)
        self.experiments = [(lr, ds, arch) for lr in learning_rates for ds in datasets for arch in architectures]

        self.next(self.load_data, foreach="experiments")

    @step
    def load_data(self):
        """
        Load and preprocess the dataset using the current experiment configuration.
        """
        # Unpack the current combination of (learning_rate, dataset, architecture)
        learning_rate, dataset_name, architecture = self.input

        # Get the dataset path from the YAML file
        dataset_path = os.path.join(self.base_dir, self.config['experiments']['datasets'][dataset_name])
        print(f"Processing dataset: {dataset_name} at {dataset_path}")
        print(f"Learning rate: {learning_rate}, Architecture: {architecture}")

        # Setup the necessary variables (e.g., batch size, num_epochs, etc.)
        self.batch_size = self.config['training']['batch_size']
        self.num_epochs = self.config['training']['num_epochs']
        self.learning_rate = learning_rate
        self.dataset_path = dataset_path
        self.architecture = architecture
        self.model_dir = os.path.join(self.base_dir, 'Models')

        # Ensure output directory exists
        self.figures_dir = os.path.join(self.dataset_path, 'Figures')
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)

        # Initialize WandB for the specific experiment
        wandb_key_path = os.path.join(self.base_dir, 'Scripts', 'training', 'wandb_key.txt')
        with open(wandb_key_path, 'r') as f:
            wandb_key = f.read().strip()
        wandb.login(key=wandb_key)

        # Setup WandB with a unique run name for each experiment
        run_name = (
            f"{self.config['logging']['project_name']}_"
            f"{self.architecture}_"
            f"LR{self.learning_rate}_"
            f"batch{self.config['training']['batch_size']}"
        ).replace('/', '_')

        wandb.init(project=self.config['logging']['project_name'], name=run_name)

        # Track key parameters explicitly in WandB
        wandb.config.update({
            "data_directory": self.dataset_path,
            "batch_size": self.config['training']['batch_size'],
            "learning_rate": learning_rate,
            "num_epochs": self.config['training']['num_epochs'],
            "architecture": architecture,
        })

        # Build augmentations based on YAML config
        augmentations = build_augmentations(self.config)

        # Initialize dataset and dataloaders using self.resolved_data_dir
        dataset = TrailsDataset(data_dir=self.dataset_path, max_height=self.config['project']['max_height'],
                                     transform=augmentations)
        print(f"Total dataset length: {len(dataset)}")

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True,
                                       num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False,
                                     num_workers=4, pin_memory=True)

        self.next(self.visualize_data_loader_output)

    @step
    def visualize_data_loader_output(self):
        """
        Visualize image-label pairs after DataLoader augmentation and save figures.
        Perform additional checks for distribution, shape, class imbalance, and normalization.
        Save figures and statistics as artifacts.
        """
        print("Saving image-label pairs and performing additional checks...")


        # Initialize lists to hold statistics and class imbalance data
        self.image_statistics = []
        self.mask_statistics = []
        self.shape_check = []
        self.class_imbalance = {"class_0": [], "class_1": []}

        # Iterate over the first 5 samples from the data loader and save visualizations
        for idx, (image, mask) in enumerate(self.val_loader):
            if idx >= 5:  # Only process the first 5 pairs
                break

            image = image[0].cpu().numpy().squeeze()  # Convert tensor to numpy array, remove batch dim
            mask = mask[0].cpu().numpy().squeeze()  # Convert mask tensor to numpy array, remove batch dim

            # Shape consistency check
            self.shape_check.append((image.shape, mask.shape))
            print(f"Image {idx + 1} shape: {image.shape}, Mask {idx + 1} shape: {mask.shape}")

            # Calculate statistics and class imbalance
            image_stats, mask_stats = calculate_statistics(image, mask)
            self.image_statistics.append(image_stats)
            self.mask_statistics.append(mask_stats)

            print(f"Image {idx + 1} stats: {image_stats}")
            print(f"Mask {idx + 1} stats (binary distribution): {mask_stats}")

            # Calculate class imbalance
            class_0_percent = (mask_stats.get(0, 0) / mask.size) * 100
            class_1_percent = (mask_stats.get(1, 0) / mask.size) * 100
            self.class_imbalance["class_0"].append(class_0_percent)
            self.class_imbalance["class_1"].append(class_1_percent)

            print(f"Class 0 (background) percentage: {class_0_percent:.2f}%")
            print(f"Class 1 (seismic line) percentage: {class_1_percent:.2f}%")

            # Save image-mask pair to the Figures folder
            pair_figure_path = save_image_mask_pair(image, mask, idx, self.figures_dir)
            print(f"Saved DataLoader image-mask pair for sample {idx + 1} to {pair_figure_path}")

        # Save statistics and other checks as artifacts
        self.image_stats_artifact = self.image_statistics
        self.mask_stats_artifact = self.mask_statistics
        self.shape_check_artifact = self.shape_check
        self.class_imbalance_artifact = self.class_imbalance

        print(f"Saved visualizations and statistics to {self.figures_dir}.")
        self.next(self.setup_model)

    @step
    def setup_model(self):
        """
        Setup the model based on architecture defined in the YAML file.
        """
        learning_rate, dataset, architecture = self.input

        def modify_first_conv_layer(model):
            model.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(
                in_channels=1, out_channels=model.segformer.encoder.patch_embeddings[0].proj.out_channels,
                kernel_size=model.segformer.encoder.patch_embeddings[0].proj.kernel_size,
                stride=model.segformer.encoder.patch_embeddings[0].proj.stride,
                padding=model.segformer.encoder.patch_embeddings[0].proj.padding,
                bias=False
            )
            return model

        if architecture.startswith("mit-"):
            # Load Segformer configuration for MiT architectures from the 'models' section
            model_config = self.config['models'][architecture]  # Ensure correct access to the 'models' key
            config = SegformerConfig(
                num_labels=1,  # Adjust as needed for your task
                hidden_sizes=model_config['hidden_sizes'],
                depths=model_config['depths'],
                decoder_hidden_size=model_config['decoder_hidden_size'],
                aspp_ratios=model_config['aspp_ratios'] if model_config['use_aspp'] else None,
                aspp_out_channels=model_config['aspp_out_channels'] if model_config['use_aspp'] else None
            )
            model = SegformerForSemanticSegmentation(config)
            model = modify_first_conv_layer(model)  # Update first layer for single-channel input
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.model = model
        self.learning_rate = learning_rate

        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train the SegFormer model with early stopping and WandB monitoring.
        """
        torch.cuda.empty_cache()

        # ** Ensure WandB is initialized ** #
        if not wandb.run:
            run_name = (
                f"{self.config['logging']['project_name']}_"
                f"{self.architecture}_"
                f"LR{self.learning_rate}_"
                f"batch{self.config['training']['batch_size']}"
            ).replace('/', '_')
            wandb.init(project=self.config['logging']['project_name'], name=run_name)

            # Track key parameters explicitly in WandB
            wandb.config.update({
                "data_directory": self.dataset_path,
                "batch_size": self.config['training']['batch_size'],
                "learning_rate": self.learning_rate,
                "num_epochs": self.config['training']['num_epochs'],
                "architecture": self.architecture,
            })

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        # pos_weight for handling class imbalance
        pos_weight = torch.tensor([9.0], device=device)  # Ensure pos_weight is on the correct device

        # Initialize BCEWithLogitsLoss with pos_weight
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        def compute_loss(outputs, targets):
            logits = F.interpolate(outputs.logits, size=targets.shape[1:], mode="bilinear", align_corners=False)
            logits = logits.to(targets.device)  # Ensure logits and targets are on the same device
            targets = targets.float()  # Convert to float
            loss = loss_fn(logits.squeeze(1), targets)  # Ensure targets and logits are properly handled
            return loss

        best_val_loss = float('inf')
        early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping_patience'],
            verbose=True
        )

        scaler = torch.cuda.amp.GradScaler()
        torch.cuda.empty_cache()

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            # Training loop
            for step, (images, masks) in enumerate(
                    tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")):
                images, masks = images.to(device), masks.long().squeeze(1).to(device)

                optimizer.zero_grad()

                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.model(pixel_values=images, labels=masks)
                    loss = compute_loss(outputs, masks)

                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

                torch.cuda.empty_cache()

            # Calculate average training loss for this epoch
            avg_train_loss = running_loss / len(self.train_loader)

            # Validation loop
            val_loss = 0.0
            self.model.eval()

            with torch.no_grad():
                for step, (val_images, val_masks) in enumerate(self.val_loader):
                    val_images, val_masks = val_images.to(device), val_masks.squeeze(1).to(device)

                    # Forward pass through the model
                    val_outputs = self.model(pixel_values=val_images)

                    # Compute the validation loss for this batch
                    batch_val_loss = compute_loss(val_outputs, val_masks).item()

                    # Accumulate the validation loss for this epoch
                    val_loss += batch_val_loss

                    # Print validation loss for this step
                    print(f"Validation Step {step + 1}/{len(self.val_loader)}, Batch Validation Loss: {batch_val_loss:.4f}")

            avg_val_loss = val_loss / len(self.val_loader)

            # Log the average losses for the epoch
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            wandb.log({
                "epoch_train_loss": avg_train_loss,
                "epoch_val_loss": avg_val_loss,
                "epoch": epoch + 1
            })

            # Adjust learning rate using the scheduler
            scheduler.step(avg_val_loss)

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(self.model_dir,
                                          f'best_segformer_epoch_{epoch + 1}_val_loss_{avg_val_loss:.4f}.pth')
                torch.save(self.model.state_dict(), model_path)

            # Early stopping
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        print("Training completed.")
        # Save training artifacts (model and metrics)
        self.avg_train_loss = avg_train_loss
        self.avg_val_loss = avg_val_loss
        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Join step after parallel processing to combine results.
        """
        print("Joining results from all parallel experiments")

        # Aggregating results for learning_rate, dataset, and architecture
        self.results = {f"experiment_{i}": {
            "learning_rate": inp.learning_rate,
            "dataset": inp.dataset,
            "architecture": inp.architecture  # Ensure architecture is passed and stored correctly
        } for i, inp in enumerate(inputs)}

        self.next(self.end)

    @step
    def end(self):
        """
        Final step to indicate the flow has completed.
        """
        print("Flow completed successfully.")

if __name__ == "__main__":
    SeismicLineSegmentationFlow()
