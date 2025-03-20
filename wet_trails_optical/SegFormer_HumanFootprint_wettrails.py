import os
import sys
import shutil
import yaml
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb
from torch.utils.checkpoint import checkpoint

# Add the parent directory (where utils_segformer.py is located) to the system path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_segformer import (
    WetTrailsDataset, EarlyStopping, build_augmentations, calculate_statistics,
    compute_class_weights, save_image_mask_pair
)

# Set environment variables for CUDA memory management and to disable Albumentations update checks.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["WANDB_SILENT"] = "false"
torch.cuda.empty_cache()

def forward_with_checkpoint(model, pixel_values, labels=None):
    """
    Wraps the forward pass of the model with checkpointing.
    Note: Checkpointing requires that the wrapped function only takes tensors as input.
    """
    def custom_forward(x):
        # Here we assume model.forward returns an object with attribute 'logits'
        return model(pixel_values=x, labels=labels).logits
    # Run checkpoint on the custom forward function
    logits = checkpoint(custom_forward, pixel_values)
    # Create a dummy output to mimic the model output structure
    class DummyOutput:
        def __init__(self, logits):
            self.logits = logits
    return DummyOutput(logits)

def main():
    # --- Load Configuration ---
    base_dir = "/"
    config_path = os.path.join(base_dir, 'config_segformer.yaml')
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Experiment Settings ---
    learning_rate = config['experiments']['learning_rates'][0]
    dataset_name = list(config['experiments']['datasets'].keys())[0]
    dataset_path = os.path.join(base_dir, config['experiments']['datasets'][dataset_name])

    dataset_path = "/media/irina/My Book/Surmont/TrainingCNN/wettrail_RGB10cm_512px"

    architecture = config['experiments']['architectures'][0]
    model_config = config['models'][architecture]
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training']['early_stopping_patience']
    model_dir = os.path.join(base_dir, 'Models')

    # --- Additional Training Options ---
    # Gradient accumulation: how many mini-batches to accumulate before an optimizer step.
    accumulation_steps = config['training'].get('accumulation_steps', 4)
    # Use gradient checkpointing to reduce memory usage:
    use_checkpoint = config['training'].get('use_checkpoint', False)

    # --- Initialize WandB ---
    wandb_key_path = os.path.join(base_dir, 'Scripts', 'training', 'wandb_key.txt')
    with open(wandb_key_path, 'r') as f:
        wandb_key = f.read().strip()
    wandb.login(key=wandb_key)
    run_name = f"WetTrails_RGB10cm_dataset_{dataset_name}_arch_{architecture}_lr_{learning_rate}_batch_{batch_size}"
    wandb.init(project='WetTrails_RGB10cm', name=run_name)
    wandb.config.update(config)

    # --- Load Dataset and Create DataLoaders ---
    dataset = WetTrailsDataset(data_dir=dataset_path, threshold=0.1, skip_threshold = 0.01)
    print(f"Total dataset length: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32,
                            pin_memory=True)

    # --- Visualize a Few Validation Samples ---
    figures_dir = os.path.join(dataset_path, 'Figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    image_statistics = []
    mask_statistics = []
    shape_check = []

    class_imbalance = {"class_0": [], "class_1": []}
    for idx, (image, mask) in enumerate(val_loader):
        if idx >= 15:
            break
        image_np = image[0].cpu().numpy().squeeze()
        mask_np = mask[0].cpu().numpy().squeeze()
        shape_check.append((image_np.shape, mask_np.shape))
        print(f"Image {idx+1} shape: {image_np.shape}, Mask {idx+1} shape: {mask_np.shape}")
        img_stats, msk_stats = calculate_statistics(image_np, mask_np)
        image_statistics.append(img_stats)
        mask_statistics.append(msk_stats)
        total_pixels = mask_np.size
        class_0_percent = (msk_stats.get(0, 0) / total_pixels) * 100
        class_1_percent = (msk_stats.get(1, 0) / total_pixels) * 100
        class_imbalance["class_0"].append(class_0_percent)
        class_imbalance["class_1"].append(class_1_percent)
        print(f"Class 0 percentage: {class_0_percent:.2f}%, Class 1 percentage: {class_1_percent:.2f}%")
        pair_figure_path = save_image_mask_pair(image_np, mask_np, idx, figures_dir)
        print(f"Saved validation sample {idx+1} to {pair_figure_path}")

    # --- Setup Model ---
    def modify_first_conv_layer(model, in_channels=3):
        # Modify the first convolution to accept the specified number of channels.
        model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=model.segformer.encoder.patch_embeddings[0].proj.out_channels,
            kernel_size=model.segformer.encoder.patch_embeddings[0].proj.kernel_size,
            stride=model.segformer.encoder.patch_embeddings[0].proj.stride,
            padding=model.segformer.encoder.patch_embeddings[0].proj.padding,
            bias=False
        )
        return model

    print('\n*******************working with: ', architecture)

    if architecture.startswith("mit-"):
        config_model = config['models'][architecture]
        segformer_config = SegformerConfig(
            num_labels=config_model['num_labels'],
            hidden_sizes=config_model['hidden_sizes'],
            depths=config_model['depths'],
            decoder_hidden_size=config_model['decoder_hidden_size'],
            aspp_ratios=config_model.get('aspp_ratios', None) if config_model.get('use_aspp', False) else None,
            aspp_out_channels=config_model.get('aspp_out_channels', None) if config_model.get('use_aspp', False) else None,
            attention_dropout_prob=config_model.get('attention_dropout_prob', 0.1)
        )
        model = SegformerForSemanticSegmentation(segformer_config)
        model = modify_first_conv_layer(model, in_channels=3)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # --- Training Setup ---
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # learning_rate = 0.00001
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                     patience=1, threshold=0.0001, cooldown=1)
    class_weights = compute_class_weights(train_loader, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))

    def compute_loss(outputs, targets):
        logits = F.interpolate(outputs.logits, size=targets.shape[1:], mode="bilinear", align_corners=False)
        return loss_fn(logits.squeeze(1), targets.float())

    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    scaler = torch.amp.GradScaler()

    # --- Training Loop with Gradient Accumulation and Optional Checkpointing ---
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, masks = images.to(device), masks.long().squeeze(1).to(device)
            # Optionally use checkpointing on the forward pass.
            if use_checkpoint:
                outputs = forward_with_checkpoint(model, images, labels=masks)
            else:
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values=images, labels=masks)
            loss = compute_loss(outputs, masks) / accumulation_steps
            scaler.scale(loss).backward()
            running_loss += loss.item() * accumulation_steps

            # Gradient accumulation: update weights every accumulation_steps iterations.
            if (step_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # In case the number of steps is not an exact multiple, do an update after the loop.
        if (step_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"epoch_train_loss": avg_train_loss, "learning_rate": current_lr, "epoch": epoch+1})

        # --- Log Additional Validation Samples ---
        val_batch = next(iter(val_loader))
        val_images, val_masks = val_batch
        val_images, val_masks = val_images.to(device), val_masks.squeeze(1).to(device)
        with torch.no_grad():
            val_outputs = model(pixel_values=val_images)
            val_preds = torch.sigmoid(val_outputs.logits).cpu().numpy()

        for i in range(min(3, val_images.shape[0])):
            # Convert from CHW to HWC for visualization if necessary.
            img_np = val_images[i].cpu().numpy()  # shape: (3, 512, 512)
            img_np = np.transpose(img_np, (1, 2, 0))  # shape: (512, 512, 3)
            wandb.log({
                f"Validation Sample {i} Input": wandb.Image(img_np, caption=f"Epoch {epoch + 1} Input"),
                f"Validation Sample {i} Ground Truth": wandb.Image(val_masks[i].cpu().numpy().squeeze(),
                                                                   caption=f"Epoch {epoch + 1} GT Mask"),
                f"Validation Sample {i} Prediction": wandb.Image(val_preds[i].squeeze(),
                                                                 caption=f"Epoch {epoch + 1} Pred Mask")
            })

        model_save_dir = os.path.join(
            model_dir,
            f"SegFormer_HumanFootprint_dataset_{dataset_name}_arch_{architecture}_lr_{learning_rate}_batch_{batch_size}_epoch_{epoch + 1}"
        )
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # --- Save Epoch Predictions (always) ---
        epoch_pred_dir = os.path.join(model_save_dir, f"Predictions")
        os.makedirs(epoch_pred_dir, exist_ok=True)
        for i in range(min(3, val_images.shape[0])):
            image_np = val_images[i].cpu().numpy().squeeze()
            gt_mask = val_masks[i].cpu().numpy().squeeze()
            pred_mask = val_preds[i].squeeze()
            input_path = os.path.join(epoch_pred_dir, f"ep{epoch+1}_sample_{i}_input.png")
            gt_path = os.path.join(epoch_pred_dir, f"ep{epoch+1}_sample_{i}_gt.png")
            pred_path = os.path.join(epoch_pred_dir, f"ep{epoch+1}_sample_{i}_pred.png")

            print("Before transpose, image_np shape:", image_np.shape)
            if image_np.ndim == 3 and image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))

            plt.imsave(input_path, image_np)
            plt.imsave(gt_path, gt_mask, cmap='gray')
            plt.imsave(pred_path, pred_mask, cmap='gray')

            print(f"Epoch {epoch+1}: Saved input sample {i} to {input_path}")
            print(f"Epoch {epoch+1}: Saved GT sample {i} to {gt_path}")
            print(f"Epoch {epoch+1}: Saved prediction sample {i} to {pred_path}")

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.squeeze(1).to(device)
                outputs = model(pixel_values=images)
                batch_val_loss = compute_loss(outputs, masks).item()
                val_loss += batch_val_loss
        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch_val_loss": avg_val_loss, "epoch": epoch + 1})
        scheduler.step(avg_val_loss)

        # Save the model and validation outputs only if validation loss improves
        if avg_val_loss < best_val_loss + 0.005:
            best_val_loss = avg_val_loss

            # Save model weights and configuration
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'pytorch_model_weights.pth'))
            if isinstance(model, nn.DataParallel):
                model.module.save_pretrained(model_save_dir)
            else:
                model.save_pretrained(model_save_dir)
            print(f"Model and configuration saved in: {model_save_dir}")
            shutil.copy(config_path, os.path.join(model_save_dir, f"segformer_epoch_{epoch + 1}_config.yaml"))
            shutil.copy(__file__, os.path.join(model_save_dir, f"segformer_epoch_{epoch + 1}_training.py"))
            wandb_run_info = {
                "wandb_run_id": wandb.run.id,
                "wandb_run_name": wandb.run.name,
                "wandb_project": wandb.run.project,
                "wandb_url": wandb.run.url
            }
            with open(os.path.join(model_save_dir, f"wandb_run_metadata_epoch_{epoch + 1}.yaml"), 'w') as f:
                yaml.dump(wandb_run_info, f)

            # Save validation outputs (predictions and ground truth) for a few samples
            val_output_dir = os.path.join(model_save_dir, "validation_outputs")
            os.makedirs(val_output_dir, exist_ok=True)
            for i in range(min(3, val_images.shape[0])):
                image_np = val_images[i].cpu().numpy().squeeze()  # Expected shape: (3, H, W) for RGB
                gt_mask = val_masks[i].cpu().numpy().squeeze()  # Expected shape: (H, W)
                pred_mask = val_preds[i].squeeze()  # Expected shape: (H, W)

                # Convert image from CHW to HWC if necessary
                if image_np.ndim == 3 and image_np.shape[0] in [3, 4]:
                    image_np = np.transpose(image_np, (1, 2, 0))

                # Save the image without a colormap (it's RGB)
                plt.imsave(os.path.join(val_output_dir, f"sample_{i}_input.png"), image_np)
                # Save ground truth and predicted masks with a grayscale colormap
                plt.imsave(os.path.join(val_output_dir, f"sample_{i}_gt.png"), gt_mask, cmap='gray')
                plt.imsave(os.path.join(val_output_dir, f"sample_{i}_pred.png"), pred_mask, cmap='gray')

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training completed.")
    # Clean up non-picklable objects before exit
    del train_loader
    del val_loader
    wandb.finish()

if __name__ == "__main__":
    main()
