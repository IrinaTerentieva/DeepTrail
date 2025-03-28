"""
Module: run_trailformer.py

This module provides a command-line entry point for running a sliding-window inference
using a SegFormer-based model on user-specified images. It leverages Hydra for configuration
management, including model parameters, data paths, and inference settings.

Usage:
    # From your project root (assuming Hydra config is in `config/`):
    python run_trailformer.py

    # Or override config parameters at runtime:
    python run_trailformer.py inference.to_predict="path/to/another_image.tif"
"""

import os
import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path

# So Python can discover local modules if needed.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.load_model import load_segformer
from inference import sliding_window_prediction
from utils.rasters import list_tif_files
from hydra.utils import get_original_cwd


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to perform segmentation inference using a SegFormer model.

    This function:
      1) Initializes paths and output directories from the Hydra configuration.
      2) Loads a trained model (either local or from Hugging Face).
      3) Collects TIFF images to predict from a user-specified path.
      4) Runs patch-wise sliding-window prediction on each image.
      5) Saves the output predictions as GeoTIFF files.

    Args:
        cfg (DictConfig):
            The Hydra configuration object containing:
              - paths.output_dir (str): Where output files will be stored.
              - inference.to_predict (str): File or directory path(s) to .tif images.
              - inference.patch_size (int): The size of each inference patch.
              - inference.overlap_size (int): Overlap (in pixels) between patches.
              - inference.threshold (float): Probability threshold to apply for postprocessing.
              - model: Model loading parameters (local vs HF, config options, etc).

    Returns:
        None

    Example:
        >>> # Example usage from command line:
        >>> # python run_trailformer.py inference.to_predict="/path/to/image.tif"
    """

    # Hydra changes the current working directory by default.
    # For logging/debugging, we show the new Hydra working directory:
    print("[DEBUG] Hydra CWD:", Path.cwd())

    # Print the entire config for transparency/debugging
    print(cfg)

    # The original working directory before Hydra changed it:
    original_cwd = Path(get_original_cwd())

    # Build the absolute output directory from config
    output_dir = original_cwd / cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Display relevant inference settings
    print("\n********* Configuration *********")
    print(f"Output Directory: {output_dir}")
    print(f"Patch Size: {cfg.inference.patch_size}")
    print(f"Overlap Size: {cfg.inference.overlap_size}")
    print(f"Threshold: {cfg.inference.threshold}")

    # Load the SegFormer model according to config (local or HuggingFace)
    model = load_segformer(cfg.model)

    # Collect all .tif files from the user-specified path
    to_predict = cfg.inference.to_predict
    image_files = list_tif_files(to_predict)
    print(f"\n[INFO] Found {len(image_files)} images for prediction")

    # Perform patch-wise inference on each image
    for img_path in image_files:
        sliding_window_prediction(
            image_path=img_path,
            model=model,
            output_dir=str(output_dir),
            patch_size=cfg.inference.patch_size,
            overlap_size=cfg.inference.overlap_size,
            threshold=cfg.inference.threshold
        )

    print("[INFO] Prediction flow completed.")


if __name__ == '__main__':
    main()
