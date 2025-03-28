# Usage Instructions

This document explains how to run the **HumanFootprint** prediction tool and how to customize its behavior using command-line configuration overrides.

## Overview

The main entry point for predictions is the script **`run_trailformer.py`**. This script performs patch-wise sliding-window inference on input GeoTIFF images using a SegFormer-based model. It leverages [Hydra](https://hydra.cc/) for flexible configuration management, so you can easily override parameters at runtime.

## Running the Prediction Script

### Basic Usage

To run the prediction with the default settings (as defined in your Hydra config files), simply execute:

```bash
python scripts/run_trailformer.py

____
python scripts/run_trailformer.py \
  inference.to_predict="/data/my_custom_input.tif" \
  inference.patch_size=512 \
  inference.overlap_size=64 \
  inference.threshold=0.15 \
  paths.output_dir="/data/prediction_results"



