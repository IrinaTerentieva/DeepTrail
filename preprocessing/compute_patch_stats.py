import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import yaml
import glob


def compute_distribution(image_path, label_path):
    """
    Load an image and its corresponding label, then return the pixel values for:
    - Trail (label==1) regions.
    - Background (label==0) regions.
    """
    with rasterio.open(image_path) as src:
        image = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            # Replace nodata with np.nan so they are not counted in statistics
            image = np.where(image == nodata, np.nan, image)
    with rasterio.open(label_path) as src:
        mask = src.read(1).astype(np.uint8)
    trail_pixels = image[mask == 1]
    background_pixels = image[mask == 0]
    return trail_pixels, background_pixels


def main():
    # --- Set Directories and Parameters (no YAML) ---
    # Base directory of your project
    base_dir = "/home/irina/HumanFootprint"

    # Directory containing your training images and label files.
    # For example, images are named like: "64_image.tif" and labels "64_label.tif"
    data_dir = os.path.join(base_dir, "DATA", "Training_CNN", "synth_tracks_1024px_10cm")

    # Output directory where statistics and histograms will be saved
    output_dir = os.path.join(base_dir, "DATA", "Statistics")
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files (assuming they end with _image.tif)
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_image.tif")))
    if not image_files:
        print(f"No image files found in {data_dir}")
        return

    all_trail = []
    all_background = []

    # Process each image-label pair
    for image_path in image_files:
        # Derive label path
        label_path = image_path.replace("_image.tif", "_label.tif")
        if not os.path.exists(label_path):
            print(f"Label file not found for image: {image_path}")
            continue
        print(f"Processing image: {os.path.basename(image_path)}")
        trail_pixels, background_pixels = compute_distribution(image_path, label_path)
        if trail_pixels.size > 0:
            all_trail.append(trail_pixels)
        if background_pixels.size > 0:
            all_background.append(background_pixels)

    # Concatenate all pixel arrays
    all_trail = np.concatenate(all_trail) if all_trail else np.array([])
    all_background = np.concatenate(all_background) if all_background else np.array([])

    # Compute summary statistics
    stats = {}
    if all_trail.size > 0:
        stats["trail"] = {
            "min": float(np.nanmin(all_trail)),
            "max": float(np.nanmax(all_trail)),
            "mean": float(np.nanmean(all_trail)),
            "std": float(np.nanstd(all_trail))
        }
    else:
        stats["trail"] = "No trail pixels found."

    if all_background.size > 0:
        stats["background"] = {
            "min": float(np.nanmin(all_background)),
            "max": float(np.nanmax(all_background)),
            "mean": float(np.nanmean(all_background)),
            "std": float(np.nanstd(all_background))
        }
    else:
        stats["background"] = "No background pixels found."

    # Save statistics to a YAML file
    stats_path = os.path.join(output_dir, "distribution_stats.yaml")
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f)
    print(f"Distribution statistics saved to {stats_path}")

    # Save histograms for trail and background pixels
    if all_trail.size > 0:
        plt.hist(all_trail[~np.isnan(all_trail)], bins=100, color='r', alpha=0.7)
        plt.title("Trail Pixels Distribution")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        trail_hist_path = os.path.join(output_dir, "trail_histogram.png")
        plt.savefig(trail_hist_path)
        plt.clf()
        print(f"Trail histogram saved to {trail_hist_path}")

    if all_background.size > 0:
        plt.hist(all_background[~np.isnan(all_background)], bins=100, color='b', alpha=0.7)
        plt.title("Background Pixels Distribution")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        background_hist_path = os.path.join(output_dir, "background_histogram.png")
        plt.savefig(background_hist_path)
        plt.clf()
        print(f"Background histogram saved to {background_hist_path}")

    # --- Normalization Update (Example) ---
    # For instance, if you want to use the computed background min and max for normalization,
    # you might update your normalization function as follows:
    #
    # def normalize_nDTM(image):
    #     new_min = stats["background"]["min"]
    #     new_max = stats["background"]["max"]
    #     image = np.clip(image, new_min, new_max)
    #     return (image - new_min) / (new_max - new_min)
    #
    # Then, re-run training/inference with this updated normalization.
    #
    # (The above is just an example. Choose the appropriate method based on your inspection
    #  of the distribution statistics.)


if __name__ == "__main__":
    main()
