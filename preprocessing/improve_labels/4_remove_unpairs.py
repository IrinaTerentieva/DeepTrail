import os

# Path to the directory containing image and label files
directory = '/home/irina/HumanFootprint/DATA/TrainingCNN/UNet_patches1024_nDTM10cm_connected'

# List all files in the directory
all_files = os.listdir(directory)

# Separate image and label files
image_files = {f for f in all_files if f.endswith('_image.tif')}
label_files = {f for f in all_files if f.endswith('_label.tif')}

# Extract common base names (without the suffix)
image_bases = {f.split('_image.tif')[0] for f in image_files}
label_bases = {f.split('_label.tif')[0] for f in label_files}

# Find files without a matching pair
unmatched_images = [f for f in image_files if f.split('_image.tif')[0] not in label_bases]
unmatched_labels = [f for f in label_files if f.split('_label.tif')[0] not in image_bases]

# Remove unmatched files
for unmatched_file in unmatched_images + unmatched_labels:
    file_path = os.path.join(directory, unmatched_file)
    os.remove(file_path)
    print(f"Removed: {file_path}")

print("Cleanup completed. All unmatched files removed.")
