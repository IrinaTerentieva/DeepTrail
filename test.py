import os
import shutil


def move_files(source_folder, destination_folder):
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: The folder '{source_folder}' does not exist.")
        return

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Move all files from source to destination
    for filename in os.listdir(source_folder):
        old_path = os.path.join(source_folder, filename)
        new_path = os.path.join(destination_folder, filename)

        # Move the file
        shutil.move(old_path, new_path)
        print(f"Moved: {filename} -> {destination_folder}")


# Specify the folder paths
source_folder = "/media/irina/My Book/Surmont/TrainingCNN/synth_tracks_1024px_10cm_v3.1"
destination_folder = "/media/irina/My Book/Surmont/TrainingCNN/synth_tracks_1024px_10cm_v3"
move_files(source_folder, destination_folder)
