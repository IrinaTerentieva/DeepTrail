import os


def rename_tif_files(directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            old_path = os.path.join(directory, filename)
            new_filename = "3.20_" + filename  # Add 'x' as the first letter
            new_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")


# Define the directory path
directory_path = "/media/irina/My Book/Surmont/TrainingCNN/synth_tracks_1024px_10cm_v3.2/"
rename_tif_files(directory_path)
