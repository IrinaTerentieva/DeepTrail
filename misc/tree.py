import os


def is_valid_folder(path):
    """Check if folder contains .py or .yaml files and does not start with 'run'."""
    folder_name = os.path.basename(path)
    if folder_name.startswith("run"):
        return False

    for file in os.listdir(path):
        if file.endswith(".py") or file.endswith(".yaml") or file.endswith(".yml"):
            return True
    return False


def list_valid_folders(start_path):
    for root, dirs, files in os.walk(start_path):
        if is_valid_folder(root):
            print(f"📁 {root}")
            for file in files:
                if file.endswith(".py") or file.endswith(".yaml") or file.endswith(".yml"):
                    print(f"    📄 {file}")


# Run it
list_valid_folders("/home/irina/HumanFootprint")
