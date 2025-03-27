import os

def print_tree(start_path, prefix=""):
    for item in os.listdir(start_path):
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            print(f"{prefix}📁 {item}")
            print_tree(path, prefix + "    ")
        else:
            print(f"{prefix}📄 {item}")

# Run it
print_tree("/home/irina/HumanFootprint")
