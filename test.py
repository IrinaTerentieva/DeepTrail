import sys
import tensorflow as tf
import torch
import numpy as np

print("Python Version:", sys.version)
print("NumPy Version:", np.__version__)

# Check TensorFlow
try:
    print("\nTensorFlow Version:", tf.__version__)
    print("TensorFlow Built with CUDA:", tf.test.is_built_with_cuda())
    print("TensorFlow Built with cuDNN:", tf.test.is_built_with_gpu_support())
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))
except ImportError:
    print("TensorFlow not installed")

# Check PyTorch
try:
    print("\nPyTorch Version:", torch.__version__)
    print("PyTorch CUDA Available:", torch.cuda.is_available())
    print("PyTorch GPU Count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("PyTorch GPU Name:", torch.cuda.get_device_name(0))
except ImportError:
    print("PyTorch not installed")
