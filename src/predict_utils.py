import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
import matplotlib.pyplot as plt
from rasterio.windows import Window

def adjust_window(window, max_width, max_height):
    col_off, row_off, width, height = map(int, window.flatten())
    width = min(width, max_width - col_off)
    height = min(height, max_height - row_off)
    return Window(col_off, row_off, width, height)

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def pad_to_shape(array, target_shape):
    diff_h = target_shape[0] - array.shape[0]
    diff_w = target_shape[1] - array.shape[1]
    return np.pad(array, ((0, diff_h), (0, diff_w), (0, 0)), mode='constant')

def predict_patch(model, patch):
    patch = normalize_image(patch) if patch.max() != patch.min() else patch
    patch = patch.reshape(1, *patch.shape)
    pred = model.predict(patch).squeeze()
    return (pred * 100).astype(np.uint8)

def plot_prediction(patch, prediction):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(patch.squeeze(), cmap='gray')
    axs[1].imshow(prediction, cmap='jet', vmin=0, vmax=100)
    plt.show()
