import os
import hydra
from omegaconf import DictConfig
import rasterio
import numpy as np
from rasterio.windows import Window
from src.predict_utils import pad_to_shape, predict_patch, adjust_window
from tensorflow.keras.models import load_model
from src.metrics import iou, iou_thresholded, jaccard_coef, dice_coef
from huggingface_hub import hf_hub_download


def load_trail_model(model_path, custom_objects):
    if model_path.startswith("hf://"):
        _, repo_path = model_path.split("hf://")
        model_path = hf_hub_download(repo_id=repo_path, filename="model.h5")
    return load_model(model_path, custom_objects=custom_objects)


@hydra.main(config_path="configs", config_name="predict", version_base=None)
def main(cfg: DictConfig):
    model = load_trail_model(cfg.model_name, custom_objects={
        'iou': iou,
        'iou_thresholded': iou_thresholded,
        'jaccard_coef': jaccard_coef,
        'dice_coef': dice_coef
    })

    with rasterio.open(cfg.input_tif) as src:
        height, width = src.height, src.width
        patch_size = model.input_shape[1]
        stride = patch_size - cfg.overlap_size

        if stride <= 0:
            raise ValueError(f"Invalid overlap_size={cfg.overlap_size}. Must be less than patch_size={patch_size}.")

        prediction = np.zeros((height, width), dtype=np.uint8)

        for row in range(0, height, stride):
            for col in range(0, width, stride):
                window = adjust_window(Window(col, row, patch_size, patch_size), width, height)
                patch = src.read(window=window)
                patch = np.moveaxis(patch, 0, -1)

                if patch.shape[:2] != (patch_size, patch_size):
                    patch = pad_to_shape(patch, (patch_size, patch_size))

                pred = predict_patch(model, patch)

                col_off, row_off, width_win, height_win = map(int, window.flatten())
                pred = pred[:height_win, :width_win]

                prediction[row_off:row_off+height_win, col_off:col_off+width_win] = np.maximum(
                    prediction[row_off:row_off+height_win, col_off:col_off+width_win],
                    pred
                )

        # Save to central output folder with model info in filename
        model_tag = os.path.basename(cfg.model_name).replace(".h5", "").replace("/", "_")
        input_name = os.path.splitext(os.path.basename(cfg.input_tif))[0]
        output_dir = os.path.join(os.getcwd(), "folder_preds")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{input_name}_{model_tag}_pred.tif")

        profile = src.profile
        profile.update(dtype='uint8', count=1, compress='lzw', nodata=0)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction, 1)

        print(f"Prediction saved to {output_path}")


if __name__ == "__main__":
    main()
