import os
import hydra
from omegaconf import DictConfig
import re
import rasterio
import numpy as np
from rasterio.windows import Window

from src.predict_utils import pad_to_shape, predict_patch, adjust_window
from src.metrics import iou, iou_thresholded, jaccard_coef, dice_coef
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

def load_trail_model(cfg, custom_objects):
    if cfg.trail_mapping.model_source == "hf":
        model_path = hf_hub_download(
            repo_id=cfg.trail_mapping.hf_repo_id,
            filename=cfg.trail_mapping.hf_filename
        )
    else:
        model_path = cfg.trail_mapping.model_weights_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
    return load_model(model_path, custom_objects=custom_objects)


@hydra.main(config_path="configs", config_name="inference", version_base=None)
def main(cfg: DictConfig):
    model = load_trail_model(cfg, custom_objects={
        'iou': iou,
        'iou_thresholded': iou_thresholded,
        'jaccard_coef': jaccard_coef,
        'dice_coef': dice_coef
    })

    # Only override if not explicitly set
    if not cfg.trail_mapping.patch_size:
        match = re.search(r'_(\d+)px$', cfg.trail_mapping.model_name)
        if match:
            cfg.trail_mapping.patch_size = int(match.group(1))
        else:
            raise ValueError("Could not infer patch size from model_name. Set trail_mapping.patch_size manually.")


    input_tif = cfg.paths_to_predict.input_to_predict_file
    with rasterio.open(input_tif) as src:
        height, width = src.height, src.width
        patch_size = cfg.trail_mapping.patch_size
        stride = patch_size - cfg.trail_mapping.overlap
        print(f'Working with {cfg.trail_mapping.patch_size} patch_size')
        print(f'Working with {stride} stride')

        if stride <= 0:
            raise ValueError(f"Invalid overlap. Must be less than patch size ({patch_size}).")

        prediction = np.zeros((height, width), dtype=np.uint8)

        # Row starts
        if height <= patch_size:
            row_starts = [0]
        else:
            row_starts = list(range(0, height - patch_size + 1, stride))
            if row_starts[-1] + patch_size < height:
                row_starts.append(height - patch_size)

        col_starts = list(range(0, width - patch_size + 1, stride))
        if col_starts[-1] + patch_size < width:
            col_starts.append(width - patch_size)

        for row in row_starts:
            for col in col_starts:
                window = adjust_window(Window(col, row, patch_size, patch_size), width, height)
                patch = src.read(window=window)
                patch = np.moveaxis(patch, 0, -1)

                if patch.shape[:2] != (patch_size, patch_size):
                    patch = pad_to_shape(patch, (patch_size, patch_size))

                pred = predict_patch(model, patch)

                col_off, row_off, width_win, height_win = map(int, window.flatten())
                pred = pred[:height_win, :width_win]

                prediction[row_off:row_off + height_win, col_off:col_off + width_win] = np.maximum(
                    prediction[row_off:row_off + height_win, col_off:col_off + width_win],
                    pred
                )

        # Save output in a subfolder next to the input file
        model_tag = cfg.trail_mapping.model_name
        input_name = os.path.splitext(os.path.basename(input_tif))[0]
        input_folder = os.path.dirname(input_tif)
        output_dir = os.path.join(input_folder, cfg.trail_mapping.output_subfolder)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{input_name}_{model_tag}_pred.tif")

        profile = src.profile
        profile.update(dtype='uint8', count=1, compress='lzw', nodata=0)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction, 1)

        print(f"Prediction saved to {output_path}")


if __name__ == "__main__":
    main()
