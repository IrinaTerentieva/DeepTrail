import os
import hydra
import geopandas as gpd
import rasterio
from omegaconf import DictConfig
from rasterio.windows import from_bounds
from rasterio.transform import rowcol

@hydra.main(config_path="configs", config_name="preprocess", version_base=None)
def main(cfg: DictConfig):
    generate_and_save_patches(
        shp=cfg.shapefile,
        tif_path=cfg.train_tif,
        mask_path=cfg.mask_tif,
        size=cfg.patch_size,
        base_dir=cfg.patch_dir,
        idnumber_start=cfg.start_id
    )

def generate_and_save_patches(shp, tif_path, mask_path, size, base_dir, idnumber_start=0):
    tif_array, mask_array, bounds, transform = align_and_crop(tif_path, mask_path)
    os.makedirs(base_dir, exist_ok=True)
    df = shp_to_patches(shp, tif_path, size)
    print(f'\nNumber of points within the shapefile: {len(df)}')

    idnumber = idnumber_start
    num_patches = 0

    for index, row in df.iterrows():
        patch = tif_array[:, int(row.py_min):int(row.py_max), int(row.px_min):int(row.px_max)]
        patch_l = mask_array[:, int(row.py_min):int(row.py_max), int(row.px_min):int(row.px_max)]

        if patch.shape[1:] > (size, size) or patch_l.shape[1:] > (size, size):
            patch = patch[:, :size, :size]
            patch_l = patch_l[:, :size, :size]
        elif patch.shape[1:] != (size, size) or patch_l.shape[1:] != (size, size):
            print(f"Skipping tile {idnumber} due to improper shape: {patch.shape}, {patch_l.shape}.")
            continue

        filename_p = os.path.join(base_dir, f'{idnumber}_img.tif')
        filename_l = os.path.join(base_dir, f'{idnumber}_lab.tif')

        save_geotiff(filename_p, patch, tif_path, row.px_min, row.py_min, size)
        save_geotiff(filename_l, patch_l, tif_path, row.px_min, row.py_min, size)

        num_patches += 1
        idnumber += 1

    print(f'Total patches generated and saved: {num_patches}')

def rowcol_to_pixelbounds(cx, cy, transform, half_size):
    px, py = rowcol(transform, cx, cy)
    return (px - half_size, py - half_size, px + half_size, py + half_size)

def align_and_crop(tif_path, mask_path):
    with rasterio.open(tif_path) as tif, rasterio.open(mask_path) as mask:
        left = max(tif.bounds.left, mask.bounds.left)
        bottom = max(tif.bounds.bottom, mask.bounds.bottom)
        right = min(tif.bounds.right, mask.bounds.right)
        top = min(tif.bounds.top, mask.bounds.top)

        tif_window = from_bounds(left, bottom, right, top, tif.transform)
        mask_window = from_bounds(left, bottom, right, top, mask.transform)

        tif_array = tif.read(window=tif_window)
        mask_array = mask.read(window=mask_window)

        min_height = min(tif_array.shape[1], mask_array.shape[1])
        min_width = min(tif_array.shape[2], mask_array.shape[2])
        tif_array = tif_array[:, :min_height, :min_width]
        mask_array = mask_array[:, :min_height, :min_width]

        return tif_array, mask_array, (left, bottom, right, top), tif.transform

def save_geotiff(filename, data, reference_tif, x_offset, y_offset, patch_size):
    with rasterio.open(reference_tif) as src:
        meta = src.meta.copy()
        new_origin_x, _ = src.transform * (x_offset, 0)
        _, new_origin_y = src.transform * (0, y_offset)

        new_transform = rasterio.transform.from_origin(new_origin_x, new_origin_y, src.res[0], src.res[1])
        meta.update({
            'driver': 'GTiff',
            'height': patch_size,
            'width': patch_size,
            'transform': new_transform,
            'count': data.shape[0]
        })

        with rasterio.open(filename, 'w', **meta) as dst:
            if data.ndim == 3 and data.shape[0] == 1:
                dst.write(data[0], 1)
            elif data.ndim == 3:
                dst.write(data)
            else:
                raise ValueError("Unexpected data shape for writing: ", data.shape)

def shp_to_patches(shp, tif_path, size=256):
    with rasterio.open(tif_path) as tif:
        left, bottom, right, top = tif.bounds
        resolution = tif.res[0]
        transform = tif.transform
        gdf = gpd.read_file(shp).to_crs(tif.crs)
        gdf_filtered = gdf.cx[left:right, bottom:top]

        half_width = (size * resolution) / 2
        gdf_filtered["image_path"] = tif_path
        gdf_filtered["center_x"] = gdf_filtered.geometry.x
        gdf_filtered["center_y"] = gdf_filtered.geometry.y
        gdf_filtered["tile_xmin"] = gdf_filtered["center_x"] - half_width
        gdf_filtered["tile_xmax"] = gdf_filtered["center_x"] + half_width
        gdf_filtered["tile_ymin"] = gdf_filtered["center_y"] - half_width
        gdf_filtered["tile_ymax"] = gdf_filtered["center_y"] + half_width

        gdf_filtered['px_min'] = None
        gdf_filtered['px_max'] = None
        gdf_filtered['py_min'] = None
        gdf_filtered['py_max'] = None

        for index, row in gdf_filtered.iterrows():
            py_min, px_min = rowcol(transform, row['tile_xmin'], row['tile_ymin'])
            py_max, px_max = rowcol(transform, row['tile_xmax'], row['tile_ymax'])
            gdf_filtered.at[index, 'px_min'] = px_min
            gdf_filtered.at[index, 'px_max'] = px_max
            gdf_filtered.at[index, 'py_max'] = py_min
            gdf_filtered.at[index, 'py_min'] = py_max

        gdf_filtered = gdf_filtered[
            (gdf_filtered['px_min'] >= 0) & (gdf_filtered['px_max'] <= tif.width) &
            (gdf_filtered['py_min'] >= 0) & (gdf_filtered['py_max'] <= tif.height)
        ]

        result = gdf_filtered[['image_path', 'center_x', 'center_y', 'px_min', 'py_min', 'px_max', 'py_max']].copy()
        print('Check Dataframe\n*****', result.head(1))

    return result
