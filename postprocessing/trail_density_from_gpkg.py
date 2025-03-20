import os
import geopandas as gpd
import numpy as np
import rasterio
from dask.config import paths
from rasterio.features import rasterize
from rasterio.transform import from_origin
from tqdm import tqdm
from multiprocessing import Pool

# === PARAMETERS (Centralized for easy changes) ===

path = "file:///media/irina/My Book/Surmont/Products/Trails/trailformer_epoch_31_val_loss_0.0592.gpkg"
name = 'trailformer'

path = "file:///media/irina/My Book/Surmont/Products/Trails/TrackFormer_1024_nDTM10cm_arch_mit-b2_lr_0.001_batch_4_epoch_11_v.3.gpkg"
name = 'trackformer'

PARAMS = {
    "trails_path": path,
    "raster_output_folder": "/media/irina/My Book/Surmont/Products/Trails/Rasterized_Trails",
    "density_output_folder": "/media/irina/My Book/Surmont/Products/Trails/Density_Outputs",
    "raster_resolution": 1,  # Trail raster resolution in meters
    "cell_size": 5,  # Density output cell size in meters
    "num_workers": 16,  # Number of CPU cores to use
    "trail_threshold": 20,  # Filter trails with 'value' > threshold
    "name": name
}


# Create output directories
os.makedirs(PARAMS["raster_output_folder"], exist_ok=True)
os.makedirs(PARAMS["density_output_folder"], exist_ok=True)

### STEP 1: RASTERIZE TRAILS ###
def rasterize_trails(args):
    """Rasterizes a subset of trails into a binary raster file."""
    subset_trails, output_tif, transform, width, height, crs = args
    try:
        trail_raster = rasterize(
            [(geom, 1) for geom in subset_trails.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Save raster
        with rasterio.open(output_tif, 'w', driver='GTiff', height=height, width=width, count=1,
                           dtype=np.uint8, crs=crs, transform=transform) as dst:
            dst.write(trail_raster, 1)

        return output_tif
    except Exception as e:
        print(f"Error rasterizing {output_tif}: {e}")
        return None

def rasterize_trails_main():
    """Handles rasterization of trails using multiprocessing."""
    trails = gpd.read_file(PARAMS["trails_path"])
    trails = trails[trails.value > PARAMS["trail_threshold"]]  # Filter trails
    print("Loaded Trails CRS:", trails.crs)

    # Get bounding box and define raster shape
    minx, miny, maxx, maxy = trails.total_bounds
    width = int((maxx - minx) / PARAMS["raster_resolution"])
    height = int((maxy - miny) / PARAMS["raster_resolution"])
    transform = from_origin(minx, maxy, PARAMS["raster_resolution"], PARAMS["raster_resolution"])

    # Split trails into chunks for multiprocessing
    num_subsets = PARAMS["num_workers"] * 2
    name = PARAMS["name"]

    trails_split = np.array_split(trails, num_subsets)
    output_tifs = [os.path.join(PARAMS["raster_output_folder"], f"{name}_part_{i}.tif") for i in range(len(trails_split))]

    print(f"Rasterizing {len(trails_split)} subsets with {PARAMS['num_workers']} CPUs...")

    with Pool(PARAMS["num_workers"]) as pool:
        results = list(tqdm(pool.imap(rasterize_trails,
                                      [(trails_split[i], output_tifs[i], transform, width, height, trails.crs)
                                       for i in range(len(trails_split))]),
                            total=len(trails_split), desc="Rasterizing Trails"))

    print(f"Finished rasterizing. {len([r for r in results if r])} raster files saved.")

### STEP 2: COMPUTE DENSITY ###
def compute_density_raster(tif_path):
    """Computes trail density in 5m x 5m cells and saves as a new raster."""
    try:
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()
            transform = src.transform
            raster_res = src.res[0]  # Pixel resolution (assuming square pixels)
            width, height = src.width, src.height
            minx, maxy = transform.c, transform.f  # Upper-left corner
            maxx = minx + (width * raster_res)
            miny = maxy - (height * raster_res)

            # Define new density grid resolution
            scale_factor = int(PARAMS["cell_size"] / raster_res)
            out_width = width // scale_factor
            out_height = height // scale_factor
            out_transform = from_origin(minx, maxy, PARAMS["cell_size"], PARAMS["cell_size"])

            # Read raster into memory
            trail_raster = src.read(1)

            # Initialize density raster
            density_raster = np.zeros((out_height, out_width), dtype=np.float32)

            # Compute density
            for i in range(out_height):
                for j in range(out_width):
                    x_start = j * scale_factor
                    x_end = x_start + scale_factor
                    y_start = i * scale_factor
                    y_end = y_start + scale_factor

                    cell_array = trail_raster[y_start:y_end, x_start:x_end]
                    total_pixels = cell_array.size
                    trail_pixels = np.count_nonzero(cell_array)
                    density = trail_pixels / total_pixels if total_pixels > 0 else 0

                    density_raster[i, j] = density

        # Save density raster
        output_path = os.path.join(PARAMS["density_output_folder"], os.path.basename(tif_path).replace(".tif", "_density.tif"))
        profile.pop("transform", None)
        profile.update(dtype="float32", compress="lzw", count=1, height=out_height, width=out_width)

        with rasterio.open(output_path, 'w', **profile, transform=out_transform) as dst:
            dst.write(density_raster, 1)

        return output_path
    except Exception as e:
        print(f"Error computing density for {tif_path}: {e}")
        return None

def compute_density_main():
    """Computes density for all rasterized trail files using multiprocessing."""
    tif_files = [os.path.join(PARAMS["raster_output_folder"], f) for f in os.listdir(PARAMS["raster_output_folder"]) if f.endswith(".tif")]
    print(f"Found {len(tif_files)} raster files for density calculation.")

    with Pool(PARAMS["num_workers"]) as pool:
        results = list(tqdm(pool.imap(compute_density_raster, tif_files),
                            total=len(tif_files), desc="Computing Density"))

    print(f"Finished density processing. {len([r for r in results if r])} density rasters saved.")

    # Delete temporary rasterized trails
    print("Deleting intermediate rasterized trail files...")
    for tif in tif_files:
        os.remove(tif)
    print("All temporary files deleted.")

### MAIN EXECUTION ###
if __name__ == "__main__":
    print("🚀 Starting Trail Rasterization & Density Computation Pipeline 🚀")

    # Step 1: Rasterize Trails
    rasterize_trails_main()

    # Step 2: Compute Density from Rasterized Trails
    compute_density_main()

    print("✅ All processes completed successfully!")
