import os
import glob
import yaml  # PyYAML package for loading configuration from a YAML file
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from tqdm import tqdm
from multiprocessing import Pool

# ------------------------------------------------------------------------------
# Load configuration parameters from the YAML file
# ------------------------------------------------------------------------------
with open("/home/irina/HumanFootprint/Scripts/config/trail_density.yaml", "r") as f:
    PARAMS = yaml.safe_load(f)

# Create necessary output directories if they don't exist.
os.makedirs(PARAMS["raster_output_folder"], exist_ok=True)
os.makedirs(PARAMS["density_output_folder"], exist_ok=True)

# ------------------------------------------------------------------------------
# STEP 1: RASTERIZE TRAILS
# ------------------------------------------------------------------------------

def rasterize_trails(args):
    """
    Rasterizes a subset of trail geometries into a binary raster file.

    Parameters:
        args (tuple): Contains the following items:
            - subset_trails (GeoDataFrame): A subset of trail features to rasterize.
            - output_tif (str): Path to save the output raster.
            - transform (affine.Affine): Georeferencing transform for the raster.
            - width (int): Raster width in pixels.
            - height (int): Raster height in pixels.
            - crs (dict): Coordinate Reference System for the raster.

    Returns:
        str: Output file path if successful; None if an error occurs.
    """
    subset_trails, output_tif, transform, width, height, crs = args
    try:
        # Rasterize the trail geometries: assign value 1 for trail pixels, 0 for background.
        trail_raster = rasterize(
            [(geom, 1) for geom in subset_trails.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Save the generated raster to disk.
        with rasterio.open(output_tif, 'w', driver='GTiff',
                           height=height, width=width, count=1,
                           dtype=np.uint8, crs=crs, transform=transform) as dst:
            dst.write(trail_raster, 1)

        return output_tif
    except Exception as e:
        print(f"Error rasterizing {output_tif}: {e}")
        return None

def rasterize_trails_main():
    """
    Manages the rasterization of trails:
      - Reads the input trail data.
      - Applies filtering based on the trail threshold.
      - Calculates raster dimensions.
      - Splits the data for parallel processing.
      - Rasterizes each data subset concurrently.
    """
    # Load trail data using GeoPandas.
    trails = gpd.read_file(PARAMS["trails_path"])
    # Filter trails: only include those with 'value' > trail_threshold.
    trails = trails[trails.value > PARAMS["trail_threshold"]]
    print("Loaded Trails CRS:", trails.crs)

    # Calculate bounding box and raster dimensions based on raster_resolution.
    minx, miny, maxx, maxy = trails.total_bounds
    width = int((maxx - minx) / PARAMS["raster_resolution"])
    height = int((maxy - miny) / PARAMS["raster_resolution"])
    transform = from_origin(minx, maxy, PARAMS["raster_resolution"], PARAMS["raster_resolution"])

    # Split the trails GeoDataFrame into multiple subsets for parallel processing.
    num_subsets = PARAMS["num_workers"] * 2
    trails_split = np.array_split(trails, num_subsets)

    # Create output file paths for each subset raster.
    output_tifs = [
        os.path.join(PARAMS["raster_output_folder"], f"{PARAMS['name']}_part_{i}.tif")
        for i in range(len(trails_split))
    ]

    print(f"Rasterizing {len(trails_split)} subsets with {PARAMS['num_workers']} CPUs...")

    # Use multiprocessing to rasterize subsets concurrently.
    with Pool(PARAMS["num_workers"]) as pool:
        results = list(tqdm(
            pool.imap(
                rasterize_trails,
                [(trails_split[i], output_tifs[i], transform, width, height, trails.crs)
                 for i in range(len(trails_split))]
            ),
            total=len(trails_split),
            desc="Rasterizing Trails"
        ))

    print(f"Finished rasterizing. {len([r for r in results if r])} raster files saved.")

# ------------------------------------------------------------------------------
# STEP 2: COMPUTE DENSITY
# ------------------------------------------------------------------------------

def compute_density_raster(tif_path):
    """
    Computes trail density for a given binary raster.
    Aggregates the raster into coarser cells (based on cell_size) and calculates the density
    as the ratio of trail pixels to the total number of pixels in each cell.

    Parameters:
        tif_path (str): Path to the binary trail raster file.

    Returns:
        str: Path to the saved density raster file; None if an error occurs.
    """
    try:
        # Open the binary raster file.
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()  # Copy the raster profile.
            transform = src.transform
            raster_res = src.res[0]  # Assumes square pixels.
            width, height = src.width, src.height
            minx, maxy = transform.c, transform.f
            maxx = minx + (width * raster_res)
            miny = maxy - (height * raster_res)

            # Calculate scaling factor: how many original pixels per new cell.
            scale_factor = int(PARAMS["cell_size"] / raster_res)
            out_width = width // scale_factor
            out_height = height // scale_factor
            out_transform = from_origin(minx, maxy, PARAMS["cell_size"], PARAMS["cell_size"])

            # Read the binary trail raster data.
            trail_raster = src.read(1)

            # Initialize an empty array for the density raster.
            density_raster = np.zeros((out_height, out_width), dtype=np.float32)

            # Compute the density for each aggregated cell.
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

        # Define the output file path for the density raster.
        output_path = os.path.join(
            PARAMS["density_output_folder"],
            os.path.basename(tif_path).replace(".tif", "_density.tif")
        )
        # Remove transform from profile and update it for the new density raster.
        profile.pop("transform", None)
        profile.update(dtype="float32", compress="lzw", count=1,
                       height=out_height, width=out_width)

        # Save the density raster.
        with rasterio.open(output_path, 'w', **profile, transform=out_transform) as dst:
            dst.write(density_raster, 1)

        return output_path
    except Exception as e:
        print(f"Error computing density for {tif_path}: {e}")
        return None

def compute_density_main():
    """
    Computes density for all rasterized trail files:
      - Finds all raster files in the raster output folder.
      - Uses multiprocessing to compute density for each file.
      - Merges individual density rasters into a final output.
    """
    # Gather all binary raster files from the raster output folder.
    tif_files = [
        os.path.join(PARAMS["raster_output_folder"], f)
        for f in os.listdir(PARAMS["raster_output_folder"]) if f.endswith(".tif")
    ]
    print(f"Found {len(tif_files)} raster files for density calculation.")

    # Compute density for each raster file in parallel.
    with Pool(PARAMS["num_workers"]) as pool:
        results = list(tqdm(
            pool.imap(compute_density_raster, tif_files),
            total=len(tif_files),
            desc="Computing Density"
        ))

    print(f"Finished density processing. {len([r for r in results if r])} density rasters saved.")

    # Merge the individual density rasters into one final density map.
    merge_density_rasters()

def merge_density_rasters():
    """
    Merges all individual density rasters into a single final density map.
    Each raster is reprojected onto a common grid based on the union of all bounds,
    and the maximum density value is taken for overlapping pixels.
    """
    # List all density raster files.
    density_tifs = [
        os.path.join(PARAMS["density_output_folder"], f)
        for f in os.listdir(PARAMS["density_output_folder"]) if f.endswith("_density.tif")
    ]

    if not density_tifs:
        print("No density files found for merging.")
        return

    # Calculate the union of bounds from all density rasters.
    bounds_list = []
    for tif in density_tifs:
        with rasterio.open(tif) as src:
            bounds_list.append(src.bounds)

    # Determine the union bounds.
    minx = min(b.left for b in bounds_list)
    miny = min(b.bottom for b in bounds_list)
    maxx = max(b.right for b in bounds_list)
    maxy = max(b.top for b in bounds_list)

    # Use resolution and CRS from the first density raster.
    with rasterio.open(density_tifs[0]) as src:
        resolution = src.res[0]
        dst_crs = src.crs

    # Define target grid dimensions and transformation.
    out_width = int(np.ceil((maxx - minx) / resolution))
    out_height = int(np.ceil((maxy - miny) / resolution))
    out_transform = from_origin(minx, maxy, resolution, resolution)

    # Initialize an empty array for the merged raster.
    merged = np.zeros((out_height, out_width), dtype=np.float32)

    # Reproject each density raster onto the common grid and merge them.
    for tif in tqdm(density_tifs, desc="Merging Density Layers"):
        with rasterio.open(tif) as src:
            src_array = src.read(1)
            dst_array = np.zeros((out_height, out_width), dtype=np.float32)
            reproject(
                source=src_array,
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            # Merge by taking the maximum density per pixel.
            merged = np.maximum(merged, dst_array)

    # Retrieve and update the profile for the final merged output.
    with rasterio.open(density_tifs[0]) as src:
        profile = src.profile.copy()
    profile.update({
        "driver": "GTiff",
        "height": out_height,
        "width": out_width,
        "transform": out_transform,
        "dtype": "float32"
    })

    # Save the final merged density map.
    with rasterio.open(PARAMS["final_output_tif"], 'w', **profile) as dst:
        dst.write(merged, 1)

    print(f"✅ Final density raster saved at: {PARAMS['final_output_tif']}")

def remove_intermediate_files():
    """
    Removes all intermediate .tif files from the raster and density output folders.
    This cleanup is done after the final density map is produced.
    """
    for folder in [PARAMS["raster_output_folder"], PARAMS["density_output_folder"]]:
        tif_files = glob.glob(os.path.join(folder, "*.tif"))
        for tif in tif_files:
            try:
                os.remove(tif)
                print(f"Removed: {tif}")
            except Exception as e:
                print(f"Error removing {tif}: {e}")

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting Trail Rasterization & Density Computation Pipeline 🚀")
    rasterize_trails_main()
    compute_density_main()
    # Remove intermediate files after final density map is produced.
    remove_intermediate_files()
    print("✅ All processes completed successfully!")
