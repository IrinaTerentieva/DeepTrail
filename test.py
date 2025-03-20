import os
import rioxarray
import dask.array as da
from dask.diagnostics import ProgressBar
from dask_cuda import LocalCUDACluster
from dask.distributed import Client


def process_file(filename, input_folder, output_folder, dst_crs="EPSG:2956", dst_res=0.1):
    """
    Resamples and reprojects a TIFF file using rioxarray, dask, and Cupy for GPU acceleration.
    Note: The GDAL-based reprojection is CPU-bound, so only some parts of the processing may run on the GPU.
    """
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # Open the file with rioxarray using dask for chunking.
    ds = rioxarray.open_rasterio(input_path, chunks={"band": 1, "x": 1024, "y": 1024})

    # Optional: Convert the underlying dask array to a GPU-backed Cupy array.
    # This will convert each chunk to a Cupy array. Note that if the subsequent operations
    # (like reprojection) use GDAL functions, they will still run on the CPU.
    try:
        import cupy as cp
        ds.data = ds.data.map_blocks(cp.asarray)
    except ImportError:
        print("Cupy is not installed. Proceeding without GPU conversion.")

    # Reproject and resample using the specified resolution and resampling method.
    ds_reprojected = ds.rio.reproject(
        dst_crs,
        resolution=dst_res,
        resampling='average'
    )

    # Write the output file. Dask will compute the necessary tasks.
    ds_reprojected.rio.to_raster(output_path)
    print(f"Processed {filename} with rioxarray and dask (GPU conversion attempted)")


def main():
    input_folder = "/media/irina/My Book1/Conoco/DATA/8_orthos_shifted_cropped"
    output_folder = "/media/irina/My Book1/Conoco/DATA/8_orthos_shifted_cropped_10cm"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all TIFF files (case-insensitive)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]

    # Process each file.
    for filename in files:
        process_file(filename, input_folder, output_folder)


if __name__ == "__main__":
    # Create a dask-cuda cluster to leverage available GPUs.
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print("Dask CUDA cluster created:", client)

    with ProgressBar():
        main()
