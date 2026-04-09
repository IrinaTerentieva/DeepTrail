"""
Single-file DTM -> nDTM converter.

nDTM is computed by subtracting a local mean surface from the DTM.
The local window radius is given in meters and is converted to pixels
based on the raster's pixel size. A mean (uniform) filter is used so the
operation is O(N) regardless of window size, which is essential for
large rasters at fine (e.g. 10 cm) resolution.
"""

import argparse
import os

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


def dtm_to_ndtm(input_path: str, output_path: str, radius_m: float = 6.0) -> str:
    """Convert a DTM raster to a normalized DTM (nDTM).

    Parameters
    ----------
    input_path : str
        Path to the input DTM GeoTIFF.
    output_path : str
        Path to write the nDTM GeoTIFF.
    radius_m : float
        Radius of the local median window, in meters.

    Returns
    -------
    str
        The output path.
    """
    with rasterio.open(input_path) as src:
        data = src.read(1, masked=True)
        profile = src.profile
        pixel_size = abs(src.transform.a)

    radius_pix = int(round(radius_m / pixel_size))
    window_size = 2 * radius_pix + 1
    print(f"Pixel size: {pixel_size} m  ->  window: {window_size}x{window_size} px "
          f"(radius {radius_m} m)")

    # Replace NaNs with the global mean so the mean filter doesn't propagate them
    filled_data = data.filled(np.nan).astype(np.float32)
    if np.isnan(filled_data).any():
        mean_val = float(np.nanmean(filled_data))
        filled_data = np.nan_to_num(filled_data, nan=mean_val)

    print("Computing local mean surface...")
    local_mean = uniform_filter(filled_data, size=window_size, mode="reflect")

    nDTM = filled_data - local_mean

    profile.update({
        "dtype": rasterio.float32,
        "driver": "GTiff",
        "count": 1,
        "nodata": np.nan,
        "compress": "lzw",
    })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(nDTM.astype(np.float32), 1)

    print(f"nDTM saved to: {output_path}")
    return output_path


def _default_output_path(input_path: str, radius_m: float) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_nDTM_{int(radius_m)}m{ext}"


def main():
    parser = argparse.ArgumentParser(description="Convert a DTM to a normalized DTM (nDTM).")
    parser.add_argument("input", help="Path to input DTM GeoTIFF")
    parser.add_argument("-o", "--output", default=None,
                        help="Path to output nDTM GeoTIFF (default: <input>_nDTM_<radius>m.tif)")
    parser.add_argument("-r", "--radius", type=float, default=6.0,
                        help="Local median window radius in meters (default: 6.0)")
    args = parser.parse_args()

    output = args.output or _default_output_path(args.input, args.radius)
    dtm_to_ndtm(args.input, output, radius_m=args.radius)


if __name__ == "__main__":
    main()
