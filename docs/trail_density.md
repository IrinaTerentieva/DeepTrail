# Trail Density from GeoPackage Pipeline

This document provides a comprehensive overview of the Trail Density Pipeline implemented in the script [`trail_density_from_gpkg.py`](../postprocessing/trail_density_from_gpkg.py). This pipeline converts vector-based trail data into a trail density map through rasterization, density computation, and merging, with intermediate cleanup.

---

## Overview

The pipeline performs the following key steps:

1. **Rasterization of Trails:**  
   Converts vector trail data (from a GeoPackage) into binary rasters, where trail geometries are burned into a grid (value `1` for trails, `0` for background).

2. **Density Calculation:**  
   Aggregates the binary raster into coarser cells (e.g., 20-meter cells) to compute a density value representing the fraction of each cell covered by trails.

3. **Merging Density Rasters:**  
   Since the rasterization is performed in parallel on data subsets, multiple density rasters are generated. These are then merged (by taking the maximum per pixel) onto a common grid.

4. **Cleanup:**  
   Removes all intermediate TIFF files after producing the final density map.

---

## Requirements

Before running the script, ensure you have the following Python packages installed:

- **geopandas**
- **numpy**
- **rasterio**
- **tqdm**

You can install the external dependencies via pip:

```bash
pip install geopandas numpy rasterio tqdm
