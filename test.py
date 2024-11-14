import geopandas as gpd
import numpy as np


def process_geopackage(file_path):
    try:
        # Read the GeoPackage file with geopandas
        gdf = gpd.read_file(file_path)

        # Find all non-geometry columns
        non_geometry_columns = gdf.columns.difference([gdf.geometry.name])

        # Identify the last non-geometry column
        last_column = non_geometry_columns[-1]

        # Replace 'None' values with 0s, round any numeric values, and convert to integers
        gdf[last_column] = gdf[last_column].replace([None, 'None'], 0)
        gdf[last_column] = gdf[last_column].apply(lambda x: np.round(float(x)) if isinstance(x, (str, float)) else x)
        gdf[last_column] = gdf[last_column].astype(int)

        # Print the modified GeoDataFrame
        print(gdf.head())

        # Save back to file if needed
        gdf.to_file(file_path, driver="GPKG", index=False)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# Example usage
file_path = "/media/irro/Data/2022/RECCAP-2/russia_test.gpkg"
process_geopackage(file_path)
