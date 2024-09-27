import importlib

packages = [
    'tensorflow', 'matplotlib', 'tifffile', 'numpy', 'PIL',
    'fiona', 'datetime', 'tensorflow.keras', 'keras', 'tensorflow_addons',
    'rasterio', 'os', 'pandas', 'cv2', 'geopandas', 'tensorflow.compat.v1'
]

for package in packages:
    try:
        importlib.import_module(package)
        print(f'{package} is installed')
    except ImportError as e:
        print(f'{package} is NOT installed: {e}')
