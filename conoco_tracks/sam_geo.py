import os
import leafmap
from samgeo import SamGeo, tms_to_geotiff

m = leafmap.Map(center=[37.6412, -122.1353], zoom=15, height="800px")
m.add_basemap("SATELLITE")
m