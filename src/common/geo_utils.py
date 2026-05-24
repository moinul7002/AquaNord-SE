import pyproj
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point

# Standard projection for Hydro-Dataset (ETRS89 / LAEA Europe)
EPSG_LAEA = 3035
WGS84 = 4326

def get_laea_transformer() -> Transformer:
    """
    Returns a pyproj transformer from WGS84 to ETRS89-LAEA.
    """
    return Transformer.from_crs(WGS84, EPSG_LAEA, always_xy=True)

def project_points_to_laea(df, lon_col='longitude', lat_col='latitude'):
    """
    Transforms a pandas DataFrame with lon/lat into a GeoDataFrame projected in LAEA.
    """
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, crs=f"EPSG:{WGS84}", geometry=geometry)
    return gdf.to_crs(epsg=EPSG_LAEA)

def get_nordic_extent_laea():
    """
    Returns the bounding box for the Nordic region study area in LAEA coordinates.
    """
    # EPSG:3035 (ETRS89-LAEA) bounding box covering Finland + Sweden inland
    # x (easting):  3 800 000 – 4 800 000  (~10°E–32°E at 60°N)
    # y (northing): 3 500 000 – 5 600 000  (~55°N–71°N)
    return {
        "min_x": 3_800_000,
        "max_x": 4_800_000,
        "min_y": 3_500_000,
        "max_y": 5_600_000,
    }
