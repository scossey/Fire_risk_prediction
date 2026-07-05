"""
gee_utils.py
------------
Google Earth Engine helper functions for the fire risk prediction pipeline.
Handles composite construction, label generation, and patch fetching via
computePixels (no export to Drive/GCS required).

Band stack (10 bands, order preserved):
    Red, Green, Blue, SWIR1, SWIR2, NDVI, NDWI, LST, Elevation, Slope
"""

import ee
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple


PROJECTION = "EPSG:32629"   # UTM Zone 29N — covers mainland Portugal
SCALE      = 20             # metres, matches Sentinel-2 10/20 m bands
BANDS      = [
    "Red", "Green", "Blue",
    "SWIR1", "SWIR2",
    "NDVI", "NDWI",
    "LST",
    "Elevation", "Slope",
]
N_BANDS = len(BANDS)        # 10


# Cloud masking

def mask_sentinel2_clouds(image: ee.Image) -> ee.Image:
    """Mask cloud and cirrus pixels using the Sentinel-2 QA60 band."""
    bit_mask = (1 << 10) | (1 << 11)           # cloud | cirrus
    mask = image.select("QA60").bitwiseAnd(bit_mask).eq(0)
    return image.updateMask(mask)


# Date helpers

def get_biweekly_date_ranges(year: int) -> List[Tuple[str, str]]:
    """
    Return bi-weekly (start, end) date-string pairs covering the fire season
    April 1 – October 31 for the given year.

    Produces ~13 periods per year.
    """
    ranges   = []
    current  = datetime(year, 4, 1)
    season_end = datetime(year, 10, 31)

    while current < season_end:
        period_end = min(current + timedelta(days=14), season_end)
        ranges.append((
            current.strftime("%Y-%m-%d"),
            period_end.strftime("%Y-%m-%d"),
        ))
        current = period_end

    return ranges


# Static layers (computed once, reused across all composites)

def _build_static_layers() -> ee.Image:
    """DEM-derived elevation and slope — time-invariant, built once."""
    dem       = ee.Image("USGS/SRTMGL1_003").select("elevation")
    elevation = dem.rename("Elevation").resample("bilinear").reproject(PROJECTION, None, SCALE).float()
    slope     = ee.Terrain.slope(dem).rename("Slope").resample("bilinear").reproject(PROJECTION, None, SCALE).float()
    return elevation.addBands(slope)


_STATIC_LAYERS = None   # lazy initialised on first composite build


def _get_static_layers() -> ee.Image:
    global _STATIC_LAYERS
    if _STATIC_LAYERS is None:
        _STATIC_LAYERS = _build_static_layers()
    return _STATIC_LAYERS


# Composite builder

def build_composite(start_date: str, end_date: str) -> ee.Image:
    """
    Build a single bi-weekly multiband composite with 10 bands:
        Red, Green, Blue, SWIR1, SWIR2, NDVI, NDWI, LST, Elevation, Slope

    Missing data (clouds, sensor gaps) is filled with 0 via .unmask().

    The CLOUDY_PIXEL_PERCENTAGE scene-level filter is intentionally omitted.
    Sentinel-2 has near-daily coverage of Portugal so the collection is
    never empty; the per-pixel QA60 masking already removes cloud pixels.
    Adding a scene-level filter risks producing empty collections over short
    two-week windows during cloudy periods, which causes downstream band
    errors. MODIS LST has genuine daily gaps (cloud cover, orbit), so its
    collection is guarded with an ee.Algorithms.If fallback.

    Args:
        start_date: inclusive start in 'YYYY-MM-DD'.
        end_date:   exclusive end   in 'YYYY-MM-DD'.

    Returns:
        ee.Image with bands in BANDS order, float32.
    """
    S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

    # Sentinel-2 QA60 per-pixel masking only
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .map(mask_sentinel2_clouds)
        .select(S2_BANDS)
        .median()
        .unmask(0)
        .float()
    )

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI").unmask(0).float()
    ndwi = s2.normalizedDifference(["B8", "B11"]).rename("NDWI").unmask(0).float()

    # MODIS LST
    lst_col = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate(start_date, end_date)
        .select("LST_Day_1km")
    )
    lst = ee.Image(
        ee.Algorithms.If(
            lst_col.size().gt(0),
            lst_col.mean()
                .multiply(0.02).subtract(273.15)
                .rename("LST")
                .resample("bilinear")
                .reproject(PROJECTION, None, SCALE)
                .unmask(0)
                .float(),
            ee.Image.constant(0).rename("LST").float(),
        )
    )

    s2_bands = (
        s2.select(["B4", "B3", "B2", "B11", "B12"])
        .rename(["Red", "Green", "Blue", "SWIR1", "SWIR2"])
    )

    return s2_bands.addBands([ndvi, ndwi, lst]).addBands(_get_static_layers())


# Label builder

def build_label(start_date: str, end_date: str) -> ee.Image:
    """
    Build a binary burned-area label from MODIS MCD64A1.

    Any pixel confirmed burned within [start_date, end_date] → 1, others → 0.

    MCD64A1 is a monthly product whose system:time_start falls on the first
    of each month. GEE's filterDate keeps images where time_start >= start,
    so a bi-weekly window starting mid-month (e.g. '2016-05-15') misses the
    May product (time_start = May 1) and returns an empty collection.

    To guarantee at least one product is captured, the query start is snapped
    to the first of the month containing start_date. An ee.Algorithms.If
    guard handles any remaining edge cases (e.g. April — first month of season
    — if no fires were recorded that month in any year).

    Args:
        start_date: inclusive start in 'YYYY-MM-DD'.
        end_date:   exclusive end   in 'YYYY-MM-DD'.

    Returns:
        Single-band ee.Image named 'FireMask', float32, values 0 or 1.
    """
    from datetime import datetime

    # Snap to month start so filterDate always captures the monthly product
    month_start = datetime.strptime(start_date, "%Y-%m-%d").replace(day=1).strftime("%Y-%m-%d")

    col = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(month_start, end_date)
        .select("BurnDate")
        .map(lambda img: img.gt(0).copyProperties(img, img.propertyNames()))
    )

    return ee.Image(
        ee.Algorithms.If(
            col.size().gt(0),
            col.max()
                .reproject(PROJECTION, None, SCALE)
                .rename("FireMask")
                .unmask(0)
                .float(),
            ee.Image.constant(0).rename("FireMask").float(),
        )
    )


# Patch fetching

def _wgs84_to_utm(lon: float, lat: float, utm_crs: str = PROJECTION):
    """Convert WGS84 lon/lat to UTM easting/northing using pyproj."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing


def fetch_patch(
    image: ee.Image,
    center_lon: float,
    center_lat: float,
    patch_size: int = 64,
    scale: int = SCALE,
    projection: str = PROJECTION,
) -> np.ndarray:
    """
    Fetch a spatial patch from GEE as a NumPy array using computePixels.
    No export to Drive or GCS required — data is streamed directly.

    The patch is centred on (center_lon, center_lat) in WGS84 and returned
    in the requested UTM projection at the given scale.

    Bounding box is computed locally via pyproj to avoid GEE geometry
    reprojection errors.

    Args:
        image:       ee.Image to sample (any number of bands).
        center_lon:  Longitude of patch centre (WGS84 decimal degrees).
        center_lat:  Latitude  of patch centre (WGS84 decimal degrees).
        patch_size:  Edge length of the square patch in pixels (default 64).
        scale:       Pixel size in metres (default 20).
        projection:  CRS string (default EPSG:32629).

    Returns:
        NumPy array of shape (patch_size, patch_size, n_bands), float32.
    """
    # Convert centre to UTM locally — no GEE geometry call needed
    half_metres = (patch_size * scale) / 2
    cx, cy = _wgs84_to_utm(center_lon, center_lat, projection)
    x_min  = cx - half_metres
    y_max  = cy + half_metres

    # Build the pixel grid request
    pixels = ee.data.computePixels({
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": patch_size, "height": patch_size},
            "affineTransform": {
                "scaleX":     scale,
                "shearX":     0,
                "translateX": x_min,
                "shearY":     0,
                "scaleY":    -scale,    # negative = top-left origin
                "translateY": y_max,
            },
            "crsCode": projection,
        },
    })

    # computePixels returns a structured numpy array; convert to (H, W, C)
    band_names = list(pixels.dtype.names)
    return np.stack([pixels[b] for b in band_names], axis=-1).astype(np.float32)