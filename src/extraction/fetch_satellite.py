"""
SW-STEP-5: Satellite spectral bands + derived WQ products for Swedish major lakes.

Sensors:
  MODIS Terra/Aqua  — daily 500m SR via GEE (2000–2025, MOD09GA / MYD09GA)
  Landsat 7/8/9     — ~16-day 30m L2 SR via GEE (2000–2025, C02 T1_L2)
  Sentinel-1 SAR    — ~6-day 10m C-band GRD via GEE (2014–2025, S1_GRD, IW VV+VH)
  Sentinel-2 MSI    — ~5-day 10m SR via GEE (2015–2025)
                       L1C TOA 2015-06-23 to 2017-03-27  (S2_HARMONIZED)
                       L2A SR  2017-03-28 onward          (S2_SR_HARMONIZED)
  Sentinel-3 OLCI   — daily 300m L2 WFR via CDSE (2016–2025, OL_2_WFR___)

Lakes covered:
    HydroLAKES-matched lakes (>= 0.5 km²) that contain monitoring stations.
    If HydroLAKES is unavailable, synthetic lake polygons are generated
    by clustering station locations.

WQ products derived per lake per observation:
  chl_a_ug_l    — Chlorophyll-a (µg/L)
  turbidity_fnu — Turbidity (FNU)
  cdom_m        — CDOM absorption coefficient (m⁻¹)
  secchi_m      — Secchi depth (m, empirical)
  owt_class_v1_provisional — Optical Water Type (4-class; pending Nordic 6-class calibration, MD-6)

Algorithms:
  Chl-a:     OC4 (O'Reilly 1998) for MODIS/Landsat; chl_nn from OLCI L2 product
  Turbidity: Nechad et al. (2010), NIR-based
  CDOM:      log(Rw_blue / Rw_green) proxy
  Secchi:    Kratzer et al. (2008) from Kd(490) or turbidity proxy

Usage:
  python -m src.extraction.fetch_satellite --sensor modis     --start-year 2000 --end-year 2025
  python -m src.extraction.fetch_satellite --sensor landsat   --start-year 2000 --end-year 2025
  python -m src.extraction.fetch_satellite --sensor sentinel1 --start-year 2014 --end-year 2025
  python -m src.extraction.fetch_satellite --sensor sentinel2 --start-year 2015 --end-year 2025
  python -m src.extraction.fetch_satellite --sensor sentinel3 --start-year 2016 --end-year 2025

Output: data/raw/satellite/SE/sweden_{sensor}_{YYYY}.parquet
  Columns: date, lake_id, sensor, n_valid_pixels, cloud_frac, rw_* bands, chl_a_ug_l,
           turbidity_fnu, cdom_m, secchi_m, owt_class_v1_provisional,
           ac_method, rw_blue_negative_flag
"""

import sys
import os
import argparse
import datetime
import zipfile
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import RAW_DIR
from common.logging_utils import setup_logger

logger = setup_logger("SATELLITE")

SAT_DIR        = RAW_DIR / "satellite" / "SE"
HYDROLAKES_DIR = RAW_DIR / "hydroatlas"

# ── Station loading (for --mode stations) ─────────────────────────────────────

# Primary: stations with actual WQ chemistry measurements (consistent with src/patches/ scope)
MEASURED_STATIONS_CSV = RAW_DIR / "slu_mvm" / "slu_mvm_measured_station_ids.csv"
# Fallback: full network CSVs (used only if measured list not yet generated)
STATION_CSVS = [
    RAW_DIR / "hydroobs" / "smhi_hydroobs_stations.csv",
    RAW_DIR / "slu_mvm"  / "slu_mvm_stations.csv",
]


def load_stations() -> pd.DataFrame:
    """
    Load WQ station coordinates for pixel-level satellite extraction.

    Uses slu_mvm_measured_station_ids.csv (stations with actual chemistry
    measurements) to stay consistent with src/patches/ and avoid extracting
    satellite data for stations that have no in-situ WQ labels to join against.
    Falls back to the full network CSVs if the measured list doesn't exist yet.
    """
    if MEASURED_STATIONS_CSV.exists():
        stations = pd.read_csv(
            MEASURED_STATIONS_CSV,
            usecols=["station_id", "latitude", "longitude"]
        ).dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id").reset_index(drop=True)
        logger.info(f"Loaded {len(stations)} measured WQ stations for satellite extraction")
        return stations

    logger.warning(
        f"Measured station list not found ({MEASURED_STATIONS_CSV.name}). "
        "Falling back to full network CSVs — run check_slu_stations.py first."
    )
    frames = []
    for p in STATION_CSVS:
        if p.exists():
            df = pd.read_csv(p, usecols=["station_id", "latitude", "longitude"])
            frames.append(df)
        else:
            logger.warning(f"Station CSV not found: {p}")
    if not frames:
        raise FileNotFoundError(
            "No station CSVs found — run HydObs and SLU MVM extractors first"
        )
    stations = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates(subset=["station_id"])
        .reset_index(drop=True)
    )
    logger.info(f"Loaded {len(stations)} unique WQ stations for satellite extraction")
    return stations


# ── Lake definitions ──────────────────────────────────────────────────────────

# Sweden bounding box (WGS84, west/south/east/north)
SWEDEN_BBOX      = (10.5, 55.0, 24.5, 69.5)
# Used for Sentinel-3 CDSE product search (covers all monitored lakes)
SWEDEN_UNION_BBOX = SWEDEN_BBOX

# Minimum lake area (km²) for stats-mode reduceRegion.
# Below this the lake bbox will contain too few clean water pixels for a
# reliable mean at the sensor's resolution.
# MODIS 500m: ~1 km²;  Landsat 30m: ~0.1 km²;  S2 20m: ~0.05 km²;  S1 10m: ~0.01 km²
# Use 0.5 km² as a conservative shared threshold — filters tiny ponds while
# keeping the hundreds of lakes monitored by HydObs / SLU MVM.
STATS_MIN_LAKE_KM2 = 0.5

# Five largest lakes kept as a hardcoded fallback in case HydroLAKES GEE asset
# is unavailable. stats mode loads all lakes dynamically from HydroLAKES.
SWEDISH_LAKES_FALLBACK = {
    "vanern":   {"name": "Lake Vänern",    "bbox": (12.40, 58.20, 14.20, 59.10), "area_km2": 5655},
    "vattern":  {"name": "Lake Vättern",   "bbox": (14.20, 57.80, 15.00, 58.90), "area_km2": 1912},
    "malaren":  {"name": "Lake Mälaren",   "bbox": (16.00, 59.00, 18.00, 59.80), "area_km2": 1140},
    "hjalmar":  {"name": "Lake Hjälmaren", "bbox": (15.60, 59.00, 16.30, 59.40), "area_km2":  479},
    "storsjon": {"name": "Lake Storsjön",  "bbox": (13.80, 62.70, 14.80, 63.20), "area_km2":  464},
}


# ── WQ derivation ─────────────────────────────────────────────────────────────

def classify_owt(rw_blue: float, rw_green: float, rw_red: float,
                  rw_nir: float, chl_a: Optional[float] = None) -> int:
    """
    Simplified OWT classification for Nordic boreal lakes.
    1=clear, 2=CDOM-rich/humic, 3=turbid, 4=eutrophic
    Most Nordic lakes fall into type 2 (brown-water humic) or 3 (turbid).
    """
    if chl_a is not None and chl_a > 20:
        return 4
    blue_red = rw_blue / max(rw_red, 1e-9)
    if rw_nir > 0.02:
        return 3  # turbid — high NIR backscatter
    if blue_red < 0.8:
        return 2  # CDOM-rich — blue absorbed by humic material
    if blue_red > 2.0 and rw_nir < 0.005:
        return 1  # clear oligotrophic
    return 3      # default turbid for Swedish lakes


def derive_wq_from_bands(rw_blue: float, rw_green: float,
                          rw_red: float, rw_nir: float) -> dict:
    """
    Derive WQ proxies from surface reflectance for MODIS / Landsat.

    Reflectance values are surface reflectance (not water-leaving), so
    these are approximate proxies — not calibrated WQ retrievals.
    Use Sentinel-3 OLCI L2 products for higher-accuracy WQ data.

    Algorithms:
      Turbidity: Nechad et al. (2010), A_T=610, C_T=0.194 (860nm analogue)
      Chl-a:     O'Reilly (1998) OC4 adapted for blue/green MODIS bands
      CDOM:      -0.5 * ln(Rw_blue / Rw_green) proxy
      Secchi:    Kratzer et al. (2008): 1.7 / Kd(490); Kd ≈ turbidity * 0.015 + 0.04

    MD-1: rw_blue_negative_flag=1 signals Sen2Cor/LaSRC over-correction over CDOM-rich
    water; OC4 and CDOM products are unreliable when this flag is set.
    MD-6: owt_class_v1_provisional — 4-class scheme pending Nordic 6-class calibration.
    """
    out: dict = {}

    # MD-1: flag negative blue before clamping — over-correction indicator
    out["rw_blue_negative_flag"] = int(rw_blue < 0)

    # Turbidity (Nechad 2010, NIR-based for turbid-dominated signal)
    nir = max(rw_nir, 0.0)
    out["turbidity_fnu"] = float(np.clip(610.0 * nir / max(1.0 - nir / 0.194, 0.05), 0, 500))

    # Chl-a OC4 (O'Reilly 1998)
    R = np.log10(max(rw_blue, 1e-6) / max(rw_green, 1e-6))
    chl_oc4 = 10 ** (0.366 - 3.067 * R + 1.930 * R**2 + 0.649 * R**3)
    out["chl_a_ug_l"] = float(np.clip(chl_oc4, 0.01, 500.0))

    # CDOM proxy
    out["cdom_m"] = float(np.clip(-0.5 * np.log(max(rw_blue, 1e-6) / max(rw_green, 1e-6)), 0, 20))

    # Secchi from turbidity proxy (Kratzer 2008)
    kd490 = out["turbidity_fnu"] * 0.015 + 0.04
    out["secchi_m"] = float(np.clip(1.7 / max(kd490, 0.1), 0.1, 40.0))

    # MD-6: provisional 4-class OWT; will be replaced by Nordic 6-class calibration
    out["owt_class_v1_provisional"] = classify_owt(
        rw_blue, rw_green, rw_red, rw_nir, out["chl_a_ug_l"]
    )

    return out


# ── GEE helpers ───────────────────────────────────────────────────────────────

def _stations_to_fc(ee, stations: pd.DataFrame):
    """Convert station DataFrame to a GEE FeatureCollection of points."""
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row.longitude), float(row.latitude)]),
            {"station_id": str(row.station_id)},
        )
        for _, row in stations.iterrows()
    ]
    return ee.FeatureCollection(features)


def _generate_lake_polygons_from_stations(stations: pd.DataFrame) -> list:
    """
    Generate synthetic lake polygons from clustering station locations.
    
    Uses spatial grid-based clustering (lat/lon 0.15° bins ≈ 16km × 11km)
    to group nearby stations into "lakes". Each cluster becomes a "synthetic lake"
    with bounding box and a unique ID.
    
    This is used as a fallback when HydroLAKES is unavailable, ensuring
    satellite stats extraction covers ALL monitored stations, not just
    the 5 hardcoded major lakes.
    
    Returns: list of {"lake_id", "name", "area_km2", "bbox", "geometry"}
    """
    if len(stations) == 0:
        logger.warning("No stations available for lake clustering")
        return []
    
    # Grid-based clustering: round lat/lon to 0.15° bins (≈16km × 11km at Sweden)
    grid_size = 0.15
    stations_copy = stations.copy()
    stations_copy["grid_lat"] = (stations_copy["latitude"] // grid_size * grid_size).round(2)
    stations_copy["grid_lon"] = (stations_copy["longitude"] // grid_size * grid_size).round(2)
    
    lakes = []
    for (lat_grid, lon_grid), group_stations in stations_copy.groupby(["grid_lat", "grid_lon"]):
        # Compute bounding box with 0.05° buffer (≈5km)
        buf = 0.05
        lon_min, lon_max = group_stations["longitude"].min(), group_stations["longitude"].max()
        lat_min, lat_max = group_stations["latitude"].min(), group_stations["latitude"].max()
        bbox = (lon_min - buf, lat_min - buf, lon_max + buf, lat_max + buf)
        
        # Estimate "area" from station count (each station ≈ 0.5 km² area of interest)
        area_km2 = len(group_stations) * 0.5
        
        # Generate lake_id from centroid coordinates (e.g., "lake_58_15")
        lon_c = (lon_min + lon_max) / 2
        lat_c = (lat_min + lat_max) / 2
        lake_id = f"cluster_{int(lat_c)}_{int(lon_c)}"
        
        lakes.append({
            "lake_id":   lake_id,
            "name":      f"Clustered lake near ({lat_c:.1f}°N, {lon_c:.1f}°E) — {len(group_stations)} stations",
            "area_km2":  area_km2,
            "bbox":      bbox,
            "geometry":  None,  # No GEE geometry needed for Python-side extraction
        })
    
    logger.info(f"Generated {len(lakes)} synthetic lakes from {len(stations)} stations via grid clustering")
    return lakes


def _gee_init():
    import ee
    project = os.getenv("GEE_PROJECT")
    if not project:
        raise RuntimeError("GEE_PROJECT env var not set")
    ee.Initialize(project=project)
    return ee


def _gee_image_date_str(ee, img):
    """Return YYYY-MM-dd for a GEE Image, parsing system:index if needed."""
    idx = ee.String(img.get("system:index"))
    date_from_index = ee.Algorithms.If(
        idx.length().eq(10),
        ee.Date.parse("YYYY_MM_dd", idx),
        ee.Date.parse("YYYYMMdd", idx.slice(-8)),
    )
    time_start = img.get("system:time_start")
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(time_start, None),
        ee.Date(date_from_index).format("YYYY-MM-dd"),
        ee.Date(time_start).format("YYYY-MM-dd"),
    )


def _load_lake_polygons_local(ee, stations: pd.DataFrame) -> list:
    """
    Load HydroLAKES polygons from the locally downloaded shapefile
    (HYDROLAKES_DIR / *.shp).  Filters to Sweden, area >= STATS_MIN_LAKE_KM2,
    and lakes that contain at least one WQ monitoring station.
    Converts each polygon to an ee.Geometry for downstream reduceRegion calls.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import mapping as _mapping
    except ImportError as exc:
        raise ImportError("geopandas is required for local HydroLAKES loading") from exc

    # LakeATLAS east-hemisphere polygon file (Sweden + Finland are eastern hemisphere)
    shp_files = list(HYDROLAKES_DIR.glob("**/*pol_east.shp"))
    if not shp_files:
        shp_files = list(HYDROLAKES_DIR.glob("**/*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found under {HYDROLAKES_DIR}")
    shp_path = shp_files[0]

    w, s, e, n = SWEDEN_BBOX
    logger.info(f"Reading HydroLAKES from {shp_path.name} (Sweden bbox)…")
    gdf = gpd.read_file(shp_path, bbox=(w, s, e, n))
    gdf = gdf[gdf["Lake_area"] >= STATS_MIN_LAKE_KM2].reset_index(drop=True)

    if gdf.empty:
        raise ValueError("No HydroLAKES features found within Sweden bounding box")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Spatial join: keep only lakes that contain at least one WQ monitoring station
    stations_gdf = gpd.GeoDataFrame(
        stations[["station_id"]],
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(stations_gdf, gdf[["Hylak_id", "geometry"]],
                       how="inner", predicate="within")
    matched_ids = set(joined["Hylak_id"].unique())
    gdf = gdf[gdf["Hylak_id"].isin(matched_ids)].reset_index(drop=True)

    if gdf.empty:
        raise ValueError("No HydroLAKES polygons contain a WQ monitoring station")

    lakes = []
    for _, row in gdf.iterrows():
        geom   = row.geometry
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        # Simplify to 0.001° (~100 m) to keep GEE geometry payloads manageable
        geom_s = geom.simplify(0.001, preserve_topology=True)
        lakes.append({
            "lake_id":  str(row["Hylak_id"]),
            "name":     str(row.get("Lake_name", "")),
            "area_km2": float(row["Lake_area"]),
            "bbox":     (bounds[0], bounds[1], bounds[2], bounds[3]),
            "geometry": ee.Geometry(_mapping(geom_s)),
        })

    logger.info(f"HydroLAKES (local shapefile): {len(lakes)} Swedish lakes with monitoring stations")
    return lakes


def load_lake_polygons_gee(ee, stations: pd.DataFrame) -> list:
    """
    Load Swedish lake polygons, trying sources in priority order:
      1. Local HydroLAKES shapefile  (HYDROLAKES_DIR / *.shp)
      2. HydroLAKES via GEE sat-io community asset
      3. JRC Global Surface Water vectorization (always accessible)

    Returns lakes >= STATS_MIN_LAKE_KM2 that contain at least one WQ station.

    Tries multiple asset paths; falls back to dynamic station grouping.

    Returns a list of dicts:
        [{"lake_id": str, "name": str, "area_km2": float,
          "bbox": (west, south, east, north), "geometry": ee.Geometry}, ...]
    """
    # 1. Local shapefile (fastest, most reliable when available)
    if any(HYDROLAKES_DIR.glob("**/*.shp")):
        try:
            return _load_lake_polygons_local(ee, stations)
        except Exception as exc:
            logger.warning(f"Local HydroLAKES load failed: {exc}; trying GEE asset…")

    w, s, e, n = SWEDEN_BBOX
    sweden_geom = ee.Geometry.Rectangle([w, s, e, n])

    # 2. GEE sat-io community asset (multiple capitalization variants)
    asset_paths = [
        "projects/sat-io/open-datasets/HydroLakes/HydroLAKES_polys_v10",
        "projects/sat-io/open-datasets/hydrolakes/HydroLAKES_polys_v10",
        "projects/sat-io/open-datasets/hydroLAKES/HydroLAKES_polys_v10",
    ]
    
    last_exc = None
    for asset_path in asset_paths:
        try:
            lakes_fc = (
                ee.FeatureCollection(asset_path)
                .filterBounds(sweden_geom)
                .filter(ee.Filter.gte("Lake_area", STATS_MIN_LAKE_KM2))
            )

            # Deduplicate stations to unique grid (0.05°) to stay under 10MB payload
            _unique_st = (
                stations
                .assign(lat_r=stations["latitude"].round(2),
                        lon_r=stations["longitude"].round(2))
                .drop_duplicates(subset=["lat_r", "lon_r"])
                .drop(columns=["lat_r", "lon_r"])
            )
            stations_fc = _stations_to_fc(ee, _unique_st)

            matched = (
                ee.Join.saveFirst("station")
                .apply(
                    lakes_fc,
                    stations_fc,
                    ee.Filter.intersects(leftField=".geo", rightField=".geo", maxError=10),
                )
            )

            info = matched.select(["Hylak_id", "Lake_name", "Lake_area"]).getInfo()
            features = info.get("features", [])
            
            lakes = []
            for feat in features:
                p = feat["properties"]
                coords = feat["geometry"]["coordinates"]
                # Flatten MultiPolygon/Polygon to get bbox
                all_pts = []
                def _collect(c):
                    if isinstance(c[0], (int, float)):
                        all_pts.append(c)
                    else:
                        for sub in c:
                            _collect(sub)
                _collect(coords)
                if not all_pts:
                    continue
                lons = [pt[0] for pt in all_pts]
                lats = [pt[1] for pt in all_pts]
                bbox = (min(lons), min(lats), max(lons), max(lats))
                lakes.append({
                    "lake_id":   str(p.get("Hylak_id", "")),
                    "name":      str(p.get("Lake_name", "")),
                    "area_km2":  float(p.get("Lake_area", 0)),
                    "bbox":      bbox,
                    "geometry":  ee.Geometry(feat["geometry"]),
                })
            
            if lakes:
                logger.info(f"HydroLAKES: {len(lakes)} Swedish lakes with monitoring stations loaded")
                return lakes
        except Exception as exc:
            last_exc = exc
            # Try next path variant
            continue

    # All HydroLAKES paths failed; derive polygons from JRC Global Surface Water
    logger.warning(
        f"HydroLAKES unavailable (last error: {last_exc}); "
        "falling back to JRC Global Surface Water vectorization."
    )
    return _load_lake_polygons_jrc(ee, stations)


def _load_lake_polygons_jrc(ee, stations: pd.DataFrame) -> list:
    """
    Derive lake polygons from JRC Global Surface Water (occurrence >= 80%).

    Vectorizes the permanent-water layer at 500 m over the Sweden bounding box,
    filters to bodies >= STATS_MIN_LAKE_KM2, then keeps only those that contain
    at least one WQ monitoring station.

    JRC/GSW1_4/GlobalSurfaceWater is a first-party GEE dataset — always accessible.
    """
    w, s, e, n = SWEDEN_BBOX
    sweden_geom = ee.Geometry.Rectangle([w, s, e, n])

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    permanent = jrc.gte(80).selfMask()

    # Vectorize at 500 m (matches MODIS; bestEffort avoids timeout over Sweden)
    water_polys = permanent.reduceToVectors(
        reducer=ee.Reducer.countEvery(),
        geometry=sweden_geom,
        scale=500,
        maxPixels=int(1e9),
        bestEffort=True,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="pixel_count",
    )

    # Add area (km²) server-side and filter out sub-threshold bodies
    water_polys = water_polys.map(
        lambda f: f.set("area_km2", f.geometry().area(maxError=100).divide(1e6))
    ).filter(ee.Filter.gte("area_km2", STATS_MIN_LAKE_KM2))

    # Spatial join — keep only polygons that contain a monitoring station
    _unique_st = (
        stations
        .assign(lat_r=stations["latitude"].round(2), lon_r=stations["longitude"].round(2))
        .drop_duplicates(subset=["lat_r", "lon_r"])
        .drop(columns=["lat_r", "lon_r"])
    )
    stations_fc = _stations_to_fc(ee, _unique_st)

    matched = (
        ee.Join.saveFirst("station")
        .apply(
            water_polys,
            stations_fc,
            ee.Filter.intersects(leftField=".geo", rightField=".geo", maxError=10),
        )
    )

    info = matched.getInfo()
    features = info.get("features", [])

    lakes = []
    for idx, feat in enumerate(features):
        p = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        all_pts: list = []

        def _collect(c):
            if isinstance(c[0], (int, float)):
                all_pts.append(c)
            else:
                for sub in c:
                    _collect(sub)

        _collect(coords)
        if not all_pts:
            continue
        lons = [pt[0] for pt in all_pts]
        lats = [pt[1] for pt in all_pts]
        bbox = (min(lons), min(lats), max(lons), max(lats))
        lat_c = (min(lats) + max(lats)) / 2
        lon_c = (min(lons) + max(lons)) / 2
        lakes.append({
            "lake_id":  f"jrc_{idx:05d}",
            "name":     f"JRC water body {lat_c:.2f}N {lon_c:.2f}E",
            "area_km2": float(p.get("area_km2", 0)),
            "bbox":     bbox,
            "geometry": ee.Geometry(feat["geometry"]),
        })

    logger.info(f"JRC GSW: derived {len(lakes)} monitored water bodies (500 m vectorization)")
    return lakes


def _filter_stations_to_lakes(stations: pd.DataFrame) -> pd.DataFrame:
    """
    Keep stations that fall within any of the 5 Swedish major lake bboxes
    (plus a 0.05° ≈ 5 km buffer). Satellite WQ retrieval over narrow rivers
    is unreliable (mixed pixels), so restricting to lake-area stations is
    appropriate for this extraction mode.
    """
    buf = 0.05
    masks = []
    for lake_info in SWEDISH_LAKES_FALLBACK.values():
        w, s, e, n = lake_info["bbox"]
        masks.append(
            (stations["latitude"]  >= s - buf) & (stations["latitude"]  <= n + buf) &
            (stations["longitude"] >= w - buf) & (stations["longitude"] <= e + buf)
        )
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return stations[combined].reset_index(drop=True)


# Hard upper bound on stations per sampleRegions() batch.
# Effective batch size is computed dynamically as floor(4500 / n_unique_dates)
# after daily mosaicking.
_GEE_STATION_BATCH = 500

# When a monthly collection has more images than this threshold, mosaic all
# tiles for each calendar date into a single daily composite before sampling.
# This handles multi-tile sensors like Sentinel-2, where hundreds of
# overlapping tiles cover Sweden per month.  After mosaicking, image count
# drops to at most 31 (one per day), keeping batch sizes manageable.
_MOSAIC_THRESHOLD = 60


def _mosaic_by_date(ee, col):
    """
    Collapse a multi-tile ImageCollection into one composite image per
    calendar date.  Preserves system:time_start on each output image.
    Used for Sentinel-2 which can produce hundreds of tiles per month.
    """
    dates = (
        col.aggregate_array("system:time_start")
        .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
        .distinct()
    )

    def _daily(date_str):
        d = ee.Date(date_str)
        return (
            col.filterDate(d, d.advance(1, "day"))
               .mosaic()
               .set("system:time_start", d.millis())
        )

    return ee.ImageCollection(dates.map(_daily))


def _gee_sample_collection_stations(
    ee,
    col,
    stations: pd.DataFrame,
    bands: list,
    scale: int,
    sensor_name: str,
    batch_label: str,
) -> list:
    """
    Sample every image in a GEE ImageCollection at station points via
    .sampleRegions(), batching stations to stay under GEE's 5000-element
    getInfo() limit.

    For sensors with many overlapping tiles per day (Sentinel-2), the
    collection is first mosaicked to daily composites so that image count
    stays at most 31 per month.  Effective batch size is then computed as
    min(_GEE_STATION_BATCH, floor(4500 / n_images)).
    """
    # Mosaic to daily composites when collection is large (e.g. S2 multi-tile)
    try:
        n_images = col.size().getInfo()
    except Exception:
        n_images = 31
    logger.info(f"  {sensor_name} {batch_label}: {n_images} images before mosaic")
    if n_images > _MOSAIC_THRESHOLD:
        col = _mosaic_by_date(ee, col)
        try:
            n_images = col.size().getInfo()
        except Exception:
            n_images = 31
        logger.info(f"  {sensor_name} {batch_label}: {n_images} daily composites after mosaic")

    safe_batch = max(1, 4500 // max(n_images, 1))
    effective_batch = min(_GEE_STATION_BATCH, safe_batch)

    all_rows = []
    n = len(stations)
    n_batches = (n + effective_batch - 1) // effective_batch

    for b_idx, batch_start in enumerate(range(0, n, effective_batch)):
        batch_df = stations.iloc[batch_start : batch_start + effective_batch]
        batch_fc = _stations_to_fc(ee, batch_df)

        def _sample_img(img, _bfc=batch_fc):
            img_date = _gee_image_date_str(ee, img)
            samples = img.select(bands).sampleRegions(
                collection=_bfc,
                scale=scale,
                tileScale=4,
                geometries=False,
            )
            return samples.map(
                lambda f: f.set("img_date", img_date)
            )

        features = []
        for attempt in range(4):
            try:
                features = col.map(_sample_img).flatten().getInfo().get("features", [])
                break
            except Exception as exc:
                if attempt == 3:
                    logger.warning(
                        f"  {sensor_name} {batch_label} batch {b_idx + 1}/{n_batches}: "
                        f"sampleRegions failed — {exc}"
                    )
                else:
                    time.sleep(2 ** attempt * 5)

        for feat in features:
            p = feat["properties"]
            station_id = p.get("station_id")
            if station_id is None:
                continue
            row = {"date": p.get("img_date"), "station_id": station_id, "sensor": sensor_name}
            for b in bands:
                row[b] = float(p[b]) if p.get(b) is not None else np.nan
            all_rows.append(row)

    return all_rows


def _fallback_lakes() -> list:
    """Fallback to station-clustered lakes when HydroLAKES is unavailable."""
    try:
        stations = load_stations()
        return _generate_lake_polygons_from_stations(stations)
    except Exception as exc:
        logger.warning(f"Station clustering fallback failed: {exc}")
        return []


# ── MODIS extraction (GEE) ────────────────────────────────────────────────────

# MODIS MOD09GA / MYD09GA surface reflectance bands used:
#   sur_refl_b01: 620–670 nm  (Red)
#   sur_refl_b02: 841–876 nm  (NIR)
#   sur_refl_b03: 459–479 nm  (Blue)
#   sur_refl_b04: 545–565 nm  (Green)
# Scale factor: 0.0001 (DN → reflectance)

_MODIS_SCALE = 0.0001
_MODIS_BANDS = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04"]
_MODIS_BAND_LABELS = {
    "sur_refl_b01": "rw_b01_red",
    "sur_refl_b02": "rw_b02_nir",
    "sur_refl_b03": "rw_b03_blue",
    "sur_refl_b04": "rw_b04_green",
}


def extract_modis_year(
    year: int, mode: str = "stats", stations: Optional[pd.DataFrame] = None,
    lake_polygons: Optional[list] = None,
) -> pd.DataFrame:
    """Extract MODIS Terra + Aqua for one year via GEE.

    mode='stats'    — lake-mean reflectance per date (one row per lake per image)
    mode='stations' — pixel value at each WQ station per date (one row per station per image)
    """
    try:
        ee = _gee_init()
    except Exception as exc:
        logger.error(f"GEE init failed: {exc}")
        return pd.DataFrame()

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    water_mask = jrc.gt(80)

    if mode == "stations":
        if stations is None or stations.empty:
            logger.error("mode=stations requires a non-empty stations DataFrame")
            return pd.DataFrame()
        all_rows = []
        w, s, e, n = SWEDEN_UNION_BBOX
        union_geom = ee.Geometry.Rectangle([w, s, e, n])
        for collection_id, sensor_name in [
            ("MODIS/061/MOD09GA", "modis_terra"),
            ("MODIS/061/MYD09GA", "modis_aqua"),
        ]:
            if sensor_name == "modis_aqua" and year < 2002:
                continue
            logger.info(f"  GEE {sensor_name} stations {year}")
            for month in range(1, 13):
                date_start = f"{year}-{month:02d}-01"
                date_end = (
                    f"{year}-12-31" if month == 12
                    else str(datetime.date(year, month + 1, 1) - datetime.timedelta(days=1))
                )
                col = (
                    ee.ImageCollection(collection_id)
                    .filterDate(date_start, date_end)
                    .filterBounds(union_geom)
                    .select(_MODIS_BANDS)
                    .map(lambda img: img.multiply(_MODIS_SCALE).updateMask(water_mask))
                )
                rows = _gee_sample_collection_stations(
                    ee, col, stations, _MODIS_BANDS, 500,
                    sensor_name, f"{year}-{month:02d}"
                )
                for row in rows:
                    rw_blue  = row.pop("sur_refl_b03", np.nan)
                    rw_green = row.pop("sur_refl_b04", np.nan)
                    rw_red   = row.pop("sur_refl_b01", np.nan)
                    rw_nir   = row.pop("sur_refl_b02", np.nan)
                    row["rw_b01_red"]   = rw_red
                    row["rw_b02_nir"]   = rw_nir
                    row["rw_b03_blue"]  = rw_blue
                    row["rw_b04_green"] = rw_green
                    if not any(np.isnan(v) for v in [rw_blue, rw_green, rw_red, rw_nir]):
                        row.update(derive_wq_from_bands(rw_blue, rw_green, rw_red, rw_nir))
                all_rows.extend(rows)
                time.sleep(0.2)
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df["ac_method"] = "modis_ac"   # MD-1
        return df.sort_values(["station_id", "sensor", "date"]).reset_index(drop=True)

    # ── stats mode ────────────────────────────────────────────────────────────
    if lake_polygons is None:
        lake_polygons = _fallback_lakes()
    all_rows = []

    for lake in lake_polygons:
        lake_id = lake["lake_id"]
        w, s, e, n = lake["bbox"]
        geometry = lake.get("geometry") or ee.Geometry.Rectangle([w, s, e, n])

        for collection_id, sensor_name in [
            ("MODIS/061/MOD09GA", "modis_terra"),
            ("MODIS/061/MYD09GA", "modis_aqua"),
        ]:
            # MYD09GA (Aqua) starts 2002-07-04
            if sensor_name == "modis_aqua" and year < 2002:
                continue

            logger.info(f"  GEE {sensor_name} {lake_id} {year}")

            for month in range(1, 13):
                date_start = f"{year}-{month:02d}-01"
                if month == 12:
                    date_end = f"{year}-12-31"
                else:
                    date_end = str(
                        datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
                    )

                col = (
                    ee.ImageCollection(collection_id)
                    .filterDate(date_start, date_end)
                    .filterBounds(geometry)
                    .select(_MODIS_BANDS)
                )

                def _make_modis_feature(img):
                    scaled = img.multiply(_MODIS_SCALE).updateMask(water_mask)
                    stats = scaled.reduceRegion(
                        reducer=ee.Reducer.mean().combine(
                            ee.Reducer.count(), sharedInputs=True
                        ),
                        geometry=geometry,
                        scale=500,
                        maxPixels=1_000_000_000,
                    )
                    return ee.Feature(
                        None, stats.set("date", _gee_image_date_str(ee, img))
                    )

                try:
                    fc_info = col.map(_make_modis_feature).getInfo()

                    for feat in fc_info.get("features", []):
                        p = feat["properties"]
                        rw_red   = p.get("sur_refl_b01_mean") or np.nan
                        rw_nir   = p.get("sur_refl_b02_mean") or np.nan
                        rw_blue  = p.get("sur_refl_b03_mean") or np.nan
                        rw_green = p.get("sur_refl_b04_mean") or np.nan
                        n_px     = int(p.get("sur_refl_b01_count") or 0)

                        row: dict = {
                            "date":            p.get("date"),
                            "lake_id":         lake_id,
                            "sensor":          sensor_name,
                            "n_valid_pixels":  n_px,
                            "cloud_frac":      np.nan,
                            "rw_b01_red":      rw_red,
                            "rw_b02_nir":      rw_nir,
                            "rw_b03_blue":     rw_blue,
                            "rw_b04_green":    rw_green,
                        }

                        if n_px > 0 and not any(
                            np.isnan(v) for v in [rw_red, rw_nir, rw_blue, rw_green]
                        ):
                            row.update(
                                derive_wq_from_bands(rw_blue, rw_green, rw_red, rw_nir)
                            )

                        all_rows.append(row)

                except Exception as exc:
                    logger.warning(
                        f"  GEE {sensor_name} {lake_id} {year}-{month:02d}: {exc}"
                    )

                time.sleep(0.3)  # avoid hammering GEE

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["ac_method"] = "modis_ac"   # MD-1
    return df.sort_values(["lake_id", "sensor", "date"]).reset_index(drop=True)


# ── Landsat extraction (GEE) ──────────────────────────────────────────────────

# Landsat Collection 2 Level-2 Surface Reflectance
# Scale: DN * 0.0000275 + (-0.2)
# QA_PIXEL cloud bits: 3=Cloud, 4=CloudShadow, 5=Snow/Ice

_LS_SCALE  = 0.0000275
_LS_OFFSET = -0.2

LANDSAT_COLLECTIONS = {
    "landsat7": {
        "id":    "LANDSAT/LE07/C02/T1_L2",
        "bands": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4"},
        "years": (1999, 2023),
    },
    "landsat8": {
        "id":    "LANDSAT/LC08/C02/T1_L2",
        "bands": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5"},
        "years": (2013, 2025),
    },
    "landsat9": {
        "id":    "LANDSAT/LC09/C02/T1_L2",
        "bands": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5"},
        "years": (2021, 2025),
    },
}


def extract_landsat_year(
    year: int, mode: str = "stats", stations: Optional[pd.DataFrame] = None,
    lake_polygons: Optional[list] = None,
) -> pd.DataFrame:
    """Extract Landsat 7/8/9 for one year via GEE.

    mode='stats'    — lake-mean per date
    mode='stations' — pixel at each WQ station per date
    """
    try:
        ee = _gee_init()
    except Exception as exc:
        logger.error(f"GEE init failed: {exc}")
        return pd.DataFrame()

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    water_mask = jrc.gt(80)

    def _cloud_mask_ls(img):
        qa = img.select("QA_PIXEL")
        bad = (
            qa.bitwiseAnd(1 << 3).neq(0)
            .Or(qa.bitwiseAnd(1 << 4).neq(0))
            .Or(qa.bitwiseAnd(1 << 5).neq(0))
        )
        return img.updateMask(bad.Not())

    if mode == "stations":
        if stations is None or stations.empty:
            logger.error("mode=stations requires a non-empty stations DataFrame")
            return pd.DataFrame()
        w, s, e, n = SWEDEN_UNION_BBOX
        union_geom = ee.Geometry.Rectangle([w, s, e, n])
        all_rows = []
        for ls_name, ls_cfg in LANDSAT_COLLECTIONS.items():
            y0, y1 = ls_cfg["years"]
            if not (y0 <= year <= y1):
                continue
            bands    = ls_cfg["bands"]
            sr_bands = list(bands.values())
            logger.info(f"  GEE {ls_name} stations {year}")
            # Process month-by-month (same pattern as MODIS) to keep each
            # collection small — full-year over Sweden's large bbox causes
            # col.size().getInfo() to hang on hundreds of tiles.
            for month in range(1, 13):
                date_start = f"{year}-{month:02d}-01"
                date_end = (
                    f"{year}-12-31" if month == 12
                    else str(datetime.date(year, month + 1, 1) - datetime.timedelta(days=1))
                )
                col = (
                    ee.ImageCollection(ls_cfg["id"])
                    .filterDate(date_start, date_end)
                    .filterBounds(union_geom)
                    .select(sr_bands + ["QA_PIXEL"])
                    .map(_cloud_mask_ls)
                    .map(lambda img: img.select(sr_bands).multiply(_LS_SCALE).add(_LS_OFFSET)
                         .updateMask(water_mask)
                         .copyProperties(img, ["system:time_start", "system:index"]))
                )
                rows = _gee_sample_collection_stations(
                    ee, col, stations, sr_bands, 30,
                    ls_name, f"{year}-{month:02d}"
                )
                for row in rows:
                    rw_blue  = row.pop(bands["blue"],  np.nan)
                    rw_green = row.pop(bands["green"], np.nan)
                    rw_red   = row.pop(bands["red"],   np.nan)
                    rw_nir   = row.pop(bands["nir"],   np.nan)
                    row.update({"rw_blue": rw_blue, "rw_green": rw_green,
                                "rw_red": rw_red,  "rw_nir": rw_nir})
                    if not any(np.isnan(v) for v in [rw_blue, rw_green, rw_red, rw_nir]):
                        row.update(derive_wq_from_bands(rw_blue, rw_green, rw_red, rw_nir))
                all_rows.extend(rows)
                time.sleep(0.2)
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df["ac_method"] = "lasrc"   # MD-1
        return df.sort_values(["station_id", "sensor", "date"]).reset_index(drop=True)

    # ── stats mode ────────────────────────────────────────────────────────────
    if lake_polygons is None:
        lake_polygons = _fallback_lakes()
    all_rows = []

    for ls_name, ls_cfg in LANDSAT_COLLECTIONS.items():
        y0, y1 = ls_cfg["years"]
        if not (y0 <= year <= y1):
            continue

        bands = ls_cfg["bands"]
        sr_bands = list(bands.values())
        qa_bands = sr_bands + ["QA_PIXEL"]

        for lake in lake_polygons:
            lake_id  = lake["lake_id"]
            w, s, e, n = lake["bbox"]
            geometry = lake.get("geometry") or ee.Geometry.Rectangle([w, s, e, n])

            logger.info(f"  GEE {ls_name} {lake_id} {year}")

            col = (
                ee.ImageCollection(ls_cfg["id"])
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .filterBounds(geometry)
                .select(qa_bands)
                .map(_cloud_mask_ls)
            )

            def _make_ls_feature(img):
                scaled = (
                    img.select(sr_bands)
                    .multiply(_LS_SCALE)
                    .add(_LS_OFFSET)
                    .updateMask(water_mask)
                )
                stats = scaled.reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        ee.Reducer.count(), sharedInputs=True
                    ),
                    geometry=geometry,
                    scale=30,
                    maxPixels=1_000_000_000,
                )
                return ee.Feature(
                    None, stats.set("date", _gee_image_date_str(ee, img))
                )

            try:
                fc_info = col.map(_make_ls_feature).getInfo()

                for feat in fc_info.get("features", []):
                    p = feat["properties"]
                    b = bands
                    rw_blue  = p.get(f"{b['blue']}_mean") or np.nan
                    rw_green = p.get(f"{b['green']}_mean") or np.nan
                    rw_red   = p.get(f"{b['red']}_mean") or np.nan
                    rw_nir   = p.get(f"{b['nir']}_mean") or np.nan
                    n_px     = int(p.get(f"{b['blue']}_count") or 0)

                    row: dict = {
                        "date":           p.get("date"),
                        "lake_id":        lake_id,
                        "sensor":         ls_name,
                        "n_valid_pixels": n_px,
                        "cloud_frac":     np.nan,
                        "rw_blue":        rw_blue,
                        "rw_green":       rw_green,
                        "rw_red":         rw_red,
                        "rw_nir":         rw_nir,
                    }

                    if n_px > 0 and not any(
                        np.isnan(v) for v in [rw_blue, rw_green, rw_red, rw_nir]
                    ):
                        row.update(
                            derive_wq_from_bands(rw_blue, rw_green, rw_red, rw_nir)
                        )

                    all_rows.append(row)

            except Exception as exc:
                logger.warning(f"  GEE {ls_name} {lake_id} {year}: {exc}")

            time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["ac_method"] = "lasrc"   # MD-1
    return df.sort_values(["lake_id", "sensor", "date"]).reset_index(drop=True)


# ── Sentinel-1 SAR GRD (GEE) ─────────────────────────────────────────────────

# COPERNICUS/S1_GRD: C-band SAR, IW mode, VV + VH polarization
# Values stored as sigma0 in dB; convert to linear for averaging and ratios
# Available from 2014-04-03 (Sentinel-1A)
#
# WQ-relevant products:
#   sigma0_vv_db   — mean VV backscatter (dB); wind/wave roughness proxy
#   sigma0_vh_db   — mean VH backscatter (dB); volume scattering proxy
#   cr_db          — cross-ratio VV−VH (dB); turbidity proxy
#   vv_std_db      — spatial std of VV within lake (texture / roughness variability)
#   ice_flag       — 1 when likely ice: VV > −10 dB AND VV_std < 2 dB
#   n_valid_pixels — water pixels within lake bbox after JRC mask


def extract_sentinel1_year(
    year: int, mode: str = "stats", stations: Optional[pd.DataFrame] = None,
    lake_polygons: Optional[list] = None,
) -> pd.DataFrame:
    """Extract Sentinel-1 IW GRD for one year via GEE.

    SAR is cloud-penetrating — no cloud masking needed.
    mode='stats'    — lake-mean VV/VH + std + ice_flag per date
    mode='stations' — VV/VH pixel value at each WQ station per date
    """
    if year < 2014:
        logger.warning(f"  S1 GRD: Sentinel-1A launched 2014-04-03, skipping {year}")
        return pd.DataFrame()

    try:
        ee = _gee_init()
    except Exception as exc:
        logger.error(f"GEE init failed: {exc}")
        return pd.DataFrame()

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    water_mask = jrc.gt(80)

    def _s1_col(geom):
        return (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(geom)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
            .map(lambda img: img.updateMask(water_mask))
        )

    if mode == "stations":
        if stations is None or stations.empty:
            logger.error("mode=stations requires a non-empty stations DataFrame")
            return pd.DataFrame()
        w, s, e, n = SWEDEN_UNION_BBOX
        union_geom = ee.Geometry.Rectangle([w, s, e, n])
        all_rows = []
        logger.info(f"  GEE sentinel1 stations {year}")
        start_month = 4 if year == 2014 else 1
        for month in range(start_month, 13):
            date_start = f"{year}-{month:02d}-01"
            date_end = (
                f"{year}-12-31" if month == 12
                else str(datetime.date(year, month + 1, 1) - datetime.timedelta(days=1))
            )
            col = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterDate(date_start, date_end)
                .filterBounds(union_geom)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .select(["VV", "VH"])
                .map(lambda img: img.updateMask(water_mask))
            )
            rows = _gee_sample_collection_stations(
                ee, col, stations, ["VV", "VH"], 10,
                "sentinel1", f"{year}-{month:02d}"
            )
            for row in rows:
                vv = row.get("VV", np.nan)
                vh = row.get("VH", np.nan)
                row["sigma0_vv_db"] = vv
                row["sigma0_vh_db"] = vh
                row["cr_db"]     = float(vv - vh) if not any(np.isnan([vv, vh])) else np.nan
                row["vv_std_db"] = np.nan
                row["ice_flag"]  = int(vv > -10.0) if not np.isnan(vv) else 0
            all_rows.extend(rows)
            time.sleep(0.2)
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df["ac_method"] = "sar"   # MD-1: SAR — no atmospheric correction applied
        return df.sort_values(["station_id", "date"]).reset_index(drop=True)

    # ── stats mode ────────────────────────────────────────────────────────────
    if lake_polygons is None:
        lake_polygons = _fallback_lakes()
    all_rows = []

    for lake in lake_polygons:
        lake_id = lake["lake_id"]
        w, s, e, n = lake["bbox"]
        geometry = lake.get("geometry") or ee.Geometry.Rectangle([w, s, e, n])

        logger.info(f"  GEE sentinel1 {lake_id} {year}")

        col = _s1_col(geometry)

        def _make_s1_feature(img):
            vv_db = img.select("VV")
            vh_db = img.select("VH")

            # Mean + std in dB (log scale — valid for geophysical interpretation)
            stats_vv = vv_db.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.stdDev().combine(
                        ee.Reducer.count(), sharedInputs=True
                    ),
                    sharedInputs=True,
                ),
                geometry=geometry,
                scale=10,
                maxPixels=1_000_000_000,
            )
            stats_vh = vh_db.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10,
                maxPixels=1_000_000_000,
            )

            return ee.Feature(
                None,
                stats_vv.combine(stats_vh).set(
                    "date", _gee_image_date_str(ee, img)
                ),
            )

        try:
            fc_info = col.map(_make_s1_feature).getInfo()

            for feat in fc_info.get("features", []):
                p = feat["properties"]
                vv_mean = p.get("VV_mean")
                vv_std  = p.get("VV_stdDev")
                vh_mean = p.get("VH_mean")
                n_px    = int(p.get("VV_count") or 0)

                if vv_mean is None or vh_mean is None:
                    continue

                vv_mean = float(vv_mean)
                vh_mean = float(vh_mean)
                vv_std  = float(vv_std) if vv_std is not None else np.nan

                # Cross-ratio (VV − VH in dB)
                cr_db = float(vv_mean - vh_mean)

                # Ice flag: high VV backscatter + low spatial std → smooth frozen surface
                ice_flag = int(vv_mean > -10.0 and (np.isnan(vv_std) or vv_std < 2.0))

                all_rows.append({
                    "date":            p.get("date"),
                    "lake_id":         lake_id,
                    "sensor":          "sentinel1",
                    "n_valid_pixels":  n_px,
                    "sigma0_vv_db":    vv_mean,
                    "sigma0_vh_db":    vh_mean,
                    "vv_std_db":       vv_std,
                    "cr_db":           cr_db,
                    "ice_flag":        ice_flag,
                })

        except Exception as exc:
            logger.warning(f"  GEE sentinel1 {lake_id} {year}: {exc}")

        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["ac_method"] = "sar"   # MD-1: SAR — no atmospheric correction applied
    return df.sort_values(["lake_id", "date"]).reset_index(drop=True)


# ── Sentinel-2 MSI SR (GEE) ──────────────────────────────────────────────────

# COPERNICUS/S2_SR_HARMONIZED: atmospherically corrected surface reflectance
# Scale: DN / 10000 → reflectance
# Cloud masking: SCL (Scene Classification Layer)
#   Exclude: 0=no data, 1=saturated/defective, 3=cloud shadow,
#            7=unclassified (excluded for reproducibility),
#            8=medium cloud, 9=high cloud, 10=thin cirrus, 11=snow/ice
#   Water-only refinement (L2A): additionally require SCL==6 (water) intersected
#   with JRC permanent water mask to restrict stats to agreed-water pixels.

_S2_SCALE = 1.0 / 10000.0

# Bands selected for WQ extraction (all at 20 m or resampled to 20 m by GEE):
#   B2  490nm  blue      — CDOM / colour / OC4
#   B3  560nm  green     — turbidity / OC4
#   B4  665nm  red       — Chl absorption / NDCI denominator
#   B5  705nm  red-edge  — NDCI numerator / Chl fluorescence peak
#   B6  740nm  red-edge  — TSS / Secchi predictor
#   B7  783nm  red-edge  — completes red-edge triplet; Chl/Secchi/TSS ML models
#   B8  842nm  NIR broad — turbidity/TSS, NDWI/AWEI variants
#   B8A 865nm  NIR narrow— AC reference; atmospheric correction proxy
#   B11 1610nm SWIR-1    — TSS/turbidity index; separates high-TSS from shadow
#   B12 2190nm SWIR-2    — AWEI_sh water mask stability; TSS/SPM models
_S2_SR_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
_S2_BANDS    = _S2_SR_BANDS + ["SCL"]   # L2A: include SCL for cloud/water masking


def extract_sentinel2_year(
    year: int, mode: str = "stats", stations: Optional[pd.DataFrame] = None,
    lake_polygons: Optional[list] = None,
) -> pd.DataFrame:
    """Extract Sentinel-2 MSI for one year via GEE.

    Collection: S2_HARMONIZED (L1C, 2015-2016) or S2_SR_HARMONIZED (L2A, 2017+).
    mode='stats'    — lake-mean per date
    mode='stations' — pixel at each WQ station per date
    """
    if year < 2015:
        logger.warning(f"  S2: Sentinel-2A launched 2015-06-23, skipping {year}")
        return pd.DataFrame()

    try:
        ee = _gee_init()
    except Exception as exc:
        logger.error(f"GEE init failed: {exc}")
        return pd.DataFrame()

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    water_mask = jrc.gt(80)

    # L2A cloud mask via SCL — also excludes no-data, saturated, and unclassified
    def _cloud_mask_l2a(img):
        scl = img.select("SCL")
        bad = (
            scl.eq(0)        # no data
            .Or(scl.eq(1))   # saturated / defective
            .Or(scl.eq(3))   # cloud shadow
            .Or(scl.eq(7))   # unclassified (masked for reproducibility)
            .Or(scl.eq(8))   # medium probability cloud
            .Or(scl.eq(9))   # high probability cloud
            .Or(scl.eq(10))  # thin cirrus
            .Or(scl.eq(11))  # snow / ice
        )
        return img.updateMask(bad.Not())

    # L1C cloud mask via QA60 (no SCL available)
    def _cloud_mask_l1c(img):
        qa = img.select("QA60")
        bad = (
            qa.bitwiseAnd(1 << 10).neq(0)  # opaque clouds
            .Or(qa.bitwiseAnd(1 << 11).neq(0))  # cirrus
        )
        return img.updateMask(bad.Not())

    # Collection choice: L2A from 2017-03-28; L1C for 2015-2016
    use_l2a = year >= 2017
    if use_l2a:
        collection_id = "COPERNICUS/S2_SR_HARMONIZED"
        cloud_mask_fn = _cloud_mask_l2a
        select_bands  = _S2_BANDS
        product_level = "L2A"
    else:
        collection_id = "COPERNICUS/S2_HARMONIZED"
        cloud_mask_fn = _cloud_mask_l1c
        select_bands  = _S2_SR_BANDS + ["QA60"]
        product_level = "L1C"

    if mode == "stations":
        if stations is None or stations.empty:
            logger.error("mode=stations requires a non-empty stations DataFrame")
            return pd.DataFrame()
        w, s, e, n = SWEDEN_UNION_BBOX
        union_geom = ee.Geometry.Rectangle([w, s, e, n])
        logger.info(f"  GEE sentinel2 ({product_level}) stations {year}")
        all_rows = []
        for month in range(1, 13):
            date_start = f"{year}-{month:02d}-01"
            date_end = (
                f"{year}-12-31" if month == 12
                else str(datetime.date(year, month + 1, 1) - datetime.timedelta(days=1))
            )
            col = (
                ee.ImageCollection(collection_id)
                .filterDate(date_start, date_end)
                .filterBounds(union_geom)
                .select(select_bands)
                .map(cloud_mask_fn)
                 .map(lambda img: img.select(_S2_SR_BANDS).multiply(_S2_SCALE)
                     .updateMask(jrc.gt(80))
                     .copyProperties(img, ["system:time_start", "system:index"]))
            )
            rows = _gee_sample_collection_stations(
                ee, col, stations, _S2_SR_BANDS, 20, "sentinel2", f"{year}-{month:02d}"
            )
            for row in rows:
                row["product_level"] = product_level
                rw_blue  = row.get("B2",  np.nan)
                rw_green = row.get("B3",  np.nan)
                rw_red   = row.get("B4",  np.nan)
                rw_re1   = row.get("B5",  np.nan)
                rw_nir   = row.get("B8A", np.nan)
                for b, col_name in zip(_S2_SR_BANDS,
                    ["rw_b2_490","rw_b3_560","rw_b4_665","rw_b5_705",
                     "rw_b6_740","rw_b7_783","rw_b8_842","rw_b8a_865",
                     "rw_b11_1610","rw_b12_2190"]):
                    row[col_name] = row.pop(b, np.nan)
                ndci = np.nan
                if not any(np.isnan([rw_re1, rw_red])):
                    denom = rw_re1 + rw_red
                    ndci = float((rw_re1 - rw_red) / max(denom, 1e-9))
                row["ndci"] = ndci
                if not any(np.isnan([rw_blue, rw_green, rw_red, rw_nir])):
                    wq = derive_wq_from_bands(rw_blue, rw_green, rw_red, rw_nir)
                    row["chl_a_oc4"] = wq.get("chl_a_ug_l", np.nan)
                    row.update({k: v for k, v in wq.items() if k != "chl_a_ug_l"})
                    if not np.isnan(ndci):
                        chl_ndci = 14.039 + 86.115 * ndci + 194.325 * ndci ** 2
                        row["chl_a_ndci"] = float(np.clip(chl_ndci, 0.01, 500.0))
                        row["chl_a_ug_l"] = row["chl_a_ndci"] if ndci > -0.05 else row["chl_a_oc4"]
                    else:
                        row["chl_a_ndci"] = np.nan
                        row["chl_a_ug_l"] = row["chl_a_oc4"]
            all_rows.extend(rows)
            time.sleep(0.2)
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df["ac_method"] = df["product_level"].map({"L2A": "sen2cor", "L1C": "toa"})  # MD-1
        return df.sort_values(["station_id", "date"]).reset_index(drop=True)

    # ── stats mode ────────────────────────────────────────────────────────────
    if lake_polygons is None:
        lake_polygons = _fallback_lakes()
    all_rows = []

    for lake in lake_polygons:
        lake_id = lake["lake_id"]
        w, s, e, n = lake["bbox"]
        geometry = lake.get("geometry") or ee.Geometry.Rectangle([w, s, e, n])

        logger.info(f"  GEE sentinel2 ({product_level}) {lake_id} {year}")

        col = (
            ee.ImageCollection(collection_id)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(geometry)
            .select(select_bands)
            .map(cloud_mask_fn)
        )

        def _make_s2_feature(img):
            # For L2A: intersect JRC permanent water with SCL==6 (Sen2Cor water class)
            # so stats only come from pixels both classifiers agree are water.
            # For L1C: SCL not available — fall back to JRC only.
            if use_l2a:
                scl_water = img.select("SCL").eq(6)
                pixel_mask = water_mask.And(scl_water)
            else:
                pixel_mask = water_mask

            scaled = (
                img.select(_S2_SR_BANDS)
                .multiply(_S2_SCALE)
                .updateMask(pixel_mask)
            )
            stats = scaled.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.count(), sharedInputs=True
                ),
                geometry=geometry,
                scale=20,
                maxPixels=1_000_000_000,
            )
            return ee.Feature(
                None, stats.set("date", _gee_image_date_str(ee, img))
            )

        try:
            fc_info = col.map(_make_s2_feature).getInfo()

            for feat in fc_info.get("features", []):
                p = feat["properties"]
                rw_blue    = p.get("B2_mean")  or np.nan   # 490nm
                rw_green   = p.get("B3_mean")  or np.nan   # 560nm
                rw_red     = p.get("B4_mean")  or np.nan   # 665nm
                rw_re1     = p.get("B5_mean")  or np.nan   # 705nm red-edge
                rw_re2     = p.get("B6_mean")  or np.nan   # 740nm red-edge
                rw_re3     = p.get("B7_mean")  or np.nan   # 783nm red-edge
                rw_nir_b   = p.get("B8_mean")  or np.nan   # 842nm NIR broad
                rw_nir     = p.get("B8A_mean") or np.nan   # 865nm NIR narrow
                rw_swir1   = p.get("B11_mean") or np.nan   # 1610nm SWIR-1
                rw_swir2   = p.get("B12_mean") or np.nan   # 2190nm SWIR-2
                n_px       = int(p.get("B2_count") or 0)

                # NDCI = (B5_705nm − B4_665nm) / (B5 + B4)
                # Mishra & Mishra 2012, RSE — calibrated for estuarine Case-2 waters;
                # sensitive for Chl-a ≳ 5 µg/L (eutrophic/mesotrophic range).
                ndci = np.nan
                if not any(np.isnan([rw_re1, rw_red])):
                    denom = rw_re1 + rw_red
                    ndci = float((rw_re1 - rw_red) / max(denom, 1e-9))

                row: dict = {
                    "date":            p.get("date"),
                    "lake_id":         lake_id,
                    "sensor":          "sentinel2",
                    "product_level":   product_level,
                    "n_valid_pixels":  n_px,
                    "cloud_frac":      np.nan,
                    "rw_b2_490":       rw_blue,
                    "rw_b3_560":       rw_green,
                    "rw_b4_665":       rw_red,
                    "rw_b5_705":       rw_re1,
                    "rw_b6_740":       rw_re2,
                    "rw_b7_783":       rw_re3,
                    "rw_b8_842":       rw_nir_b,
                    "rw_b8a_865":      rw_nir,
                    "rw_b11_1610":     rw_swir1,
                    "rw_b12_2190":     rw_swir2,
                    "ndci":            ndci,
                    "chl_a_oc4":       np.nan,
                    "chl_a_ndci":      np.nan,
                    "chl_a_ug_l":      np.nan,
                }

                if n_px > 0 and not any(
                    np.isnan(v) for v in [rw_blue, rw_green, rw_red, rw_nir]
                ):
                    wq = derive_wq_from_bands(rw_blue, rw_green, rw_red, rw_nir)
                    row.update(wq)
                    # Store OC4 under its own key before potentially overwriting chl_a_ug_l
                    row["chl_a_oc4"] = wq.get("chl_a_ug_l", np.nan)

                    # NDCI-based Chl-a (Mishra & Mishra 2012):
                    #   chl_ndci = 14.039 + 86.115×NDCI + 194.325×NDCI²
                    # Use NDCI when NDCI > −0.05 (mesotrophic/eutrophic range where
                    # it outperforms OC4 for turbid Case-2 waters). Fall back to OC4
                    # for oligotrophic/clear water (NDCI ≤ −0.05) where OC4 is more
                    # reliable and NDCI is near its noise floor.
                    if not np.isnan(ndci):
                        chl_ndci = 14.039 + 86.115 * ndci + 194.325 * ndci ** 2
                        row["chl_a_ndci"] = float(np.clip(chl_ndci, 0.01, 500.0))
                        if ndci > -0.05:
                            row["chl_a_ug_l"] = row["chl_a_ndci"]
                        # else: chl_a_ug_l stays as OC4 from wq dict

                all_rows.append(row)

        except Exception as exc:
            logger.warning(f"  GEE sentinel2 {lake_id} {year}: {exc}")

        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["ac_method"] = df["product_level"].map({"L2A": "sen2cor", "L1C": "toa"})  # MD-1
    return df.sort_values(["lake_id", "date"]).reset_index(drop=True)


# ── Sentinel-3 OLCI L2 WFR (CDSE) ────────────────────────────────────────────

CDSE_TOKEN_URL    = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_ODATA_URL    = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

# WQSF flag bits (WQSF_lsb 32-bit word)
_WQSF_INVALID      = 1 << 0
_WQSF_WATER        = 1 << 1
_WQSF_CLOUD        = 1 << 2
_WQSF_CLOUD_AMB    = 1 << 3
_WQSF_SNOW_ICE     = 1 << 5
_WQSF_INLAND_WATER = 1 << 6
_WQSF_AC_FAIL      = 1 << 18

# OLCI reflectance bands to extract (wavelength in nm)
OLCI_BANDS = {
    "Oa03": 443,   # CDOM absorption
    "Oa04": 490,   # Chl-a blue
    "Oa06": 560,   # Chl-a green peak
    "Oa08": 665,   # Chl-a absorption
    "Oa10": 681,   # Chl-a fluorescence
    "Oa11": 709,   # Chl-a red edge (NDCI)
    "Oa12": 754,   # NIR baseline
    "Oa17": 865,   # AC reference / turbidity NIR
}


def _get_cdse_token() -> str:
    resp = requests.post(
        CDSE_TOKEN_URL,
        data={
            "grant_type": "password",
            "username":   os.getenv("CDSE_USER", ""),
            "password":   os.getenv("CDSE_PASSWORD", ""),
            "client_id":  "cdse-public",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _search_olci_wfr(bbox, year: int) -> list:
    """
    Search CDSE OData for S3 OLCI WFR products over bbox for a full year.
    Filters by cloud cover < max_cloud_pct to skip heavily cloudy scenes.
    """
    w, s, e, n = bbox
    footprint = (
        f"geography'SRID=4326;"
        f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))'"
    )

    # Note: CDSE S3 OLCI WFR products carry no 'cloudCover' OData attribute —
    # any() on a missing attribute returns false, silently killing the search.
    # Cloud/invalid pixels are already masked by the WQSF bitmask in _extract_olci_lake.
    base_filter = (
        "Collection/Name eq 'SENTINEL-3' and "
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' and "
        "att/OData.CSC.StringAttribute/Value eq 'OL_2_WFR___') and "
        f"ContentDate/Start ge {year}-01-01T00:00:00.000Z and "
        f"ContentDate/Start le {year}-12-31T23:59:59.000Z and "
        f"OData.CSC.Intersects(area={footprint})"
    )

    products = []
    skip = 0

    while True:
        try:
            resp = requests.get(
                CDSE_ODATA_URL,
                params={"$filter": base_filter, "$top": 100, "$skip": skip,
                        "$orderby": "ContentDate/Start asc"},
                timeout=60,
            )
            resp.raise_for_status()
            batch = resp.json().get("value", [])
        except Exception as exc:
            logger.warning(f"  CDSE search failed (skip={skip}): {exc}")
            break

        products.extend(batch)

        if len(batch) < 100:
            break
        skip += 100

    return products


def _extract_olci_stations_batch(
    zf: zipfile.ZipFile,
    stations: pd.DataFrame,
    max_dist_km: float = 1.5,
) -> list:
    """
    Vectorised per-station extraction from one OLCI L2 WFR granule.

    geo_coordinates.nc is full-resolution (same shape as all science files).
    KD-tree is built on a step-16 subsampled grid for speed (~77k points vs
    20M), then a ±(STEP+2) window search on the full-res flag array finds the
    nearest valid water pixel.

    Confirmed file names from OLCI L2 WFR (processing baseline 003):
      flags  : wqsf.nc        (NOT WQSF_lsb.nc)
      Kd(490): trsp.nc        variable KD490_M07  (NOT kd490_m07.nc)
    """
    import xarray as xr
    from scipy.spatial import cKDTree

    names     = zf.namelist()
    prod_dirs = sorted({nm.split("/")[0] for nm in names if "/" in nm})
    if not prod_dirs:
        return []
    prod = prod_dirs[0]

    def _read_nc(nc_name: str):
        full = f"{prod}/{nc_name}"
        if full not in names:
            return None
        try:
            return xr.open_dataset(BytesIO(zf.read(full)), engine="h5netcdf")
        except Exception:
            return None

    # ── 1. Full-resolution coordinates ───────────────────────────────────────
    geo = _read_nc("geo_coordinates.nc")
    if geo is None:
        return []
    olci_lat = geo["latitude"].values.astype(np.float32)
    olci_lon = geo["longitude"].values.astype(np.float32)
    geo.close()
    fr_rows, fr_cols = olci_lat.shape   # e.g. (4091, 4865)

    # ── 2. Quality flags ─────────────────────────────────────────────────────
    flags_ds = _read_nc("wqsf.nc")
    if flags_ds is None:
        return []
    flags_raw = flags_ds[list(flags_ds.data_vars)[0]].values.astype(np.int64)
    flags_ds.close()
    if flags_raw.ndim == 3:
        flags_raw = flags_raw[0]

    # ── 3. Bbox pre-filter on stations ───────────────────────────────────────
    lat_min, lat_max = float(olci_lat.min()), float(olci_lat.max())
    lon_min, lon_max = float(olci_lon.min()), float(olci_lon.max())
    st_lat = stations["latitude"].values.astype(float)
    st_lon = stations["longitude"].values.astype(float)
    in_box  = (
        (st_lat >= lat_min - 0.05) & (st_lat <= lat_max + 0.05) &
        (st_lon >= lon_min - 0.05) & (st_lon <= lon_max + 0.05)
    )
    cand_mask = np.where(in_box)[0]
    if len(cand_mask) == 0:
        return []

    # ── 4. KD-tree on step-16 subsampled grid (speed) ────────────────────────
    STEP      = 16
    sub_lat   = olci_lat[::STEP, ::STEP]
    sub_lon   = olci_lon[::STEP, ::STEP]
    sub_shape = sub_lat.shape

    tree = cKDTree(np.column_stack([sub_lat.ravel(), sub_lon.ravel()]))
    dists, sub_flat = tree.query(
        np.column_stack([st_lat[cand_mask], st_lon[cand_mask]]), k=1
    )
    max_dist_deg = max_dist_km / 111.0
    close     = dists < max_dist_deg
    cand_mask = cand_mask[close]
    sub_flat  = sub_flat[close]
    if len(cand_mask) == 0:
        return []

    # Approximate full-res position from subsampled index
    sub_i, sub_j = np.unravel_index(sub_flat, sub_shape)
    fr_i0 = np.clip(sub_i * STEP, 0, fr_rows - 1)
    fr_j0 = np.clip(sub_j * STEP, 0, fr_cols - 1)

    # ── 5. Window search ±(STEP+2) px for nearest valid water pixel ──────────
    WIN   = STEP + 2   # covers full subsampling uncertainty
    best_r = np.full(len(cand_mask), -1, dtype=np.int32)
    best_c = np.full(len(cand_mask), -1, dtype=np.int32)

    for k in range(len(cand_mask)):
        r0, c0 = int(fr_i0[k]), int(fr_j0[k])
        r_lo = max(0, r0 - WIN);  r_hi = min(fr_rows, r0 + WIN + 1)
        c_lo = max(0, c0 - WIN);  c_hi = min(fr_cols, c0 + WIN + 1)

        win = flags_raw[r_lo:r_hi, c_lo:c_hi]
        wtr = (win & (_WQSF_WATER | _WQSF_INLAND_WATER)) != 0
        cld = (win & (_WQSF_CLOUD | _WQSF_CLOUD_AMB))    != 0
        bad = (win & (_WQSF_SNOW_ICE | _WQSF_AC_FAIL | _WQSF_INVALID)) != 0
        ok  = wtr & ~cld & ~bad

        if not ok.any():
            continue

        vi, vj = np.where(ok)
        cr, cc = r0 - r_lo, c0 - c_lo
        dist2  = (vi - cr) ** 2 + (vj - cc) ** 2
        bi     = dist2.argmin()
        best_r[k] = r_lo + vi[bi]
        best_c[k] = c_lo + vj[bi]

    found = best_r >= 0
    logger.info(
        f"  S3 batch: {len(cand_mask)} candidate stations → "
        f"{found.sum()} with valid water pixel"
    )
    cand_mask = cand_mask[found]
    i_idx     = best_r[found]
    j_idx     = best_c[found]
    if len(cand_mask) == 0:
        return []

    # ── 6. Extract NC variables at matched pixel positions ───────────────────
    extracted: dict = {}

    for col, nc_file, var_override in [
        ("chl_a_oc4me", "chl_oc4me.nc", None),
        ("chl_a_nn",    "chl_nn.nc",    None),
        ("tsm_nn",      "tsm_nn.nc",    None),
        ("kd490",       "trsp.nc",      "KD490_M07"),   # Kd490 lives in trsp.nc
    ]:
        ds = _read_nc(nc_file)
        if ds is not None:
            var = var_override if (var_override and var_override in ds.data_vars) \
                  else list(ds.data_vars)[0]
            arr = ds[var].values
            if arr.ndim == 3:
                arr = arr[0]
            extracted[col] = arr[i_idx, j_idx].astype(float)
            ds.close()
        else:
            extracted[col] = np.full(len(cand_mask), np.nan)

    for band_name, wl in OLCI_BANDS.items():
        ds = _read_nc(f"{band_name}_reflectance.nc")
        col = f"rw_{band_name.lower()}_{wl}nm"
        if ds is not None:
            arr = ds[list(ds.data_vars)[0]].values
            if arr.ndim == 3:
                arr = arr[0]
            extracted[col] = arr[i_idx, j_idx].astype(float)
            ds.close()
        else:
            extracted[col] = np.full(len(cand_mask), np.nan)

    # ── 6. Derived WQ products (vectorised) ───────────────────────────────────
    kd490 = extracted["kd490"]
    with np.errstate(divide="ignore", invalid="ignore"):
        extracted["secchi_m"] = np.where(
            (~np.isnan(kd490)) & (kd490 > 0), np.clip(1.7 / kd490, 0.1, 40.0), np.nan
        )

    tsm = extracted["tsm_nn"]
    extracted["turbidity_fnu"] = np.where(~np.isnan(tsm), tsm * 0.8, np.nan)

    chl_nn  = extracted["chl_a_nn"]
    chl_oc4 = extracted["chl_a_oc4me"]
    extracted["chl_a_ug_l"] = np.where(~np.isnan(chl_nn), chl_nn, chl_oc4)

    rw_blue  = extracted.get("rw_oa03_443nm", np.full(len(cand_mask), np.nan))
    rw_green = extracted.get("rw_oa06_560nm", np.full(len(cand_mask), np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        cdom_raw = -0.5 * np.log(
            np.maximum(rw_blue, 1e-9) / np.maximum(rw_green, 1e-9)
        )
    extracted["cdom_m"] = np.where(
        ~np.isnan(rw_blue) & ~np.isnan(rw_green), np.clip(cdom_raw, 0, 20), np.nan
    )

    rw_red = extracted.get("rw_oa08_665nm", np.full(len(cand_mask), np.nan))
    rw_nir = extracted.get("rw_oa17_865nm", np.full(len(cand_mask), np.nan))
    owt = np.zeros(len(cand_mask), dtype=int)
    for k in range(len(cand_mask)):
        if not any(np.isnan([rw_blue[k], rw_green[k], rw_red[k], rw_nir[k]])):
            owt[k] = classify_owt(
                rw_blue[k], rw_green[k], rw_red[k], rw_nir[k],
                extracted["chl_a_ug_l"][k],
            )
    extracted["owt_class_v1_provisional"] = owt

    # ── 7. Build output rows ──────────────────────────────────────────────────
    station_ids = stations["station_id"].values
    out_cols    = [c for c in extracted if c != "kd490"]  # kd490 is intermediate only

    rows = []
    for k, st_global_idx in enumerate(cand_mask):
        row: dict = {
            "station_id":            str(station_ids[st_global_idx]),
            "n_valid_pixels":        1,
            "cloud_frac":            0.0,
            "ac_method":             "olci_l2",
            "rw_blue_negative_flag": int(rw_blue[k] < 0) if not np.isnan(rw_blue[k]) else 0,
        }
        for col in out_cols:
            v = extracted[col][k]
            row[col] = float(v) if not np.isnan(v) else np.nan
        rows.append(row)

    return rows


def _extract_olci_lake(zf: zipfile.ZipFile, lake_bbox: tuple) -> Optional[dict]:
    """
    Extract lake-mean WQ values from an open OLCI L2 WFR zip file.
    Returns None when < 10 valid water pixels found in lake bbox.
    """
    import xarray as xr

    w, s, e, n = lake_bbox

    # Find .SEN3 product directory in zip
    names = zf.namelist()
    prod_dirs = sorted({nm.split("/")[0] for nm in names if "/" in nm})
    if not prod_dirs:
        return None
    prod = prod_dirs[0]

    def _read_nc(nc_name: str) -> Optional[object]:
        full = f"{prod}/{nc_name}"
        if full not in names:
            return None
        try:
            raw = zf.read(full)
            return xr.open_dataset(BytesIO(raw), engine="h5netcdf")
        except Exception:
            return None

    # Load geo coordinates
    geo_ds = _read_nc("geo_coordinates.nc")
    if geo_ds is None:
        return None

    lat = geo_ds["latitude"].values
    lon = geo_ds["longitude"].values
    geo_ds.close()

    lake_mask = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
    n_lake = int(lake_mask.sum())
    if n_lake == 0:
        return None

    # Quality flags — file is wqsf.nc (not WQSF_lsb.nc) in baseline 003+
    flags_ds = _read_nc("wqsf.nc")
    if flags_ds is None:
        return None
    flags = flags_ds[list(flags_ds.data_vars)[0]].values.astype(np.int64)
    flags_ds.close()

    water = (flags & (_WQSF_WATER | _WQSF_INLAND_WATER)) != 0
    cloud = (flags & (_WQSF_CLOUD | _WQSF_CLOUD_AMB)) != 0
    bad   = (flags & (_WQSF_SNOW_ICE | _WQSF_AC_FAIL | _WQSF_INVALID)) != 0
    valid = lake_mask & water & ~cloud & ~bad

    n_valid = int(valid.sum())
    cloud_frac = float((lake_mask & cloud).sum()) / max(n_lake, 1)

    if n_valid < 10:
        return None

    result: dict = {"n_valid_pixels": n_valid, "cloud_frac": cloud_frac}

    # L2 WQ products — Kd490 is in trsp.nc as KD490_M07 (not kd490_m07.nc)
    for col_name, nc_file, var_name in [
        ("chl_a_oc4me", "chl_oc4me.nc", None),
        ("chl_a_nn",    "chl_nn.nc",    None),
        ("tsm_nn",      "tsm_nn.nc",    None),
        ("kd490",       "trsp.nc",      "KD490_M07"),
    ]:
        ds = _read_nc(nc_file)
        if ds is not None:
            var = var_name if (var_name and var_name in ds.data_vars) \
                  else list(ds.data_vars)[0]
            arr = ds[var].values
            result[col_name] = float(np.nanmean(arr[valid]))
            ds.close()
        else:
            result[col_name] = np.nan

    # Rw reflectance bands
    for band_name, wl in OLCI_BANDS.items():
        ds = _read_nc(f"{band_name}_reflectance.nc")
        col_name = f"rw_{band_name.lower()}_{wl}nm"
        if ds is not None:
            arr = ds[list(ds.data_vars)[0]].values
            result[col_name] = float(np.nanmean(arr[valid]))
            ds.close()
        else:
            result[col_name] = np.nan

    # Secchi from Kd(490) (Kratzer 2008: Zsd = 1.7 / Kd490)
    kd490 = result.get("kd490", np.nan)
    if not np.isnan(kd490) and kd490 > 0:
        result["secchi_m"] = float(np.clip(1.7 / kd490, 0.1, 40.0))
    else:
        result["secchi_m"] = np.nan

    # Turbidity proxy from TSM_NN (TSM g/m³ → FNU ≈ 0.8 × TSM for mineral-dominated)
    tsm = result.get("tsm_nn", np.nan)
    result["turbidity_fnu"] = float(tsm * 0.8) if not np.isnan(tsm) else np.nan

    # Chl-a: prefer neural-network retrieval
    chl_nn = result.get("chl_a_nn", np.nan)
    result["chl_a_ug_l"] = float(chl_nn) if not np.isnan(chl_nn) else result.get("chl_a_oc4me", np.nan)

    # CDOM proxy from Rw_443 / Rw_560
    rw_blue  = result.get("rw_oa03_443nm", np.nan)
    rw_green = result.get("rw_oa06_560nm", np.nan)
    if not any(np.isnan([rw_blue, rw_green])):
        result["cdom_m"] = float(
            np.clip(-0.5 * np.log(max(rw_blue, 1e-9) / max(rw_green, 1e-9)), 0, 20)
        )
    else:
        result["cdom_m"] = np.nan

    # OWT
    rw_red = result.get("rw_oa08_665nm", np.nan)
    rw_nir = result.get("rw_oa17_865nm", np.nan)
    if not any(np.isnan([rw_blue, rw_green, rw_red, rw_nir])):
        result["owt_class_v1_provisional"] = classify_owt(
            rw_blue, rw_green, rw_red, rw_nir, result.get("chl_a_ug_l")
        )
    else:
        result["owt_class_v1_provisional"] = 0

    return result


def extract_sentinel3_year(
    year: int,
    mode: str = "stats",
    lake_polygons: Optional[list] = None,
    stations: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Download and extract S3 OLCI WFR for one year via CDSE.

    --mode stats    : lake-polygon means (2877 HydroLAKES lakes, output: lake_id)
    --mode stations : per-station pixel extraction for all 18k+ stations via
                      KD-tree nearest-pixel lookup (output: station_id)
    """
    if year < 2016:
        logger.warning(f"  S3 OLCI: Sentinel-3A launched April 2016, skipping {year}")
        return pd.DataFrame()

    try:
        token = _get_cdse_token()
    except Exception as exc:
        logger.error(f"CDSE auth failed: {exc}")
        return pd.DataFrame()

    if mode == "stations":
        if stations is None:
            stations = load_stations()
        logger.info(
            f"  S3 OLCI {year} [stations mode]: {len(stations)} stations"
        )
    else:
        if lake_polygons is None:
            lake_polygons = _fallback_lakes()

    logger.info(f"  S3 OLCI {year}: searching products over Sweden bbox...")
    products = _search_olci_wfr(SWEDEN_UNION_BBOX, year)
    logger.info(f"  S3 OLCI {year}: {len(products)} products found")

    # ── Season filter: keep only May–Oct (ice-free) ───────────────────────────
    # Winter products are (a) ice/snow-flagged in WQSF, (b) have no SLU MVM
    # samples to match. Cutting ~50% of products before any download.
    def _prod_month(p):
        try:
            return pd.Timestamp(p.get("ContentDate", {}).get("Start", "")).month
        except Exception:
            return 0

    products = [p for p in products if 5 <= _prod_month(p) <= 10]
    logger.info(f"  S3 OLCI {year}: {len(products)} ice-free products (May–Oct)")

    if not products:
        return pd.DataFrame()

    # ── Parallel download + extract (4 workers) ───────────────────────────────
    # Thread-safe shared token; each worker refreshes on 401 and notifies others
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _token     = [token]
    _token_lck = threading.Lock()

    def _process_one(prod):
        prod_id   = prod["Id"]
        prod_name = prod.get("Name", "")
        date_raw  = prod.get("ContentDate", {}).get("Start", "")
        try:
            date = pd.Timestamp(date_raw).date()
        except Exception:
            return []

        for attempt in range(2):
            with _token_lck:
                hdrs = {"Authorization": f"Bearer {_token[0]}"}
            try:
                resp = requests.get(
                    f"{CDSE_DOWNLOAD_URL}({prod_id})/$value",
                    headers=hdrs, stream=True, timeout=300,
                )
                if resp.status_code == 401 and attempt == 0:
                    with _token_lck:
                        try:
                            _token[0] = _get_cdse_token()
                        except Exception:
                            pass
                    continue
                resp.raise_for_status()
                break
            except requests.HTTPError as exc:
                logger.warning(f"  S3 {date} HTTP {exc.response.status_code}")
                return []
            except Exception as exc:
                logger.warning(f"  S3 {date}: {exc}")
                return []

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    fh.write(chunk)
            with zipfile.ZipFile(tmp_path) as zf:
                if mode == "stations":
                    rows = _extract_olci_stations_batch(zf, stations)
                    for r in rows:
                        r["date"]    = pd.Timestamp(date)
                        r["sensor"]  = "sentinel3_olci"
                        r["product"] = prod_name
                    return rows
                else:
                    out = []
                    for lake in lake_polygons:
                        s = _extract_olci_lake(zf, lake["bbox"])
                        if s is not None:
                            out.append({
                                "date":    pd.Timestamp(date),
                                "lake_id": lake["lake_id"],
                                "sensor":  "sentinel3_olci",
                                "product": prod_name,
                                **s,
                            })
                    return out
        except zipfile.BadZipFile:
            logger.warning(f"  S3 {date} bad zip — skipping")
            return []
        except Exception as exc:
            logger.warning(f"  S3 {date}: {exc}")
            return []
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    all_rows  = []
    completed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_process_one, p): p for p in products}
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)
            completed += 1
            if completed % 50 == 0:
                n_st = len({r.get("station_id") or r.get("lake_id") for r in all_rows})
                logger.info(
                    f"  S3 {year}: {completed}/{len(products)} done — "
                    f"{len(all_rows)} rows, {n_st} unique IDs"
                )
                with _token_lck:
                    try:
                        _token[0] = _get_cdse_token()
                    except Exception:
                        pass

    if not all_rows:
        return pd.DataFrame()

    df  = pd.DataFrame(all_rows)
    key = "station_id" if mode == "stations" else "lake_id"
    df  = df.sort_values([key, "date"]).reset_index(drop=True)
    return df


# ── Output ────────────────────────────────────────────────────────────────────

def save_year(df: pd.DataFrame, sensor: str, year: int, suffix: str = "") -> None:
    if df.empty:
        logger.warning(f"  {sensor} {year}: no data to save")
        return
    SAT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAT_DIR / f"sweden_{sensor}_{year}{suffix}.parquet"
    df.to_parquet(out_path, index=False)
    id_col = "station_id" if "station_id" in df.columns else "lake_id"
    id_label = "stations" if id_col == "station_id" else "lakes"
    logger.info(
        f"  {sensor} {year}: saved {len(df)} rows "
        f"({df[id_col].nunique()} {id_label}) -> {out_path}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

_SENSOR_EXTRACTORS = {
    "modis":     (extract_modis_year,      2000, 2025),
    "landsat":   (extract_landsat_year,    2000, 2025),
    "sentinel1": (extract_sentinel1_year,  2014, 2025),
    "sentinel2": (extract_sentinel2_year,  2015, 2025),
    "sentinel3": (extract_sentinel3_year,  2016, 2025),
}


def run(sensor: str, start_year: int, end_year: int, mode: str = "stats") -> None:
    if sensor not in _SENSOR_EXTRACTORS:
        logger.error(f"Unknown sensor '{sensor}'. Choose from: {list(_SENSOR_EXTRACTORS)}")
        return

    extract_fn, yr_lo, yr_hi = _SENSOR_EXTRACTORS[sensor]
    start_year = max(start_year, yr_lo)
    end_year   = min(end_year,   yr_hi)

    stations      = None
    lake_polygons = None

    if mode == "stations":
        try:
            stations = load_stations()
            logger.info(f"  stations mode: {len(stations)} stations loaded")
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return

    if mode == "stats" and sensor != "sentinel3":
        # S3 stats mode loads its own lake list inside extract_sentinel3_year
        _monitor_stations = load_stations()
        ee = _gee_init()
        lake_polygons = load_lake_polygons_gee(ee, _monitor_stations)
        logger.info(f"  stats mode: {len(lake_polygons)} monitored lakes (HydroLAKES)")

    suffix = "_stations" if mode == "stations" else ""
    logger.info(f"Starting {sensor} ({mode}) extraction {start_year}–{end_year}")

    for year in range(start_year, end_year + 1):
        out_path     = SAT_DIR / f"sweden_{sensor}_{year}{suffix}.parquet"
        out_path_old = SAT_DIR / f"sweden_{sensor}{suffix}_{year}.parquet"
        if out_path.exists() or out_path_old.exists():
            logger.info(f"  {sensor} {year}: already exists, skipping")
            continue

        logger.info(f"  Processing {sensor} {year}...")
        if sensor == "sentinel3":
            df = extract_fn(year, mode=mode, stations=stations,
                            lake_polygons=lake_polygons)
        elif mode == "stations":
            df = extract_fn(year, mode=mode, stations=stations)
        else:
            df = extract_fn(year, lake_polygons=lake_polygons)
        save_year(df, sensor, year, suffix=suffix)

    logger.info(f"{sensor} ({mode}) extraction complete")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SW-STEP-5: Satellite WQ extraction for Swedish lakes and WQ stations"
    )
    parser.add_argument(
        "--sensor",
        choices=list(_SENSOR_EXTRACTORS),
        required=True,
        help="Sensor: modis | landsat | sentinel1 | sentinel2 | sentinel3",
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year",   type=int, default=2025)
    parser.add_argument(
        "--mode",
        choices=["stats", "stations"],
        default="stats",
        help=(
            "stats    — lake-mean reflectance per date (one row per lake)\n"
            "stations — pixel at each WQ station per date (one row per station)\n"
            "           Requires smhi_hydroobs_stations.csv and slu_mvm_stations.csv."
        ),
    )
    args = parser.parse_args()
    run(sensor=args.sensor, start_year=args.start_year, end_year=args.end_year, mode=args.mode)


if __name__ == "__main__":
    main()
