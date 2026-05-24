#!/usr/bin/env python3
"""
SW-STEP-6 — Extract MESAN gridded met at SLU MVM station locations.

Reads annual MESAN zarr stores (data/raw/mesan/zarr/mesan_YYYY.zarr) and extracts
all available variables at the nearest grid cell for each of the 18,743 measured
SLU MVM stations.

Two zarr formats exist:
  2000-2023: Only 'time' stored as coordinate. Geographic position is recovered
             by reading GRIB rotation params from variable attrs (lat_sp, lon_sp,
             first-grid-point, increment) and un-rotating station WGS84 coords
             into the rotated lat/lon frame via pyproj — then direct index lookup.
  2024-2025: 'latitude' and 'longitude' stored as 2D coordinate arrays.
             Uses cKDTree nearest-neighbour lookup.

HIRLAM (2000-2013): ~17 variables, grid 116×102,  rotation (lat_sp=-30, lon_sp=-10)
AROME  (2014-2023): ~22-27 variables, grid 779×630, rotation (lat_sp=-30, lon_sp=+15)
AM25H2 (2024-2025): ~11 variables, grid 1069×949, lat/lon stored in zarr

Output: data/interim/SE/mesan/station_mesan_YYYY.parquet
  Columns: station_id, date, <mesan_vars...>, mesan_product, nearest_grid_distance_m

Note: Nearest-neighbour grid assignment is used for all MESAN variables.
      Bilinear interpolation would be preferable for smooth fields (T, P, wind)
      and is recommended for future work. nearest_grid_distance_m allows
      downstream quality-screening by distance threshold.

Usage:
    python -m src.processing.station_grid_join
    python -m src.processing.station_grid_join --start-year 2010 --end-year 2020
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW      = Path("data/raw")
INTERIM  = Path("data/interim/SE/mesan")
MESAN    = RAW / "mesan" / "zarr"
STATIONS = RAW / "slu_mvm" / "slu_mvm_measured_station_ids.csv"


def load_stations() -> pd.DataFrame:
    df = pd.read_csv(STATIONS, usecols=["station_id", "latitude", "longitude"])
    df["station_id"] = df["station_id"].astype(str)
    return df.dropna(subset=["latitude", "longitude"])


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised haversine distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlam = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Method A: 2024-2025 — lat/lon stored as 2D coordinate arrays ──────────────

def _indices_from_kdtree(ds: xr.Dataset, stations: pd.DataFrame):
    """Return (y_idxs, x_idxs, dist_m) for each station using KD-tree on stored 2D lat/lon."""
    lats_2d = ds["latitude"].values   # (y, x)
    lons_2d = ds["longitude"].values  # (y, x)
    ny, nx = lats_2d.shape
    coords = np.column_stack([lats_2d.ravel(), lons_2d.ravel()])
    tree = cKDTree(coords)
    st_coords = stations[["latitude", "longitude"]].values
    _, flat_idx = tree.query(st_coords)
    flat_ys, flat_xs = np.unravel_index(np.arange(ny * nx), (ny, nx))
    y_idxs = flat_ys[flat_idx]
    x_idxs = flat_xs[flat_idx]
    matched_lats = lats_2d.ravel()[flat_idx]
    matched_lons = lons_2d.ravel()[flat_idx]
    dist_m = _haversine_m(
        stations["latitude"].values.astype(float),
        stations["longitude"].values.astype(float),
        matched_lats, matched_lons,
    )
    return y_idxs, x_idxs, dist_m


# ── Method B: 2000-2023 — un-rotate stations into GRIB rotated lat/lon frame ──

def _read_grib_grid_params(ds: xr.Dataset):
    """
    Read GRIB rotation parameters from ds.attrs (dataset-level, consistent across
    all 2000-2023 zarr stores). Returns dict with keys: lat_sp, lon_sp, lat0, lon0,
    dlat, dlon, ny, nx — or None if not a rotated_ll grid (e.g. 2024-2025).
    """
    a = ds.attrs
    if a.get("GRIB_gridType") != "rotated_ll":
        return None
    return {
        "lat_sp": float(a["GRIB_latitudeOfSouthernPoleInDegrees"]),
        "lon_sp": float(a["GRIB_longitudeOfSouthernPoleInDegrees"]),
        "lat0":   float(a["GRIB_latitudeOfFirstGridPointInDegrees"]),
        "lon0":   float(a["GRIB_longitudeOfFirstGridPointInDegrees"]),
        "dlat":   float(a["GRIB_jDirectionIncrementInDegrees"]),
        "dlon":   float(a["GRIB_iDirectionIncrementInDegrees"]),
        "ny":     int(a["GRIB_Ny"]),
        "nx":     int(a["GRIB_Nx"]),
    }


def _indices_from_rotation(ds: xr.Dataset, stations: pd.DataFrame):
    """
    Un-rotate station WGS84 lat/lon into the GRIB rotated frame, then compute
    grid (y, x) indices directly — O(n_stations), no KD-tree needed.
    Returns (y_idx, x_idx, dist_m).
    """
    p = _read_grib_grid_params(ds)
    if p is None:
        return None, None, None

    # CF convention: grid north pole = (-lat_sp, lon_sp + 180°)
    grid_np_lat = -p["lat_sp"]
    grid_np_lon = (p["lon_sp"] + 180.0) % 360.0

    crs_rotated = CRS.from_cf({
        "grid_mapping_name": "rotated_latitude_longitude",
        "grid_north_pole_latitude":  grid_np_lat,
        "grid_north_pole_longitude": grid_np_lon,
    })
    crs_wgs84 = CRS.from_epsg(4326)
    t_fwd = Transformer.from_crs(crs_wgs84, crs_rotated, always_xy=True)

    rlons, rlats = t_fwd.transform(
        stations["longitude"].values.astype(float),
        stations["latitude"].values.astype(float),
    )

    # Direct index from rotated first-point + increment
    x_idx = np.round((rlons - p["lon0"]) / p["dlon"]).astype(int)
    y_idx = np.round((rlats - p["lat0"]) / p["dlat"]).astype(int)
    x_idx = np.clip(x_idx, 0, p["nx"] - 1)
    y_idx = np.clip(y_idx, 0, p["ny"] - 1)

    # Back-transform matched cell centres to WGS84 for haversine distance
    matched_rlon = p["lon0"] + x_idx * p["dlon"]
    matched_rlat = p["lat0"] + y_idx * p["dlat"]
    t_inv = Transformer.from_crs(crs_rotated, crs_wgs84, always_xy=True)
    matched_lons, matched_lats = t_inv.transform(matched_rlon, matched_rlat)
    dist_m = _haversine_m(
        stations["latitude"].values.astype(float),
        stations["longitude"].values.astype(float),
        matched_lats, matched_lons,
    )

    return y_idx, x_idx, dist_m


# ── Core extraction ────────────────────────────────────────────────────────────

def extract_year(year: int, stations: pd.DataFrame) -> pd.DataFrame:
    zarr_path = MESAN / f"mesan_{year}.zarr"
    if not zarr_path.exists():
        logger.warning(f"  {year}: zarr not found — skipping")
        return pd.DataFrame()

    mesan_product = "HIRLAM" if year <= 2013 else "AROME"
    logger.info(f"  {year}: loading {zarr_path.name}  [{mesan_product}]")

    ds = xr.open_zarr(str(zarr_path))

    # Choose lookup method based on what's stored in the zarr
    if "latitude" in ds.coords and "longitude" in ds.coords:
        # 2024-2025: geographic coords stored directly
        y_idxs, x_idxs, dist_m = _indices_from_kdtree(ds, stations)
    else:
        # 2000-2023: recover from GRIB rotation params
        y_idxs, x_idxs, dist_m = _indices_from_rotation(ds, stations)
        if y_idxs is None:
            logger.error(f"  {year}: could not determine grid indices — skipping")
            ds.close()
            return pd.DataFrame()

    dates = pd.to_datetime(ds["time"].values).date
    data_vars = list(ds.data_vars)
    logger.info(f"  {year}: {len(data_vars)} variables, {len(dates)} days, {len(stations)} stations")

    # Vectorised extraction: arr[:,y_idxs,x_idxs] → shape (n_time, n_stations)
    var_arrays = {}
    for var in data_vars:
        try:
            arr = ds[var].values  # (time, y, x)
            var_arrays[var] = arr[:, y_idxs, x_idxs]
        except Exception as exc:
            logger.debug(f"  {year}/{var}: skipped — {exc}")

    ds.close()

    n_time     = len(dates)
    n_stations = len(stations)
    rows = {
        "station_id":              np.tile(stations["station_id"].values, n_time),
        "date":                    np.repeat(dates, n_stations),
        "mesan_product":           mesan_product,
        "nearest_grid_distance_m": np.tile(dist_m.astype(np.float32), n_time),
    }
    for var, mat in var_arrays.items():
        rows[var] = mat.ravel()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Entry points ───────────────────────────────────────────────────────────────

def run(start_year: int = 2000, end_year: int = 2025) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)

    logger.info("Loading station list")
    stations = load_stations()
    logger.info(f"  {len(stations):,} stations with coordinates")

    for year in range(start_year, end_year + 1):
        out_path = INTERIM / f"station_mesan_{year}.parquet"
        if out_path.exists():
            logger.info(f"  {year}: already exists, skipping")
            continue

        df = extract_year(year, stations)
        if df.empty:
            logger.warning(f"  {year}: no data extracted")
            continue

        df.to_parquet(out_path, index=False)
        logger.info(f"  {year}: wrote {len(df):,} rows → {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Extract MESAN at SLU MVM station locations")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year",   type=int, default=2025)
    parser.add_argument("--country",    type=str, default="SE")
    args = parser.parse_args()

    if args.country != "SE":
        logger.warning("Only SE is supported in v1 — ignoring --country")

    run(args.start_year, args.end_year)


if __name__ == "__main__":
    main()
