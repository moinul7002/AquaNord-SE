#!/usr/bin/env python3
"""
ERA5-Land meteorological patches for WQ monitoring stations via CDS API.

NOT the snow-hydrology ERA5 patch script (scripts/download_era5_land_patches.py).
This script targets WQ-relevant meteorological drivers with a larger spatial
footprint matched to catchment-scale forcing processes.

Dataset:    reanalysis-era5-land  (~9 km, 0.1° grid)
Variables:  t2m    2 m temperature                  (K → °C)
            tp     total precipitation               (m → mm)
            u10    10 m U-wind component             (m/s)
            v10    10 m V-wind component             (m/s)
            ssrd   surface solar radiation downwards (J/m²)
            d2m    2 m dewpoint temperature          (K → °C)
            sp     surface pressure                  (Pa → hPa)
            sro    surface runoff                    (m)
            ro     total runoff                      (m)

Footprint:  144 km × 144 km  (half_m = 72 000 m  →  ~12 × 12 native pixels at 9 km)
Canonical:  12 × 12  (native resolution; no artificial upsample — ERA5 has no sub-grid detail to preserve)
Coverage:   2000–2025, all 12 months

Design rationale — why 144 km for ERA5 while optical sensors use 10 km:
  ERA5's 9 km grid cannot represent spatial gradients at 10 km scale.  144 km
  captures the upstream catchment meteorology that drives WQ (runoff, sediment
  load, thermal stratification) while keeping ~16 native cells for meaningful
  spatial variation.

Download strategy: one Nordic-bbox NetCDF per month (same as snow ERA5 script)
then extract per-station patches from the cached file.  Accumulated variables
(tp, ssrd, sro, ro) are stored as 12:00 UTC snapshots (12-hour accumulation).
Instantaneous variables (t2m, u10, v10, d2m, sp) are 12:00 UTC values.

Output:
    data/processed/era5_wq_patches/<station_id>/<YYYY>.npz
        patch      (days_in_year, 32, 32, 9)  float32   canonical tensor; NaN for missing days
        mask       (days_in_year, 32, 32)      uint8     1 = all-variable finite
        dates      (days_in_year,)             |S10      every calendar day in the year
        channels   (9,)                        object    short variable names
        station_id, latitude, longitude, footprint_m

Usage:
    python -m src.patches.era5_wq_patches \\
        --start-year 2000 --end-year 2025

    # Pilot: 10 stations, 2020
    python -m src.patches.era5_wq_patches \\
        --start-year 2020 --end-year 2020 --pilot 10
"""

import argparse
import calendar
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.patches._utils import (
    build_year_month_list, load_wq_stations,
    resize_to_canonical, save_shard, shard_exists,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

# CDS long names → short names used as channel labels
_ERA5_VARS = {
    '2m_temperature':                    't2m',
    'total_precipitation':               'tp',
    '10m_u_component_of_wind':           'u10',
    '10m_v_component_of_wind':           'v10',
    'surface_solar_radiation_downwards': 'ssrd',
    '2m_dewpoint_temperature':           'd2m',
    'surface_pressure':                  'sp',
    'surface_runoff':                    'sro',
    'runoff':                            'ro',
}
_CHANNELS = list(_ERA5_VARS.values())

# Unit conversions applied after extraction
_K_TO_C   = {'t2m', 'd2m'}       # subtract 273.15
_PA_TO_HPA = {'sp'}               # divide by 100
_M_TO_MM   = {'tp', 'sro', 'ro'}  # multiply by 1000

NORDIC_AREA  = [71.0, 5.0, 55.0, 32.0]  # [N, W, S, E] for CDS request
PATCH_HALF_M = 72_000.0                   # 144 km × 144 km footprint
ERA5_RES_DEG = 0.1                        # ~9 km per grid cell
NATIVE_CELLS = round(PATCH_HALF_M / (ERA5_RES_DEG * 111_000))  # ~6 half → 12 cells per side
CANONICAL    = 2 * NATIVE_CELLS   # keep at native resolution — no artificial upsample
SUFFIX       = 'ERA5'


# ── CDS bootstrap ─────────────────────────────────────────────────────────────

def _check_cdsapi():
    try:
        import cdsapi
        return cdsapi
    except ImportError:
        logger.error(
            "cdsapi not installed.  Run: pip install cdsapi\n"
            "Configure ~/.cdsapirc — register at https://cds.climate.copernicus.eu/"
        )
        sys.exit(1)


# ── CDS download ──────────────────────────────────────────────────────────────

def _unzip_if_needed(path: Path) -> Path:
    if not zipfile.is_zipfile(path):
        return path
    logger.info(f"  {path.name} is a zip — extracting inner NetCDF")
    with zipfile.ZipFile(path) as z:
        nc_members = [m for m in z.namelist() if m.endswith('.nc')]
        if not nc_members:
            raise RuntimeError(f"No .nc file found inside {path}: {z.namelist()}")
        tmp = path.parent / f".{path.stem}_unzip"
        tmp.mkdir(exist_ok=True)
        z.extract(nc_members[0], tmp)
    extracted = tmp / nc_members[0]
    path.unlink()
    extracted.replace(path)
    tmp.rmdir()
    return path


def download_era5_month(
    year: int,
    month: int,
    raw_dir: Path,
    client,
    max_retries: int = 3,
) -> Optional[Path]:
    """Download ERA5-Land multi-variable NetCDF for (year, month).  Returns path or None."""
    _, n_days = calendar.monthrange(year, month)
    out_file  = raw_dir / f"era5_wq_{year}{month:02d}.nc"
    if out_file.exists():
        logger.info(f"  {out_file.name} already exists — skipping download")
        return out_file

    request = {
        'variable': list(_ERA5_VARS.keys()),
        'year':     str(year),
        'month':    f"{month:02d}",
        'day':      [f"{d:02d}" for d in range(1, n_days + 1)],
        'time':     '12:00',
        'area':     NORDIC_AREA,
        'format':   'netcdf',
    }
    for attempt in range(1, max_retries + 1):
        logger.info(f"  Downloading ERA5-WQ {year}-{month:02d} (attempt {attempt})")
        try:
            client.retrieve('reanalysis-era5-land', request, str(out_file))
            return _unzip_if_needed(out_file)
        except Exception as exc:
            logger.warning(f"  Attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                time.sleep(60)
    logger.error(f"  ERA5-WQ download failed permanently for {year}-{month:02d}")
    return None


# ── patch extraction ──────────────────────────────────────────────────────────

def _apply_unit_conversions(data: dict) -> dict:
    for short in _K_TO_C:
        if short in data:
            data[short] = data[short] - 273.15
    for short in _PA_TO_HPA:
        if short in data:
            data[short] = data[short] / 100.0
    for short in _M_TO_MM:
        if short in data:
            data[short] = data[short] * 1000.0
    return data


STATION_BATCH = 500   # stations held in RAM at once; 500 × 12.9 MB ≈ 6.5 GB peak


def _load_month_grid(nc_file: Path):
    """
    Load one monthly ERA5 NetCDF into a numpy array.

    Returns (full, lats, lons, date_strs, present_channels) or None on failure.
    The Nordic bbox at 0.1° is ~160 × 270 grid points × 31 days × 9 vars ≈ 50 MB —
    small enough to hold 12 months in memory simultaneously (~600 MB for a full year).
    """
    try:
        import xarray as xr
    except ImportError:
        logger.error("xarray not installed.  Run: pip install xarray netcdf4")
        sys.exit(1)

    ds = xr.open_dataset(nc_file)
    time_dim  = 'valid_time' if 'valid_time' in ds.coords else 'time'
    date_strs = [d.strftime('%Y-%m-%d')
                 for d in pd.to_datetime(ds[time_dim].values)]
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    var_data = {}
    for long_name, short in _ERA5_VARS.items():
        found = next((v for v in [short, long_name] if v in ds.data_vars), None)
        if found is None:
            logger.warning(f"  {long_name} ({short}) missing from {nc_file.name}")
            continue
        var_data[short] = ds[found].values.astype(np.float32)

    var_data = _apply_unit_conversions(var_data)
    present  = [s for s in _CHANNELS if s in var_data]
    ds.close()

    if not present:
        logger.error(f"No variables extracted from {nc_file.name}")
        return None

    full = np.stack([var_data[s] for s in present], axis=-1)  # (T, H, W, C)
    return full, lats, lons, date_strs, present


def _extract_batch_from_grid(
    full: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    date_strs: list,
    channels: list,
    batch_stations: pd.DataFrame,
) -> dict:
    """
    Extract patches for a batch of stations from a pre-loaded monthly grid.

    Called once per batch per month — the grid is NOT reloaded.
    Returns {station_id: {patch, mask, dates, channels, latitude, longitude}}.
    """
    T, H_g, W_g, C = full.shape
    half = NATIVE_CELLS
    result = {}

    for _, row in batch_stations.iterrows():
        sid  = str(row['station_id'])
        slat = float(row['latitude'])
        slon = float(row['longitude'])

        cy = int(np.argmin(np.abs(lats - slat)))
        cx = int(np.argmin(np.abs(lons - slon)))

        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half

        native = np.full((T, 2 * half, 2 * half, C), np.nan, dtype=np.float32)
        sy0, sy1 = max(y0, 0), min(y1, H_g)
        sx0, sx1 = max(x0, 0), min(x1, W_g)
        dy0, dx0 = sy0 - y0, sx0 - x0
        native[:, dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0), :] = \
            full[:, sy0:sy1, sx0:sx1, :]

        # CANONICAL == 2*NATIVE_CELLS == native size — no resize needed for ERA5
        patch_32 = native.astype(np.float32)
        mask_32  = (~np.isnan(patch_32).any(axis=-1)).astype(np.uint8)

        result[sid] = {
            'patch':    patch_32,
            'mask':     mask_32,
            'dates':    date_strs,
            'channels': channels,
            'latitude': slat,
            'longitude': slon,
        }
    return result


# ── annual shard helpers ──────────────────────────────────────────────────────

def _annual_shard_path(shard_dir: Path, station_id: str, year: int) -> Path:
    return shard_dir / station_id / f"{year}.npz"


def all_annual_shards_exist(stations: pd.DataFrame, shard_dir: Path, year: int) -> bool:
    return all(
        _annual_shard_path(shard_dir, str(sid), year).exists()
        for sid in stations['station_id']
    )


def save_annual_shard(
    shard_dir: Path,
    station_id: str,
    lat: float,
    lon: float,
    year: int,
    patch: np.ndarray,
    mask: np.ndarray,
    dates: list,
    channels: list,
) -> Path:
    station_dir = shard_dir / station_id
    station_dir.mkdir(parents=True, exist_ok=True)
    out = station_dir / f"{year}.npz"
    np.savez_compressed(
        out,
        patch=patch,
        mask=mask,
        dates=np.array(dates, dtype='S10'),
        channels=np.array(channels, dtype=object),
        station_id=station_id,
        latitude=np.float64(lat),
        longitude=np.float64(lon),
        footprint_m=np.float64(PATCH_HALF_M * 2),
    )
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='ERA5-Land WQ patches (32×32, 10 met vars, 144 km footprint) via CDS.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--slu-station-file',    default='data/raw/slu_mvm/slu_mvm_stations.csv')
    p.add_argument('--hydobs-station-file', default=None)
    p.add_argument('--fmi-station-file',    default=None)
    p.add_argument('--raw-dir',   default='data/raw/era5_wq_nc',
                   help='Directory for cached monthly NetCDF downloads')
    p.add_argument('--shard-dir', default='data/processed/era5_wq_patches')
    p.add_argument('--start-year', type=int, default=2000)
    p.add_argument('--end-year',   type=int, default=2025)
    p.add_argument('--pilot',      type=int, default=None)
    args = p.parse_args()

    cdsapi = _check_cdsapi()

    raw_dir   = Path(args.raw_dir)
    shard_dir = Path(args.shard_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    stations = load_wq_stations(
        slu_file    = Path(args.slu_station_file),
        hydobs_file = Path(args.hydobs_station_file) if args.hydobs_station_file else None,
        fmi_file    = Path(args.fmi_station_file)    if args.fmi_station_file    else None,
        pilot_n     = args.pilot,
    )

    logger.info(f"Footprint: {int(PATCH_HALF_M*2/1000)} km × {int(PATCH_HALF_M*2/1000)} km"
                f"  →  ~{NATIVE_CELLS*2} × {NATIVE_CELLS*2} native cells  →  32×32 canonical")

    client = cdsapi.Client()
    t0 = time.time()
    years = list(range(args.start_year, args.end_year + 1))
    n_stations = len(stations)
    n_batches  = (n_stations + STATION_BATCH - 1) // STATION_BATCH
    logger.info(
        f"Years: {years[0]}–{years[-1]}  stations: {n_stations:,}  "
        f"batches/year: {n_batches}  batch_size: {STATION_BATCH}"
    )

    for yi, y in enumerate(years, 1):
        logger.info(f"\n[{yi}/{len(years)}] {y}")

        if all_annual_shards_exist(stations, shard_dir, y):
            logger.info(f"  All annual shards exist for {y} — skipping")
            continue

        # Full-year date list (every calendar day)
        year_dates = []
        for m in range(1, 13):
            _, n_days = calendar.monthrange(y, m)
            for d in range(1, n_days + 1):
                year_dates.append(f"{y}-{m:02d}-{d:02d}")
        date_to_idx = {d: i for i, d in enumerate(year_dates)}
        n_days_year = len(year_dates)

        # Phase 1 — load all 12 monthly grids once (~600 MB total for Nordic bbox)
        monthly_grids: dict = {}
        for m in range(1, 13):
            nc_file = download_era5_month(y, m, raw_dir, client)
            if nc_file is None:
                logger.warning(f"  Download failed for {y}-{m:02d} — days will be NaN")
                continue
            grid = _load_month_grid(nc_file)
            if grid is not None:
                monthly_grids[m] = grid
                logger.info(f"  {y}-{m:02d} grid loaded")

        if not monthly_grids:
            logger.warning(f"  No grids for {y} — skipping")
            continue

        # Discover channel info from first available grid
        first_grid = next(iter(monthly_grids.values()))
        channels_list = first_grid[4]
        n_bands       = len(channels_list)

        # Phase 2 — extract station batches from pre-loaded grids
        n_written = 0
        for b_idx in range(n_batches):
            batch_df  = stations.iloc[b_idx * STATION_BATCH:(b_idx + 1) * STATION_BATCH]
            batch_sids = batch_df['station_id'].astype(str).tolist()

            if all(_annual_shard_path(shard_dir, sid, y).exists() for sid in batch_sids):
                n_written += len(batch_sids)
                continue

            # Annual accumulator arrays for this batch only
            batch_patches: dict = {}
            batch_masks:   dict = {}
            lat_map:       dict = {}
            lon_map:       dict = {}
            for row in batch_df.itertuples(index=False):
                sid = str(row.station_id)
                lat_map[sid] = float(row.latitude)
                lon_map[sid] = float(row.longitude)
                batch_patches[sid] = np.full(
                    (n_days_year, CANONICAL, CANONICAL, n_bands), np.nan, dtype=np.float32
                )
                batch_masks[sid] = np.zeros(
                    (n_days_year, CANONICAL, CANONICAL), dtype=np.uint8
                )

            for m, (full, lats, lons, date_strs, _) in monthly_grids.items():
                month_data = _extract_batch_from_grid(
                    full, lats, lons, date_strs, channels_list, batch_df
                )
                for sid, payload in month_data.items():
                    for d_str, p, mk in zip(payload['dates'], payload['patch'], payload['mask']):
                        t = date_to_idx.get(d_str)
                        if t is not None:
                            batch_patches[sid][t] = p
                            batch_masks[sid][t]   = mk

            for sid in batch_sids:
                save_annual_shard(
                    shard_dir=shard_dir, station_id=sid,
                    lat=lat_map[sid], lon=lon_map[sid],
                    year=y,
                    patch=batch_patches[sid],
                    mask=batch_masks[sid],
                    dates=year_dates,
                    channels=channels_list,
                )
                n_written += 1

            if (b_idx + 1) % 10 == 0 or b_idx == n_batches - 1:
                logger.info(f"  {y}  batch {b_idx + 1}/{n_batches}  shards written: {n_written:,}")

        logger.info(f"  {y} complete — {n_written:,} annual shards")

    logger.info(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == '__main__':
    main()
