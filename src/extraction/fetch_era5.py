"""
ERA5-Land + ERA5 single-levels extractor (SW-STEP-4).

Downloads ERA5-Land (2m temperature, precipitation, snow depth) and
ERA5 single-levels (total cloud cover) monthly for the Nordic domain,
aggregates hourly → daily, and extracts values at WQ station locations
using nearest-grid-point interpolation.

CDS API: https://cds.climate.copernicus.eu/api  (cdsapi >= 0.7.4)
NetCDF4 library required: pip install netcdf4

Variables saved per station per day:
  t2m_mean_c  — 2 m air temperature daily mean (°C)
  t2m_min_c   — 2 m air temperature daily min  (°C)
  t2m_max_c   — 2 m air temperature daily max  (°C)
  tp_mm       — total precipitation daily sum  (mm)
  sde_m       — snow depth water equivalent at 06 UTC (m)
  tcc_frac    — total cloud cover daily mean   (0–1)
  lmlt_c      — ERA5-Land lake mixed-layer temperature (°C, FLake output)
                cross-validates SLU MVM water_temp_c (Tier 1 evaluation)

ERA5 known quirks handled:
  - Precipitation: accumulated from 00 UTC → daily total via diff + clip
  - Temperature: Kelvin → Celsius (subtract 273.15)
  - Dimension name is 'valid_time' not 'time' in downloaded NetCDF

Usage:
    python -m src.extraction.fetch_era5
    python -m src.extraction.fetch_era5 --start-year 2020 --end-year 2025
    python -m src.extraction.fetch_era5 --start-year 2000 --end-year 2025
"""

import sys
import os
import argparse
import datetime
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import RAW_DIR
from common.logging_utils import setup_logger

logger = setup_logger("ERA5")

ERA5_DIR = RAW_DIR / "era5"

# Nordic domain bbox: covers Sweden + Finland + surrounding area
NORDIC_BBOX = [71, 10, 55, 32]   # [N, W, S, E] in EPSG:4326

# Station CSV files to load (Sweden WQ monitoring network)
STATION_CSVS = [
    RAW_DIR / "hydroobs"   / "smhi_hydroobs_stations.csv",
    RAW_DIR / "slu_mvm"    / "slu_mvm_stations.csv",
]


# ── Station loading ───────────────────────────────────────────────────────────

def load_stations() -> pd.DataFrame:
    """Load and merge all Sweden WQ station locations."""
    frames = []
    for p in STATION_CSVS:
        if p.exists():
            df = pd.read_csv(p, usecols=["station_id", "latitude", "longitude"])
            df["source"] = p.stem
            frames.append(df)
        else:
            logger.warning(f"Station CSV not found: {p}")

    if not frames:
        raise FileNotFoundError("No station CSVs found — run HydObs and SLU MVM extractors first")

    stations = pd.concat(frames, ignore_index=True)
    stations = stations.dropna(subset=["latitude", "longitude"])

    # Keep one record per unique grid cell (~0.1° resolution)
    stations["lat_r"] = stations["latitude"].round(1)
    stations["lon_r"] = stations["longitude"].round(1)
    stations = stations.drop_duplicates(subset=["lat_r", "lon_r"]).drop(columns=["lat_r", "lon_r"])

    # Clip to Nordic bbox
    n, w, s, e = NORDIC_BBOX
    stations = stations[
        (stations["latitude"]  >= s) & (stations["latitude"]  <= n) &
        (stations["longitude"] >= w) & (stations["longitude"] <= e)
    ].reset_index(drop=True)

    logger.info(f"Loaded {len(stations)} unique station grid-cells from {len(frames)} CSVs")
    return stations


# ── CDS download helpers ──────────────────────────────────────────────────────

def _cds_client():
    import cdsapi
    return cdsapi.Client()


def _download_era5land(c, year: int, month: int, tmp_path: str) -> bool:
    """Download ERA5-Land for one month. Returns True on success."""
    days_in_month = (datetime.date(year, month % 12 + 1, 1) - datetime.timedelta(days=1)).day \
        if month < 12 else 31
    try:
        r = c.retrieve(
            "reanalysis-era5-land",
            {
                "variable": ["2m_temperature", "total_precipitation", "snow_depth",
                             "lake_mix_layer_temperature"],
                "year":  str(year),
                "month": f"{month:02d}",
                "day":   [f"{d:02d}" for d in range(1, days_in_month + 1)],
                "time":  [f"{h:02d}:00" for h in range(24)],
                "format": "netcdf",
                "area":  NORDIC_BBOX,
                "download_format": "unarchived",
            },
        )
        r.download(tmp_path)
        return True
    except Exception as exc:
        logger.error(f"ERA5-Land download failed {year}-{month:02d}: {exc}")
        return False


def _download_era5sl(c, year: int, month: int, tmp_path: str) -> bool:
    """Download ERA5 single-levels (total cloud cover) for one month."""
    days_in_month = (datetime.date(year, month % 12 + 1, 1) - datetime.timedelta(days=1)).day \
        if month < 12 else 31
    try:
        r = c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "variable": ["total_cloud_cover"],
                "product_type": "reanalysis",
                "year":  str(year),
                "month": f"{month:02d}",
                "day":   [f"{d:02d}" for d in range(1, days_in_month + 1)],
                "time":  [f"{h:02d}:00" for h in range(24)],
                "format": "netcdf",
                "area":  NORDIC_BBOX,
                "download_format": "unarchived",
            },
        )
        r.download(tmp_path)
        return True
    except Exception as exc:
        logger.error(f"ERA5-SL download failed {year}-{month:02d}: {exc}")
        return False


# ── Daily aggregation ─────────────────────────────────────────────────────────

def _aggregate_daily(nc_path: str) -> "xr.Dataset":
    """
    Load hourly ERA5-Land NetCDF and aggregate to daily values.

    Handles ERA5 quirks:
      - Dimension is 'valid_time' not 'time'
      - Precipitation is accumulated from 00 UTC → diff + clip → sum
      - Temperature is in Kelvin → subtract 273.15
    """
    import xarray as xr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(nc_path, engine="netcdf4")

    # Rename valid_time → time for consistent downstream use
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    daily_vars = {}

    # Temperature (K → °C)
    if "t2m" in ds:
        t_c = ds["t2m"] - 273.15
        daily_vars["t2m_mean_c"] = t_c.resample(time="1D").mean()
        daily_vars["t2m_min_c"]  = t_c.resample(time="1D").min()
        daily_vars["t2m_max_c"]  = t_c.resample(time="1D").max()

    # Precipitation (m → mm): accumulated from 00 UTC, diff + clip to get hourly amounts
    if "tp" in ds:
        tp_mm = ds["tp"] * 1000.0
        tp_diff = tp_mm.diff("time").clip(min=0)
        daily_vars["tp_mm"] = tp_diff.resample(time="1D").sum()

    # Snow depth: take value at 06 UTC as daily representative (climate convention)
    if "sde" in ds:
        mask_06 = (ds.time.dt.hour == 6).values
        if mask_06.any():
            sde_06 = ds["sde"].isel(time=mask_06).resample(time="1D").mean()
        else:
            sde_06 = ds["sde"].resample(time="1D").mean()
        daily_vars["sde_m"] = sde_06.clip(min=0)

    # Lake mixed-layer temperature (ERA5-Land FLake output) — Tier 1 evaluation
    lmlt_var = next((v for v in ds.data_vars if "lmlt" in v.lower()
                     or "lake_mix" in v.lower()), None)
    if lmlt_var:
        daily_vars["lmlt_c"] = ds[lmlt_var].resample(time="1D").mean() - 273.15

    ds.close()

    import xarray as xr
    return xr.Dataset(daily_vars)


def _aggregate_daily_sl(nc_path: str) -> "xr.Dataset":
    """Load hourly ERA5 single-levels NetCDF and aggregate total cloud cover to daily mean."""
    import xarray as xr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(nc_path, engine="netcdf4")

    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    tcc_var = next((v for v in ds.data_vars if "tcc" in v.lower() or "cloud_cover" in v.lower()), None)
    if tcc_var is None:
        ds.close()
        return xr.Dataset()

    daily = xr.Dataset({"tcc_frac": ds[tcc_var].resample(time="1D").mean()})
    ds.close()
    return daily


# ── Station extraction ────────────────────────────────────────────────────────

def _extract_at_stations(daily_ds: "xr.Dataset", stations: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ERA5 daily values at each station location using nearest grid point.

    Returns a DataFrame with columns: station_id, date, <era5_vars...>
    """
    if daily_ds is None or len(daily_ds.data_vars) == 0:
        return pd.DataFrame()

    lats = daily_ds["latitude"].values if "latitude" in daily_ds.coords else daily_ds.coords["latitude"].values
    lons = daily_ds["longitude"].values if "longitude" in daily_ds.coords else daily_ds.coords["longitude"].values
    dates = pd.DatetimeIndex(daily_ds["time"].values)

    rows = []
    for _, st in stations.iterrows():
        lat_idx = int(np.argmin(np.abs(lats - st["latitude"])))
        lon_idx = int(np.argmin(np.abs(lons - st["longitude"])))

        rec = {"station_id": st["station_id"]}
        station_vals = {}
        for var in daily_ds.data_vars:
            try:
                arr = daily_ds[var].isel(latitude=lat_idx, longitude=lon_idx).values
                station_vals[var] = arr
            except Exception:
                station_vals[var] = np.full(len(dates), np.nan)

        for i, d in enumerate(dates):
            row = {"station_id": st["station_id"], "date": d.date()}
            for var, arr in station_vals.items():
                row[var] = float(arr[i]) if i < len(arr) else np.nan
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Main extraction loop ──────────────────────────────────────────────────────

def run(start_year: int = 2000, end_year: int = 2025) -> None:
    ERA5_DIR.mkdir(parents=True, exist_ok=True)

    try:
        stations = load_stations()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    c = _cds_client()

    for year in range(start_year, end_year + 1):
        out_path = ERA5_DIR / f"era5_sweden_{year}.parquet"
        if out_path.exists():
            logger.info(f"  {year}: already exists, skipping")
            continue

        logger.info(f"Processing ERA5 year {year}...")
        year_frames = []

        for month in range(1, 13):
            logger.info(f"  {year}-{month:02d}: downloading ERA5-Land...")

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
                land_tmp = f.name
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
                sl_tmp = f.name

            try:
                ok_land = _download_era5land(c, year, month, land_tmp)
                ok_sl   = _download_era5sl(c, year, month, sl_tmp)

                if not ok_land:
                    logger.warning(f"  {year}-{month:02d}: ERA5-Land download failed, skipping month")
                    continue

                daily_land = _aggregate_daily(land_tmp)
                df_land = _extract_at_stations(daily_land, stations)

                # ERA5-Land (0.1°) and ERA5-SL (0.25°) grids differ — extract
                # at stations separately and join DataFrames to avoid grid mismatch NaNs
                if ok_sl:
                    daily_sl = _aggregate_daily_sl(sl_tmp)
                    df_sl = _extract_at_stations(daily_sl, stations)
                    if not df_sl.empty and not df_land.empty:
                        month_df = df_land.merge(
                            df_sl, on=["station_id", "date"], how="left"
                        )
                    else:
                        month_df = df_land
                else:
                    month_df = df_land

                if not month_df.empty:
                    year_frames.append(month_df)
                    logger.info(f"  {year}-{month:02d}: extracted {len(month_df)} station-days")

            finally:
                for p in (land_tmp, sl_tmp):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

        if year_frames:
            year_df = pd.concat(year_frames, ignore_index=True)
            year_df["date"] = pd.to_datetime(year_df["date"])
            year_df = year_df.sort_values(["station_id", "date"]).reset_index(drop=True)
            year_df.to_parquet(out_path, index=False)
            logger.info(
                f"  {year}: saved {len(year_df)} rows "
                f"({year_df['station_id'].nunique()} stations) → {out_path}"
            )
        else:
            logger.warning(f"  {year}: no data extracted")

    logger.info("ERA5 extraction complete")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ERA5-Land extractor (SW-STEP-4)")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year",   type=int, default=2025)
    args = parser.parse_args()
    run(start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
