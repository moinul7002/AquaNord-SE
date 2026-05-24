"""
MESAN Extractor — SMHI Meteorological Analysis (HIRLAM 2000-2013, AROME 2014-2025).

Two SMHI grid archive products cover 2000-2025 continuously:
  Product 4 (MESAN HIRLAM): 1997-2015, ~0.4 MB/file, 17 parameters
  Product 6 (MESAN AROME):  2014-2026, ~20 MB/file, 22+ parameters

Both use the same ATOM feed + file URL structure:
  ATOM: https://opendata-download-grid-archive.smhi.se/feed/{pid}/{YYYY}/{MM}/{DD}
  File: https://opendata-download-grid-archive.smhi.se/data/{pid}/{YYYYMM}/MESAN_{YYYYMMDDHHMM}+000H00M

Downloads one analysis file per day (default 06:00 UTC), parses all GRIB2 parameters,
saves as zarr per year.

CRITICAL dependency versions:
  eccodes >= 2.35.0  (rotated-grid lat/lon fix ECC-1781)
  cfgrib  >= 0.9.10  (endStep regression fix)

Unit conversions applied at parse time:
  t, tmax, tmin, Tiw : K  -> °C  (subtract 273.15)
  MSL                : Pa -> hPa (divide by 100)
  r                  : already % (do NOT multiply by 100)
  spp                : -9 means no precipitation -> set NaN

Usage:
    # Full extraction 2000-2025 (one file per day, 06 UTC)
    python -m src.extraction.fetch_mesan --start-year 2000 --end-year 2025

    # Test: 5 files only
    python -m src.extraction.fetch_mesan --start-year 2023 --end-year 2023 --max-files 5

    # Live mode: query current conditions at all WQ station locations
    python -m src.extraction.fetch_mesan --mode live
"""

import sys
import os
import io
import re
import time
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from xml.etree import ElementTree as ET
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import RAW_DIR
from common.logging_utils import setup_logger

logger = setup_logger("MESAN")

ARCHIVE_BASE = "https://opendata-download-grid-archive.smhi.se"
LIVE_BASE    = "https://opendata-download-metanalys.smhi.se/api/category/mesan2g/version/2"

GRIB_DIR = RAW_DIR / "mesan" / "grib"
ZARR_DIR = RAW_DIR / "mesan" / "zarr"
LIVE_DIR = RAW_DIR / "mesan" / "live"

# ── Product routing ────────────────────────────────────────────────────────

def get_product_id(year: int, month: int = 1) -> int:
    """
    Return SMHI grid archive product ID for a given year/month.

    Product 4 (MESAN HIRLAM, GRIB2):  1997–2014  → cfgrib reads shortNames
    Product 6 (MESAN AROME,  GRIB2):  2014–2023  → cfgrib reads shortNames
    Product 5 (AM25H2,       GRIB1):  2024–2026  → parse_grib_edition1 (eccodes)
      AM25H2 uses SMHI local table 253 (edition 1) with paramId=0 / unknown
      shortNames in eccodes, so it is decoded via a hardcoded (iop, ltype, level,
      timeRangeIndicator) lookup in parse_grib_edition1.
    """
    if year <= 2014:
        return 4
    if year >= 2024:
        return 5
    return 6


# ── GRIB parameter map — all parameters verified from real files ──────────
# (shortName, GRIB unit) -> (output_column, conversion_fn)

def _k_to_c(x):    return x - 273.15
def _pa_to_hpa(x): return x / 100.0
def _identity(x):  return x
def _spp_fix(x):   return np.where(x == -9, np.nan, x)


GRIB_PARAMS = {
    # Temperature
    "t":       ("air_temp_c",          _k_to_c),
    "tmax":    ("air_temp_max_c",       _k_to_c),
    "tmin":    ("air_temp_min_c",       _k_to_c),
    "Tiw":     ("wet_bulb_c",           _k_to_c),
    # Humidity / pressure / visibility
    "r":       ("humidity_pct",         _identity),   # already %
    "MSL":     ("pressure_hpa",         _pa_to_hpa),
    "vis":     ("visibility_m",         _identity),
    # Wind (AROME has ws/wd directly; HIRLAM only u/v)
    "ws":      ("wind_speed_ms",        _identity),
    "wd":      ("wind_dir_deg",         _identity),
    "gust":    ("wind_gust_ms",         _identity),
    "u":       ("wind_u_ms",            _identity),   # intermediate
    "v":       ("wind_v_ms",            _identity),   # intermediate
    # Precipitation
    "prec1h":  ("precip_1h_mm",         _identity),
    "prec3h":  ("precip_3h_mm",         _identity),
    "prec12h": ("precip_12h_mm",        _identity),
    "prec24h": ("precip_24h_mm",        _identity),
    "prtype":  ("precip_type",          _identity),   # code 0-12, keep as int
    "prsort":  ("precip_sort",          _identity),   # code 0-6, keep as int
    "spp":     ("precip_frozen_pct",    _spp_fix),    # -9 = no precip
    "pmax":    ("precip_rate_max",      _identity),
    "pmean":   ("precip_rate_mean",     _identity),
    "pmedian": ("precip_rate_median",   _identity),
    "pmin":    ("precip_rate_min",      _identity),
    # Snow
    "frsn1h":  ("fresh_snow_1h_cm",     _identity),
    "frsn3h":  ("fresh_snow_3h_cm",     _identity),
    "frsn12h": ("fresh_snow_12h_cm",    _identity),
    "frsn24h": ("fresh_snow_24h_cm",    _identity),
    "sfgrd":   ("snowfall_gradient_m",  _identity),
    # Cloud
    "tcc":     ("cloud_total_frac",     _identity),
    "lcc":     ("cloud_low_frac",       _identity),
    "mcc":     ("cloud_med_frac",       _identity),
    "hcc":     ("cloud_high_frac",      _identity),
    "c_sigfr": ("cloud_sig_frac",       _identity),
    "cb_sig":  ("cloud_base_m",         _identity),
    "ct_sig":  ("cloud_top_m",          _identity),
}

# Live REST API parameter names (mesan2g v2 returns human-readable names)
LIVE_PARAM_MAP = {
    "air_temperature":                        "air_temp_c",
    "wind_speed":                             "wind_speed_ms",
    "wind_from_direction":                    "wind_dir_deg",
    "wind_speed_of_gust":                     "wind_gust_ms",
    "relative_humidity":                      "humidity_pct",
    "air_pressure_at_mean_sea_level":         "pressure_hpa",
    "visibility_in_air":                      "visibility_m",
    "wet_bulb_temperature":                   "wet_bulb_c",
    "precipitation_amount_last_1_hours":      "precip_1h_mm",
    "precipitation_amount_last_3_hours":      "precip_3h_mm",
    "precipitation_frozen_part":              "precip_frozen_pct",
    "precipitation_sort":                     "precip_sort",
    "predominant_precipitation_type_at_surface": "precip_type",
    "change_over_time_in_surface_snow_amount_24_hours": "fresh_snow_24h_cm",
    "cloud_area_fraction":                    "cloud_total_frac",
    "low_type_cloud_area_fraction":           "cloud_low_frac",
    "medium_type_cloud_area_fraction":        "cloud_med_frac",
    "high_type_cloud_area_fraction":          "cloud_high_frac",
    "symbol_code":                            "weather_symbol",  # 1-27
}


# ── Session ────────────────────────────────────────────────────────────────

def _get_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


# ── ATOM feed walker ───────────────────────────────────────────────────────

def get_file_url(
    year: int, month: int, day: int, hour: int = 6,
    pid: Optional[int] = None,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """
    Walk the SMHI ATOM feed to get the GRIB2 download URL for a specific
    date/time. Falls back to the direct URL pattern if ATOM walk fails.

    Direct pattern (works for known files):
        /data/{pid}/{YYYYMM}/MESAN_{YYYYMMDDHHMM}+000H00M
    """
    if pid is None:
        pid = get_product_id(year, month)

    yyyymm = f"{year:04d}{month:02d}"
    ts     = f"{year:04d}{month:02d}{day:02d}{hour:02d}00"
    prefix = {
        4: "MESAN",
        5: "AM25H2",
        6: "MESAN",
    }.get(pid, "MESAN")
    direct = f"{ARCHIVE_BASE}/data/{pid}/{yyyymm}/{prefix}_{ts}+000H00M"
    return direct


def verify_url(url: str, session: requests.Session) -> bool:
    """HEAD request to check if a file URL exists."""
    try:
        r = session.head(url, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


# ── GRIB download + parse ──────────────────────────────────────────────────

def download_grib(
    year: int, month: int, day: int, hour: int = 6,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
    delay: float = 0.5,
) -> Optional[Path]:
    """Download one MESAN GRIB2 file. Returns local path or None."""
    if session is None:
        session = _get_session()

    pid      = get_product_id(year, month)
    url      = get_file_url(year, month, day, hour, pid, session)
    out_path = GRIB_DIR / str(year) / Path(url).name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 10_000:
        return out_path

    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, timeout=120, stream=True)
            if r.status_code == 200:
                tmp = out_path.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                if tmp.stat().st_size < 1000:
                    tmp.unlink()
                    return None
                tmp.rename(out_path)
                logger.info(
                    f"  Downloaded {out_path.name} "
                    f"({out_path.stat().st_size/1e6:.1f} MB, product={pid})"
                )
                return out_path
            elif r.status_code == 404:
                return None
            else:
                logger.warning(f"  HTTP {r.status_code} attempt {attempt}: {out_path.name}")
        except Exception as exc:
            logger.warning(f"  Error attempt {attempt}: {exc}")
        if attempt < max_retries:
            time.sleep(delay * attempt)

    return None


def parse_grib_edition1(grib_path: Path) -> Optional[object]:
    """
    Parse a product-5 AM25H2 file (GRIB edition 1, SMHI local table 253).

    cfgrib cannot decode table 253 — all shortNames come back as 'unknown'.
    This function reads raw messages via eccodes and uses a hardcoded
    (indicatorOfParameter, indicatorOfTypeOfLevel, level, timeRangeIndicator)
    lookup table derived from inspecting real AM25H2 files.

    Available in product 5 (fewer params than product 6):
      air_temp_c, wet_bulb_c (skin), wind_u/v → wind_speed_ms, wind_gust_ms,
      precip_1h_mm, pressure_hpa, cloud_total/low/med/high_frac, fresh_snow_1h_cm
    """
    try:
        import eccodes
        import xarray as xr
    except ImportError:
        logger.error("eccodes not installed — run pip install eccodes")
        return None

    # (iop, ltype, level, tri) → (output_column, conversion_fn)
    # Verified from AM25H2_202501010600+000H00M (table=253, edition=1)
    GRIB1_TABLE253: Dict[tuple, tuple] = {
        (11,  "sfc", 2,   0): ("air_temp_c",       _k_to_c),
        (11,  "sfc", 0,   0): ("wet_bulb_c",        _k_to_c),    # skin temp, wet-bulb proxy
        (33,  "sfc", 10,  0): ("wind_u_ms",          _identity),  # intermediate
        (34,  "sfc", 10,  0): ("wind_v_ms",          _identity),  # intermediate
        (228, "sfc", 10,  2): ("wind_gust_ms",        _identity),
        (61,  "sfc", 0,   4): ("precip_1h_mm",        _identity),
        (1,   "103", 0,   0): ("pressure_hpa",        _pa_to_hpa),
        (71,  "sfc", 0,   0): ("cloud_total_frac",   _identity),
        (73,  "sfc", 0,   0): ("cloud_low_frac",     _identity),
        (74,  "sfc", 0,   0): ("cloud_med_frac",     _identity),
        (75,  "sfc", 0,   0): ("cloud_high_frac",    _identity),
        (185, "sfc", 0,   4): ("fresh_snow_1h_cm",   _identity),
    }

    nj = ni = None
    lat2d = lon2d = None
    merged: Dict[str, np.ndarray] = {}

    try:
        with open(grib_path, "rb") as f:
            while True:
                h = eccodes.codes_grib_new_from_file(f)
                if h is None:
                    break
                try:
                    iop   = int(eccodes.codes_get(h, "indicatorOfParameter"))
                    ltype = str(eccodes.codes_get(h, "indicatorOfTypeOfLevel"))
                    level = int(eccodes.codes_get(h, "level"))
                    tri   = int(eccodes.codes_get(h, "timeRangeIndicator"))
                    key   = (iop, ltype, level, tri)
                    if key not in GRIB1_TABLE253:
                        continue
                    col_name, conv_fn = GRIB1_TABLE253[key]
                    if col_name in merged:
                        continue  # already have this variable

                    if nj is None:
                        nj = eccodes.codes_get(h, "Nj")
                        ni = eccodes.codes_get(h, "Ni")
                        raw_lats = eccodes.codes_get_array(h, "latitudes").reshape(nj, ni)
                        raw_lons = eccodes.codes_get_array(h, "longitudes").reshape(nj, ni)
                        # eccodes returns 0-360; convert to -180..+180
                        lat2d = raw_lats.astype(np.float32)
                        lon2d = np.where(raw_lons > 180, raw_lons - 360, raw_lons).astype(np.float32)

                    vals = eccodes.codes_get_values(h).reshape(nj, ni).astype(np.float32)
                    merged[col_name] = conv_fn(vals)
                except Exception as exc:
                    logger.debug(f"  edition1 msg skip in {grib_path.name}: {exc}")
                finally:
                    eccodes.codes_release(h)
    except Exception as exc:
        logger.warning(f"  edition1 parse failed {grib_path.name}: {exc}")
        return None

    if not merged:
        logger.warning(f"  {grib_path.name}: no table-253 params matched (edition1 path)")
        return None

    # Derive wind_speed from u/v
    if "wind_speed_ms" not in merged and "wind_u_ms" in merged and "wind_v_ms" in merged:
        merged["wind_speed_ms"] = np.hypot(merged.pop("wind_u_ms"), merged.pop("wind_v_ms")).astype(np.float32)
    else:
        merged.pop("wind_u_ms", None)
        merged.pop("wind_v_ms", None)

    data_vars = {col: (["y", "x"], arr) for col, arr in merged.items()}
    coords: Dict[str, object] = {}
    if lat2d is not None:
        coords["latitude"]  = (["y", "x"], lat2d)
        coords["longitude"] = (["y", "x"], lon2d)

    return xr.Dataset(data_vars, coords=coords)


def parse_grib(grib_path: Path) -> Optional[object]:
    """
    Parse a MESAN GRIB file into an xarray Dataset.

    Dispatches to parse_grib_edition1 for product-5 AM25H2 files (GRIB edition 1,
    SMHI local table 253). Falls back to cfgrib for edition-2 AROME/HIRLAM files.
    Applies all unit conversions. Derives wind_speed from u/v if ws absent.
    Returns None on parse failure.
    """
    # Detect GRIB edition to choose the right parser
    try:
        import eccodes as _ec
        with open(grib_path, "rb") as _f:
            _h = _ec.codes_grib_new_from_file(_f)
            if _h is not None:
                _edition = _ec.codes_get(_h, "editionNumber")
                _ec.codes_release(_h)
                if _edition == 1:
                    return parse_grib_edition1(grib_path)
    except Exception:
        pass  # fall through to cfgrib

    try:
        import cfgrib
        import xarray as xr
    except ImportError:
        logger.error("cfgrib/xarray not installed")
        return None

    try:
        import warnings as _w
        _w.filterwarnings("ignore", category=FutureWarning)
        # indexpath="" disables index-file caching → no "older than GRIB file" noise
        raw_datasets = cfgrib.open_datasets(str(grib_path), indexpath="")
        logger.debug(f"  cfgrib opened {len(raw_datasets)} datasets from {grib_path.name}")
    except Exception as exc:
        logger.warning(f"  cfgrib failed {grib_path.name}: {exc}")
        return None

    merged: Dict[str, object] = {}
    for i, ds in enumerate(raw_datasets):
        logger.debug(f"    dataset {i}: vars={list(ds.data_vars.keys())}")
        for var in ds.data_vars:
            sn  = ds[var].attrs.get("GRIB_shortName", var)
            cfg = GRIB_PARAMS.get(sn)
            if cfg is None:
                logger.debug(f"      {var} (shortName={sn}): not in GRIB_PARAMS, skipping")
                continue
            col_name, conv_fn = cfg
            try:
                arr = conv_fn(ds[var].values.astype(np.float32))
                merged[col_name] = (ds[var].dims, arr, ds[var].attrs)
                logger.debug(f"      {var} ({sn}) -> {col_name}: OK")
            except Exception as exc:
                logger.debug(f"      {var} ({sn}): conversion failed: {exc}")
                continue

    if not merged:
        logger.warning(f"  {grib_path.name}: no parameters matched in any dataset")
        return None

    # Derive wind speed from u/v if ws not available
    if "wind_speed_ms" not in merged and "wind_u_ms" in merged and "wind_v_ms" in merged:
        u = merged["wind_u_ms"][1]
        v = merged["wind_v_ms"][1]
        merged["wind_speed_ms"] = (merged["wind_u_ms"][0], np.hypot(u, v).astype(np.float32), {})

    merged.pop("wind_u_ms", None)
    merged.pop("wind_v_ms", None)

    # Build xarray Dataset — drop conflicting level coordinates
    import xarray as xr
    drop_coords = {"heightAboveGround", "level", "pressure", "depthBelowLandLayer"}
    das = []
    for col, (dims, arr, attrs) in merged.items():
        # Get coordinates from first raw dataset that has this variable
        ref_ds = next(
            (ds for ds in raw_datasets
             if any(ds[v].attrs.get("GRIB_shortName") == next(
                 (k for k, (c, _) in GRIB_PARAMS.items() if c == col), None
             ) for v in ds.data_vars)),
            raw_datasets[0],
        )
        coords = {k: v for k, v in ref_ds.coords.items() if k not in drop_coords}
        da = xr.DataArray(arr, dims=dims, attrs=attrs, name=col)
        for cname in list(da.coords):
            if cname in drop_coords:
                da = da.drop_vars(cname)
        das.append(da)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_out = xr.merge(das, compat="override")

    return ds_out


def _save_year_zarr(year, year_datasets, year_dates, zarr_path, save_zarr):
    if not save_zarr:
        return
    if not year_datasets:
        logger.warning(f"  {year}: no datasets to aggregate (parse_grib returned None for all files)")
        return
    try:
        import xarray as xr
        from collections import Counter

        # 1. Deduplicate by date — keep the last download for any repeated date
        seen: dict = {}
        for ds, d in zip(year_datasets, year_dates):
            seen[d] = ds
        dedup_dates = sorted(seen.keys())
        dedup_datasets = [seen[d] for d in dedup_dates]
        if len(dedup_datasets) < len(year_datasets):
            logger.info(
                f"  {year}: deduped {len(year_datasets)} → {len(dedup_datasets)} datasets"
            )

        # 2. Filter to dominant (y, x) grid shape — handles mid-year grid-version changes
        def _shape(ds):
            return (ds.dims.get("y", 0), ds.dims.get("x", 0))

        shapes = [_shape(ds) for ds in dedup_datasets]
        dominant = Counter(shapes).most_common(1)[0][0]
        keep = [(ds, d) for ds, d, s in zip(dedup_datasets, dedup_dates, shapes)
                if s == dominant]
        dropped_dates = [d for ds, d, s in zip(dedup_datasets, dedup_dates, shapes)
                         if s != dominant]
        if dropped_dates:
            logger.warning(
                f"  {year}: dropped {len(dropped_dates)} datasets with non-dominant grid "
                f"(dominant={dominant}, other shapes={set(shapes)-{dominant}}): "
                + ", ".join(str(d) for d in dropped_dates)
            )
            dedup_datasets, dedup_dates = zip(*keep)

        # 3. Strip scalar time/valid_time/step coords that cfgrib attaches to each
        #    dataset — they conflict with the new time dim added by xr.concat.
        #    Also squeeze any existing time dim (e.g. multi-step GRIB messages).
        cleaned = []
        for ds in dedup_datasets:
            drop = [c for c in ds.coords if c in ("time", "valid_time", "step")]
            if drop:
                ds = ds.drop_vars(drop, errors="ignore")
            if "time" in ds.dims:
                ds = ds.isel(time=0, drop=True)
            cleaned.append(ds)

        time_index = pd.DatetimeIndex([pd.Timestamp(d) for d in dedup_dates], name="time")
        combined = xr.concat(cleaned, dim=time_index)
        combined.to_zarr(str(zarr_path), mode="w")
        logger.info(f"Saved {year} zarr: {len(cleaned)} days -> {zarr_path}")
    except Exception as exc:
        logger.error(f"Failed to save zarr for {year}: {exc}")


# ── Bulk download mode ─────────────────────────────────────────────────────

def run_bulk(
    start_year: int = 2000,
    end_year:   int = 2025,
    hour:       int = 6,
    max_files:  Optional[int] = None,
    delay:      float = 0.5,
    save_zarr:  bool = True,
    delete_grib: bool = False,
) -> Dict:
    """
    Download all MESAN GRIB2 files for the date range, parse, and save as zarr.

    One file per day at the specified UTC hour.
    Product 4 (HIRLAM) for 2000-2013, Product 6 (AROME) for 2014-2024,
    Product 5 (historical AROME) for 2025-2026.
    """
    GRIB_DIR.mkdir(parents=True, exist_ok=True)
    ZARR_DIR.mkdir(parents=True, exist_ok=True)

    # Remove any stale cfgrib index files left from previous runs.
    # We use indexpath="" so no new ones are created, but old ones can linger.
    idx_files = list(GRIB_DIR.rglob("*.idx"))
    if idx_files:
        for f in idx_files:
            f.unlink(missing_ok=True)
        logger.info(f"Removed {len(idx_files)} stale .idx files from {GRIB_DIR}")

    if save_zarr:
        try:
            import zarr  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "zarr is not installed — run `pip install zarr` before starting. "
                "Without zarr, GRIB files will be deleted but nothing saved."
            )

    session = _get_session()
    stats   = {"downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0}
    count   = 0

    for year in range(start_year, end_year + 1):
        year_datasets = []
        year_dates    = []

        zarr_path = ZARR_DIR / f"mesan_{year}.zarr"
        if zarr_path.exists():
            logger.info(f"  {year}: zarr exists, skipping.")
            stats["skipped"] += 1
            continue

        for month in range(1, 13):
            if month == 12:
                days = (datetime.date(year + 1, 1, 1) - datetime.date(year, month, 1)).days
            else:
                days = (datetime.date(year, month + 1, 1) - datetime.date(year, month, 1)).days

            for day in range(1, days + 1):
                if max_files and count >= max_files:
                    logger.info(f"Reached max_files={max_files}, stopping.")
                    _save_year_zarr(year, year_datasets, year_dates, zarr_path, save_zarr)
                    return stats

                grib_path = download_grib(year, month, day, hour, session, delay=delay)
                count += 1

                if grib_path is None:
                    stats["failed"] += 1
                    continue

                stats["downloaded"] += 1
                stats["total_mb"]   += grib_path.stat().st_size / 1e6

                if save_zarr:
                    ds = parse_grib(grib_path)
                    if ds is not None:
                        year_datasets.append(ds)
                        year_dates.append(datetime.date(year, month, day))
                    else:
                        logger.debug(f"    {grib_path.name}: parse_grib returned None")

                if delete_grib and grib_path.exists():
                    grib_path.unlink()
                    grib_path.with_suffix(grib_path.suffix + ".idx").unlink(missing_ok=True)

                time.sleep(delay)

        logger.info(f"  {year}: accumulated {len(year_datasets)}/{stats['downloaded']} datasets from downloads")
        _save_year_zarr(year, year_datasets, year_dates, zarr_path, save_zarr)

    logger.info(
        f"Done — downloaded={stats['downloaded']}, "
        f"skipped={stats['skipped']}, failed={stats['failed']}, "
        f"total={stats['total_mb']:.0f} MB"
    )
    return stats


# ── Live point mode ────────────────────────────────────────────────────────

def fetch_live_point(lon: float, lat: float, session: requests.Session) -> Optional[Dict]:
    """Query MESAN2g REST API for a single point (last 24h). Returns latest record."""
    url = f"{LIVE_BASE}/geotype/point/lon/{lon:.4f}/lat/{lat:.4f}/data.json"
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        ts = r.json().get("timeSeries", [])
        if not ts:
            return None
        latest = ts[0]
        row = {"time": latest["time"]}
        for k, v in latest.get("data", {}).items():
            col = LIVE_PARAM_MAP.get(k)
            if col:
                val = v if col not in ("precip_type", "precip_sort", "weather_symbol") else v
                if col == "precip_frozen_pct" and v == -9:
                    val = None
                row[col] = val
        return row
    except Exception as exc:
        logger.debug(f"Live query failed ({lon:.2f},{lat:.2f}): {exc}")
        return None


def run_live(stations_csv: Optional[Path] = None) -> pd.DataFrame:
    """Query live MESAN for all WQ station locations (last 24h)."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    session = _get_session()

    candidates = [
        RAW_DIR / "hydroobs"   / "smhi_hydroobs_stations.csv",
        RAW_DIR / "slu_mvm"    / "slu_mvm_stations.csv",
    ]
    if stations_csv:
        candidates = [stations_csv]

    frames = []
    for p in candidates:
        if p.exists():
            frames.append(pd.read_csv(p, usecols=["station_id", "latitude", "longitude"]))

    if not frames:
        logger.error("No station CSV found.")
        return pd.DataFrame()

    stations = pd.concat(frames).drop_duplicates(["latitude", "longitude"]).dropna()
    logger.info(f"Querying MESAN live for {len(stations)} station locations...")

    rows = []
    for _, st in stations.iterrows():
        rec = fetch_live_point(st["longitude"], st["latitude"], session)
        if rec:
            rec.update({"station_id": st["station_id"],
                        "latitude": st["latitude"], "longitude": st["longitude"]})
            rows.append(rec)
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    today = datetime.date.today().isoformat()
    out   = LIVE_DIR / f"mesan_live_{today}.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Saved {len(df)} live records -> {out}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MESAN extractor (HIRLAM 2000-2013, AROME 2014-2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 2000-2025 extraction (one file/day, 06 UTC)
  python -m src.extraction.fetch_mesan --start-year 2000 --end-year 2025

  # Quick test: 5 files from 2023
  python -m src.extraction.fetch_mesan --start-year 2023 --end-year 2023 --max-files 5

  # Live point queries for all WQ stations
  python -m src.extraction.fetch_mesan --mode live
        """,
    )
    parser.add_argument("--mode",        choices=["bulk", "live"], default="bulk")
    parser.add_argument("--start-year",  type=int,   default=2000)
    parser.add_argument("--end-year",    type=int,   default=2025)
    parser.add_argument("--hour",        type=int,   default=6)
    parser.add_argument("--max-files",   type=int,   default=None)
    parser.add_argument("--delay",       type=float, default=0.5)
    parser.add_argument("--no-zarr",     action="store_true")
    parser.add_argument("--delete-grib", action="store_true",
                        help="Delete GRIB after parsing to save disk")
    args = parser.parse_args()

    if args.mode == "live":
        run_live()
    else:
        run_bulk(
            start_year=args.start_year,
            end_year=args.end_year,
            hour=args.hour,
            max_files=args.max_files,
            delay=args.delay,
            save_zarr=not args.no_zarr,
            delete_grib=args.delete_grib,
        )


if __name__ == "__main__":
    main()
