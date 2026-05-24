"""
Shared utilities for src/patches/* WQ satellite patch extraction scripts.

Distinct from the snow-cover patch scripts in scripts/ (download_modis_patches.py,
download_era5_land_patches.py), which target MOD10A1 NDSI and ERA5-Land snow vars.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── GEE bootstrap ─────────────────────────────────────────────────────────────

def check_ee():
    try:
        import ee
        return ee
    except ImportError:
        logger.error(
            "earthengine-api not installed.\n"
            "  pip install earthengine-api\n"
            "  earthengine authenticate   (one-time, opens browser)"
        )
        sys.exit(1)


# ── station loading ───────────────────────────────────────────────────────────

def load_wq_stations(
    slu_file: Path,
    hydobs_file: Optional[Path] = None,
    fmi_file: Optional[Path] = None,
    pilot_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load WQ monitoring stations.

    Primary source: SLU MVM chemistry stations (Sweden).
    Optional:       SMHI HydObs hydrological stations + FMI Finnish stations.

    Returns DataFrame with columns [station_id, latitude, longitude, source].
    """
    frames = []
    for path, source in [
        (slu_file,    'SLU_MVM'),
        (hydobs_file, 'SMHI_HydObs'),
        (fmi_file,    'FMI'),
    ]:
        if path is None:
            continue
        if path.exists():
            df = pd.read_csv(path, usecols=['station_id', 'latitude', 'longitude'])
            df['source'] = source
            frames.append(df)
            logger.info(f"Loaded {len(df):,} stations from {path.name}")
        else:
            logger.warning(f"Station file not found: {path}")

    if not frames:
        logger.error("No station metadata found — aborting.")
        sys.exit(1)

    stations = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=['latitude', 'longitude'])
        .drop_duplicates(subset=['station_id'])
    )
    stations['station_id'] = stations['station_id'].astype(str)
    stations = stations.sort_values('station_id').reset_index(drop=True)

    if pilot_n:
        stations = stations.head(pilot_n).reset_index(drop=True)
        logger.info(f"PILOT MODE: {len(stations)} stations")

    counts = stations['source'].value_counts().to_dict()
    logger.info(f"Total: {len(stations):,} stations — {counts}")
    return stations


# ── time helpers ──────────────────────────────────────────────────────────────

def build_year_month_list(start_year: int, end_year: int) -> List[Tuple[int, int]]:
    """All 12 months for every year in [start_year, end_year]."""
    return [(y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)]


# ── geometry ──────────────────────────────────────────────────────────────────

def station_rect(ee, lat: float, lon: float, half_m: float):
    """Square bounding box centred on (lat, lon) with side = 2 × half_m metres."""
    return ee.Geometry.Point(lon, lat).buffer(half_m).bounds()


# ── array helpers ─────────────────────────────────────────────────────────────

def crop_pad(arr: np.ndarray, size: int, fill: float = np.nan) -> np.ndarray:
    """Centre-crop a 2-D array to size × size, padding with fill if smaller."""
    h, w = arr.shape
    out  = np.full((size, size), fill, dtype=np.float32)
    H, W = min(h, size), min(w, size)
    h0, w0 = (h - H) // 2, (w - W) // 2
    dh, dw = (size - H) // 2, (size - W) // 2
    out[dh:dh + H, dw:dw + W] = arr[h0:h0 + H, w0:w0 + W]
    return out


def resize_to_canonical(arr: np.ndarray, target: int = 32) -> np.ndarray:
    """
    Bilinear resize of a (H, W) or (H, W, C) float32 array to (target, target[, C]).

    Uses scipy.ndimage.zoom when available; falls back to nearest-neighbour.
    NaN values are preserved by interpolating on a filled copy and restoring NaN
    wherever the original (nearest-neighbour) lookup was NaN.
    """
    h, w = arr.shape[:2]
    if h == target and w == target:
        return arr.astype(np.float32)

    try:
        from scipy.ndimage import zoom
        if arr.ndim == 2:
            return zoom(arr, (target / h, target / w), order=1).astype(np.float32)
        return zoom(arr, (target / h, target / w, 1), order=1).astype(np.float32)
    except ImportError:
        ys = (np.arange(target) * h / target).astype(int)
        xs = (np.arange(target) * w / target).astype(int)
        if arr.ndim == 2:
            return arr[np.ix_(ys, xs)].astype(np.float32)
        return arr[ys[:, None], xs[None, :], :].astype(np.float32)


# ── I/O ───────────────────────────────────────────────────────────────────────

def shard_exists(shard_dir: Path, station_id: str, year: int, month: int, suffix: str) -> bool:
    return (shard_dir / station_id / f"{year}-{month:02d}_{suffix}.npz").exists()


def annual_shard_exists(shard_dir: Path, station_id: str, year: int, suffix: str) -> bool:
    return (shard_dir / station_id / f"{year}_{suffix}.npz").exists()


def save_annual_shard(
    shard_dir: Path,
    station_id: str,
    lat: float,
    lon: float,
    year: int,
    suffix: str,
    patch: np.ndarray,
    mask: np.ndarray,
    dates: list,
    channels: list,
    footprint_m: float,
    **extra,
) -> Path:
    """Write one annual NPZ shard: data/processed/{sensor}_wq_patches/<sid>/<YYYY>_<SUFFIX>.npz"""
    station_dir = shard_dir / station_id
    station_dir.mkdir(parents=True, exist_ok=True)
    out = station_dir / f"{year}_{suffix}.npz"
    np.savez_compressed(
        out,
        patch=patch,
        mask=mask,
        dates=np.array(dates, dtype='S10'),
        channels=np.array(channels, dtype=object),
        station_id=station_id,
        latitude=np.float64(lat),
        longitude=np.float64(lon),
        footprint_m=np.float64(footprint_m),
        **extra,
    )
    return out


def save_shard(
    shard_dir: Path,
    station_id: str,
    lat: float,
    lon: float,
    year: int,
    month: int,
    suffix: str,
    patch: np.ndarray,
    mask: np.ndarray,
    dates: list,
    channels: list,
    footprint_m: float,
    **extra,
) -> Path:
    """
    Write one NPZ shard.  Layout (for all WQ patch scripts):

        patch       (n_obs, 32, 32, n_bands)  float32  canonical 32×32 tensor
        mask        (n_obs, 32, 32)            uint8    1 = valid, 0 = cloud/fill
        dates       (n_obs,)                   |S10     ISO YYYY-MM-DD
        channels    (n_bands,)                 object   band labels
        station_id, latitude, longitude, footprint_m   scalars

    Optional extra keys (e.g. scene_ids, patch_native) passed via **extra.
    """
    station_dir = shard_dir / station_id
    station_dir.mkdir(parents=True, exist_ok=True)
    out = station_dir / f"{year}-{month:02d}_{suffix}.npz"
    np.savez_compressed(
        out,
        patch=patch,
        mask=mask,
        dates=np.array(dates, dtype='S10'),
        channels=np.array(channels, dtype=object),
        station_id=station_id,
        latitude=np.float64(lat),
        longitude=np.float64(lon),
        footprint_m=np.float64(footprint_m),
        **extra,
    )
    return out
