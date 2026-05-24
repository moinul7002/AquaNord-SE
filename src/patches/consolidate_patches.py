#!/usr/bin/env python3
"""
Consolidate per-station NPZ patch shards into sensor-level HDF5 files.

MD-5: Per-station NPZ shards are the raw extraction format (resumable,
per-station checkpointing).  This script merges them into one HDF5 file per
sensor for efficient PyTorch/JAX dataloading.

Why HDF5 over zarr:
  zarr v3 uses atomic writes (.partial → rename) that fail intermittently on
  Windows when Windows Defender holds a file lock during scanning.  HDF5/h5py
  handles writes internally and is reliable on all platforms.

Output layout (one .h5 per sensor):
    data/processed/patches_hdf5/{sensor}.h5
        /patch       (n_stations, n_days, H, W, n_bands)  float32  gzip-4
        /mask        (n_stations, n_days, H, W)            uint8    gzip-4
        /dates       (n_days,)                             bytes    ISO YYYY-MM-DD
        /channels    (n_bands,)                            bytes    band labels
        /station_ids (n_stations,)                         bytes    station IDs
        attrs: sensor, footprint_m, spatial_shape, n_stations, n_days, n_bands,
               start_date, end_date

Time axis covers all calendar days from start_date to end_date.
Missing days (no observation or extraction gap) are NaN patch / 0 mask.

Usage:
    python -m src.patches.consolidate_patches --sensor era5_wq
    python -m src.patches.consolidate_patches --sensor all
    python -m src.patches.consolidate_patches --sensor modis_wq --pilot 10
"""

import argparse
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ── sensor registry ────────────────────────────────────────────────────────────

SENSORS = {
    "modis_wq": {
        "shard_dirs": [("data/processed/modis_wq_patches", ["MOD", "MYD"])],
        "start_date": date(2000, 1, 1),
        "end_date":   date(2025, 12, 31),
    },
    "landsat_wq": {
        "shard_dirs": [("data/processed/landsat_wq_patches", ["L7", "L8", "L9"])],
        "start_date": date(2000, 1, 1),
        "end_date":   date(2025, 12, 31),
    },
    "sentinel1_wq": {
        "shard_dirs": [("data/processed/sentinel1_wq_patches", ["S1"])],
        "start_date": date(2014, 4, 1),
        "end_date":   date(2025, 12, 31),
    },
    "sentinel2_wq": {
        "shard_dirs": [("data/processed/sentinel2_wq_patches", ["S2"])],
        "start_date": date(2015, 6, 23),
        "end_date":   date(2025, 12, 31),
    },
    "era5_wq": {
        "shard_dirs": [("data/processed/era5_wq_patches", [""])],  # <YYYY>.npz — no suffix
        "start_date": date(2000, 1, 1),
        "end_date":   date(2025, 12, 31),
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_h5py():
    try:
        import h5py
        return h5py
    except ImportError:
        logger.error("h5py not installed.  Run: pip install h5py")
        raise


def _build_date_index(start: date, end: date):
    days, d = [], start
    while d <= end:
        days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days


def _shard_glob(station_dir: Path, sfx: str):
    return sorted(station_dir.glob(f"*_{sfx}.npz" if sfx else "*.npz"))


def _collect_station_ids(shard_dir: Path) -> list:
    return sorted(p.name for p in shard_dir.iterdir() if p.is_dir())


# ── consolidation ─────────────────────────────────────────────────────────────

def consolidate_sensor(
    sensor_name: str,
    out_dir: Path,
    pilot_n: Optional[int] = None,
    country: Optional[str] = None,
) -> None:
    h5py = _check_h5py()
    cfg  = SENSORS[sensor_name]

    all_dates  = _build_date_index(cfg["start_date"], cfg["end_date"])
    date_index = {d: i for i, d in enumerate(all_dates)}
    n_days     = len(all_dates)

    # Collect station IDs
    station_ids, shard_paths = set(), []
    for raw_dir_str, suffixes in cfg["shard_dirs"]:
        shard_dir = Path(raw_dir_str)
        if not shard_dir.exists():
            logger.warning(f"Shard dir not found: {shard_dir}")
            continue
        station_ids.update(_collect_station_ids(shard_dir))
        shard_paths.append((shard_dir, suffixes))

    station_ids = sorted(station_ids)
    if pilot_n:
        station_ids = station_ids[:pilot_n]
        logger.info(f"PILOT MODE: {len(station_ids)} stations")

    if not station_ids:
        logger.error(f"No station shards found for {sensor_name}")
        return

    # Discover spatial shape and channels from first shard
    H = W = n_bands = None
    channels = []
    footprint_m = 0.0
    for sid in station_ids:
        for shard_dir, suffixes in shard_paths:
            for sfx in suffixes:
                hits = _shard_glob(shard_dir / sid, sfx)
                if hits:
                    z = np.load(hits[0], allow_pickle=True)
                    patch = z["patch"]
                    H, W, n_bands = patch.shape[-3], patch.shape[-2], patch.shape[-1]
                    channels    = [str(c) for c in z["channels"].tolist()]
                    footprint_m = float(z.get("footprint_m", 0))
                    break
            if H is not None:
                break
        if H is not None:
            break

    if H is None:
        logger.error(f"Could not read shape from any shard for {sensor_name}")
        return

    n_stations = len(station_ids)
    logger.info(
        f"Consolidating {sensor_name}: {n_stations:,} stations · "
        f"{n_days:,} days · {H}×{W} · {n_bands} bands"
    )

    out_path = out_dir / f"{sensor_name}.h5"
    tmp_path = out_dir / f"{sensor_name}_writing.h5"

    # Remove stale tmp file if present
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except PermissionError:
            pass

    chunk_t = min(365, n_days)

    with h5py.File(tmp_path, "w") as f:

        # float16 halves storage vs float32; shuffle reorders bytes before gzip
        # (large gains on float arrays with repeated/NaN regions); gzip-6 for
        # a small extra ratio hit at acceptable write speed.
        _PATCH_FILL = np.float16(np.nan)
        patch_ds = f.create_dataset(
            "patch",
            shape=(n_stations, n_days, H, W, n_bands),
            chunks=(1, chunk_t, H, W, n_bands),
            dtype=np.float16,
            fillvalue=_PATCH_FILL,
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        mask_ds = f.create_dataset(
            "mask",
            shape=(n_stations, n_days, H, W),
            chunks=(1, chunk_t, H, W),
            dtype=np.uint8,
            fillvalue=0,
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )

        # Metadata arrays — store as fixed-length byte strings
        str_dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("dates",       data=np.array(all_dates,   dtype="S10"))
        f.create_dataset("channels",    data=np.array(channels,    dtype="S64"),  dtype=str_dt)
        f.create_dataset("station_ids", data=np.array(station_ids, dtype="S32"),  dtype=str_dt)

        # Attributes
        f.attrs["sensor"]        = sensor_name
        f.attrs["footprint_m"]   = footprint_m
        f.attrs["spatial_shape"] = [H, W]
        f.attrs["n_stations"]    = n_stations
        f.attrs["n_days"]        = n_days
        f.attrs["n_bands"]       = n_bands
        f.attrs["start_date"]    = cfg["start_date"].isoformat()
        f.attrs["end_date"]      = cfg["end_date"].isoformat()
        f.attrs["patch_dtype"]   = "float16"

        # Fill data station by station
        for s_idx, sid in enumerate(station_ids):
            if s_idx % 500 == 0:
                logger.info(f"  [{s_idx}/{n_stations}] station {sid}")

            for shard_dir, suffixes in shard_paths:
                station_dir = shard_dir / sid
                if not station_dir.exists():
                    continue
                for sfx in suffixes:
                    for npz_path in _shard_glob(station_dir, sfx):
                        try:
                            z         = np.load(npz_path, allow_pickle=True)
                            raw_patch = z["patch"]
                            raw_mask  = z["mask"]
                            raw_dates = [
                                d.decode() if isinstance(d, bytes) else str(d)
                                for d in z["dates"]
                            ]

                            # (n_obs, H, W, C) — handle single-obs case
                            if raw_patch.ndim == 3:
                                raw_patch = raw_patch[np.newaxis]
                                raw_mask  = raw_mask[np.newaxis]

                            for obs_i, d_str in enumerate(raw_dates):
                                t_idx = date_index.get(d_str)
                                if t_idx is None:
                                    continue
                                patch_ds[s_idx, t_idx] = raw_patch[obs_i].astype(np.float16)
                                mask_ds[s_idx, t_idx]  = raw_mask[obs_i]

                        except Exception as exc:
                            logger.warning(f"  Skipped {npz_path.name}: {exc}")

    # Atomic rename: replace final file only when write is fully complete
    if out_path.exists():
        for attempt in range(5):
            try:
                out_path.unlink()
                break
            except PermissionError:
                logger.warning(f"  Waiting for file lock to release ({attempt+1}/5)...")
                time.sleep(3)
    os.replace(tmp_path, out_path)

    size_gb = out_path.stat().st_size / 1e9
    logger.info(f"Wrote {out_path}  ({n_stations:,} stations, {n_days:,} days, {size_gb:.2f} GB)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Consolidate WQ patch NPZ shards into sensor-level HDF5 files (MD-5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sensor",  default="all",
                   help=f"Sensor: {list(SENSORS)} or 'all'")
    p.add_argument("--out-dir", default="data/processed/patches_hdf5")
    p.add_argument("--pilot",   type=int, default=None,
                   help="Limit to first N stations (testing)")
    p.add_argument("--country", default=None, help="SE / FI (informational only)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(SENSORS) if args.sensor == "all" else [args.sensor]
    for sensor_name in targets:
        if sensor_name not in SENSORS:
            logger.error(f"Unknown sensor '{sensor_name}'.  Choose from: {list(SENSORS)}")
            continue
        consolidate_sensor(sensor_name, out_dir, pilot_n=args.pilot, country=args.country)

    logger.info("Done.")


if __name__ == "__main__":
    main()
