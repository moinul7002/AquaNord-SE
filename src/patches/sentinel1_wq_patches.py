#!/usr/bin/env python3
"""
Sentinel-1 C-band SAR patches for WQ monitoring stations via GEE.

Sensor:   Sentinel-1 IW GRD  (COPERNICUS/S1_GRD)  2014–2025
Bands:    VV  vertical-vertical backscatter  (dB)
          VH  vertical-horizontal backscatter (dB)
Footprint: 10 km × 10 km  (half_m = 5 000 m)
Scale:     20 m for sampleRectangle  →  native ~ 500 × 500 px
           (10 m native exceeds sampleRectangle pixel limit per band)
Canonical: 32 × 32  (bilinear resize from 500 × 500)

SAR is cloud-penetrating — no cloud mask applied.
Ascending and descending passes are stored in the same monthly shard.
Pass direction (ASC / DES) is recorded in the scene_passes array.

Output:
    data/processed/sentinel1_wq_patches/<station_id>/<YYYY-MM>_S1.npz
        patch        (n_scenes, 32, 32, 2)  float32   [VV, VH] in dB
        mask         (n_scenes, 32, 32)     uint8     1 = finite dB value
        dates        (n_scenes,)            |S10
        scene_passes (n_scenes,)            |S3       'ASC' or 'DES'
        channels     (2,)                   object    ['vv_db', 'vh_db']
        station_id, latitude, longitude, footprint_m

Usage:
    python -m src.patches.sentinel1_wq_patches \\
        --start-year 2014 --end-year 2025 --gee-project <id>
"""

import argparse
import calendar
import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.patches._utils import (
    build_year_month_list, check_ee, load_wq_stations,
    resize_to_canonical, save_annual_shard, annual_shard_exists, station_rect,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_COLLECTION  = 'COPERNICUS/S1_GRD'
_BANDS       = ['VV', 'VH']
_CHANNELS    = ['vv_db', 'vh_db']
_DB_MIN      = -40.0   # plausible SAR dB range
_DB_MAX      =  5.0

PATCH_HALF_M = 5_000.0
GEE_SCALE_M  = 20       # sampleRectangle at 20 m (10 m native → 500×500/band exceeds limit)
CANONICAL    = 32
SUFFIX       = 'S1'


# ── extraction ────────────────────────────────────────────────────────────────

def extract_month(
    ee, lat: float, lon: float,
    year: int, month: int,
    max_retries: int = 3,
) -> list:
    """
    Pull all S1 IW GRD VV+VH scenes for one station-month.

    Returns a list of scene dicts: {date, pass_dir, patch (32,32,2), mask (32,32)}.
    Ascending and descending passes are stored separately within the list.
    """
    rect = station_rect(ee, lat, lon, PATCH_HALF_M)
    _, n_days = calendar.monthrange(year, month)

    coll = (ee.ImageCollection(_COLLECTION)
            .filterDate(f"{year}-{month:02d}-01",
                        f"{year}-{month:02d}-{n_days:02d}")
            .filterBounds(rect)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .select(_BANDS))

    try:
        n_imgs = coll.size().getInfo()
    except Exception:
        return []
    if n_imgs == 0:
        return []

    scenes   = []
    img_list = coll.toList(n_imgs)

    for idx in range(n_imgs):
        img      = ee.Image(img_list.get(idx))
        date_iso = img.date().format('YYYY-MM-dd').getInfo()
        try:
            pass_dir = img.get('orbitProperties_pass').getInfo() or 'UNKNOWN'
            pass_dir = 'ASC' if 'ASC' in pass_dir.upper() else 'DES'
        except Exception:
            pass_dir = 'UNK'

        for attempt in range(1, max_retries + 1):
            try:
                info  = img.sampleRectangle(
                    region=rect, scale=GEE_SCALE_M, defaultValue=np.nan
                ).getInfo()
                props = info.get('properties', {})

                band_arrs, ok = [], True
                for b in _BANDS:
                    raw = props.get(b)
                    if raw is None:
                        ok = False
                        break
                    band_arrs.append(np.asarray(raw, dtype=np.float32))
                if not ok:
                    break

                native = np.stack(band_arrs, axis=-1)   # (H, W, 2)
                valid  = np.all(
                    np.isfinite(native) & (native >= _DB_MIN) & (native <= _DB_MAX),
                    axis=-1,
                )
                if not valid.any():
                    break
                native[~valid] = np.nan

                patch_32 = resize_to_canonical(native, CANONICAL)
                mask_32  = resize_to_canonical(
                    valid.astype(np.float32), CANONICAL
                ).round().astype(np.uint8)

                scenes.append({
                    'date':     date_iso,
                    'pass_dir': pass_dir,
                    'patch':    patch_32,
                    'mask':     mask_32,
                })
                break

            except Exception as exc:
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                else:
                    logger.warning(f"  S1 {date_iso}: sampleRectangle failed — {exc}")

    return scenes


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Sentinel-1 IW GRD WQ patches (32×32, VV+VH dB) via GEE.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--slu-station-file',    default='data/raw/slu_mvm/slu_mvm_measured_station_ids.csv')
    p.add_argument('--hydobs-station-file', default=None)
    p.add_argument('--fmi-station-file',    default=None)
    p.add_argument('--shard-dir',   default='data/processed/sentinel1_wq_patches')
    p.add_argument('--start-year',  type=int, default=2014)
    p.add_argument('--end-year',    type=int, default=2025)
    p.add_argument('--pilot',       type=int, default=None)
    p.add_argument('--gee-project', default=None)
    args = p.parse_args()

    ee = check_ee()
    try:
        ee.Initialize(project=args.gee_project) if args.gee_project else ee.Initialize()
        logger.info(f"GEE initialised (project={args.gee_project or 'default'})")
    except Exception as exc:
        logger.error(f"GEE init failed: {exc}")
        sys.exit(1)

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    stations = load_wq_stations(
        slu_file    = Path(args.slu_station_file),
        hydobs_file = Path(args.hydobs_station_file) if args.hydobs_station_file else None,
        fmi_file    = Path(args.fmi_station_file)    if args.fmi_station_file    else None,
        pilot_n     = args.pilot,
    )

    years = list(range(args.start_year, args.end_year + 1))
    logger.info(f"Years: {years[0]}–{years[-1]}  stations: {len(stations):,}")

    t0 = time.time()
    for yi, y in enumerate(years, 1):
        logger.info(f"\n[{yi}/{len(years)}] {y}")
        for _, row in stations.iterrows():
            sid = str(row['station_id'])
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            if annual_shard_exists(shard_dir, sid, y, SUFFIX):
                continue

            # Accumulate all scenes across 12 months
            all_scenes = []
            for m in range(1, 13):
                all_scenes.extend(extract_month(ee, lat, lon, y, m))

            if not all_scenes:
                continue

            save_annual_shard(
                shard_dir=shard_dir, station_id=sid, lat=lat, lon=lon,
                year=y, suffix=SUFFIX,
                patch=np.stack([s['patch']    for s in all_scenes]),
                mask=np.stack([s['mask']     for s in all_scenes]),
                dates=[s['date']     for s in all_scenes],
                channels=_CHANNELS, footprint_m=PATCH_HALF_M * 2,
                scene_passes=np.array([s['pass_dir'] for s in all_scenes], dtype='S3'),
            )
            logger.info(f"  {sid}  {len(all_scenes)} scenes")

    logger.info(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == '__main__':
    main()
