#!/usr/bin/env python3
"""
Landsat 7 / 8 / 9 surface-reflectance patches for WQ monitoring stations via GEE.

Sensor:   Landsat 7 ETM+  (LE07/C02/T1_L2)  2000–2023
          Landsat 8 OLI   (LC08/C02/T1_L2)  2013–2025
          Landsat 9 OLI-2 (LC09/C02/T1_L2)  2021–2025

Bands (6, harmonised across all three missions):
    blue    482 / 479 nm    L8/L9 SR_B2  /  L7 SR_B1   30 m
    green   562 / 560 nm    L8/L9 SR_B3  /  L7 SR_B2   30 m
    red     655 / 660 nm    L8/L9 SR_B4  /  L7 SR_B3   30 m
    nir     865 / 835 nm    L8/L9 SR_B5  /  L7 SR_B4   30 m
    swir1  1609 /1650 nm    L8/L9 SR_B6  /  L7 SR_B5   30 m
    swir2  2201 /2220 nm    L8/L9 SR_B7  /  L7 SR_B7   30 m

Footprint:  10 km × 10 km  (half_m = 5 000 m  →  ~ 333 × 333 px at 30 m native)
Canonical:  32 × 32  (bilinear resize from native)

Cloud mask: QA_PIXEL bits  1 (dilated cloud), 2 (cirrus), 3 (cloud), 4 (cloud shadow)
Scale:      Landsat C02 L2  SR = raw × 0.0000275 − 0.2

Output:
    data/processed/landsat_wq_patches/<station_id>/<YYYY-MM>_L7.npz
    data/processed/landsat_wq_patches/<station_id>/<YYYY-MM>_L8.npz
    data/processed/landsat_wq_patches/<station_id>/<YYYY-MM>_L9.npz
        patch    (n_scenes, 32, 32, 6)  float32   canonical tensor
        mask     (n_scenes, 32, 32)     uint8     cloud-free pixel flag
        dates    (n_scenes,)            |S10
        channels (6,)                   object    harmonised band labels
        satellite (n_scenes,)           |S2       'L7' / 'L8' / 'L9'
        station_id, latitude, longitude, footprint_m

Usage:
    python -m src.patches.landsat_wq_patches \\
        --start-year 2000 --end-year 2025 --gee-project <id>
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

_SENSORS = [
    # (collection_id,            suffix, start_year, end_year, band_map)
    ('LANDSAT/LE07/C02/T1_L2', 'L7', 2000, 2023,
     {'blue': 'SR_B1', 'green': 'SR_B2', 'red': 'SR_B3',
      'nir': 'SR_B4', 'swir1': 'SR_B5', 'swir2': 'SR_B7'}),
    ('LANDSAT/LC08/C02/T1_L2', 'L8', 2013, 2025,
     {'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4',
      'nir': 'SR_B5', 'swir1': 'SR_B6', 'swir2': 'SR_B7'}),
    ('LANDSAT/LC09/C02/T1_L2', 'L9', 2021, 2025,
     {'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4',
      'nir': 'SR_B5', 'swir1': 'SR_B6', 'swir2': 'SR_B7'}),
]

_CHANNELS     = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
_SR_SCALE     = 0.0000275
_SR_OFFSET    = -0.2
_SR_MIN       = -0.1    # valid SR range after scaling
_SR_MAX       =  1.5

# QA_PIXEL cloud bits
_QA_CLOUD_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)

PATCH_HALF_M  = 5_000.0
GEE_SCALE_M   = 30
CANONICAL     = 32


# ── cloud masking ─────────────────────────────────────────────────────────────

def _mask_landsat(ee, img):
    qa    = img.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(_QA_CLOUD_BITS).neq(0)
    return img.updateMask(cloud.Not())


# ── extraction ────────────────────────────────────────────────────────────────

def extract_month(
    ee, lat: float, lon: float,
    year: int, month: int,
    collection_id: str, band_map: dict,
    max_retries: int = 3,
) -> list:
    """
    Pull all clear Landsat scenes for one station-month from one collection.

    Returns a list of scene dicts: {date, patch (32,32,6), mask (32,32)}.
    """
    rect = station_rect(ee, lat, lon, PATCH_HALF_M)
    _, n_days = calendar.monthrange(year, month)
    date_start = f"{year}-{month:02d}-01"
    date_end   = f"{year}-{month:02d}-{n_days:02d}"

    coll = (ee.ImageCollection(collection_id)
            .filterDate(date_start, f"{year}-{month:02d}-{n_days:02d}")
            .filterBounds(rect)
            .map(lambda img: _mask_landsat(ee, img)))

    try:
        n_imgs = coll.size().getInfo()
    except Exception:
        return []
    if n_imgs == 0:
        return []

    # Select and rename to harmonised channel order
    src_bands = [band_map[c] for c in _CHANNELS]
    coll = coll.select(src_bands + ['QA_PIXEL'], _CHANNELS + ['QA_PIXEL'])
    coll = coll.select(_CHANNELS).map(
        lambda img: img.multiply(_SR_SCALE).add(_SR_OFFSET)
    )

    # Get per-image dates and extract one at a time
    scenes = []
    img_list = coll.toList(n_imgs)
    for idx in range(n_imgs):
        img = ee.Image(img_list.get(idx))
        date_iso = img.date().format('YYYY-MM-dd').getInfo()

        for attempt in range(1, max_retries + 1):
            try:
                info  = img.sampleRectangle(
                    region=rect, scale=GEE_SCALE_M, defaultValue=-9999
                ).getInfo()
                props = info.get('properties', {})

                band_arrs, ok = [], True
                for c in _CHANNELS:
                    raw = props.get(c)
                    if raw is None:
                        ok = False
                        break
                    band_arrs.append(np.asarray(raw, dtype=np.float32))
                if not ok:
                    break

                native = np.stack(band_arrs, axis=-1)    # (H, W, 6)
                valid  = np.all((native >= _SR_MIN) & (native <= _SR_MAX), axis=-1)
                if not valid.any():
                    break
                native[~valid] = np.nan

                patch_32 = resize_to_canonical(native, CANONICAL)
                mask_32  = resize_to_canonical(
                    valid.astype(np.float32), CANONICAL
                ).round().astype(np.uint8)

                scenes.append({'date': date_iso, 'patch': patch_32, 'mask': mask_32})
                break

            except Exception as exc:
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                else:
                    logger.warning(f"  Landsat {date_iso}: sampleRectangle failed — {exc}")

    return scenes


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Landsat 7/8/9 WQ patches (32×32, 6 SR bands) via GEE.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--slu-station-file',    default='data/raw/slu_mvm/slu_mvm_measured_station_ids.csv')
    p.add_argument('--hydobs-station-file', default=None)
    p.add_argument('--fmi-station-file',    default=None)
    p.add_argument('--shard-dir',   default='data/processed/landsat_wq_patches')
    p.add_argument('--start-year',  type=int, default=2000)
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
    logger.info(f"Years: {years[0]}–{years[-1]}  stations: {len(stations):,}  sensors: L7 + L8 + L9")

    t0 = time.time()
    for yi, y in enumerate(years, 1):
        logger.info(f"\n[{yi}/{len(years)}] {y}")
        for coll_id, sfx, yr_start, yr_end, band_map in _SENSORS:
            if not (yr_start <= y <= yr_end):
                continue
            for _, row in stations.iterrows():
                sid = str(row['station_id'])
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                if annual_shard_exists(shard_dir, sid, y, sfx):
                    continue

                # Accumulate all scenes across 12 months
                all_scenes = []
                for m in range(1, 13):
                    all_scenes.extend(extract_month(ee, lat, lon, y, m, coll_id, band_map))

                if not all_scenes:
                    continue
                save_annual_shard(
                    shard_dir=shard_dir, station_id=sid, lat=lat, lon=lon,
                    year=y, suffix=sfx,
                    patch=np.stack([s['patch'] for s in all_scenes]),
                    mask=np.stack([s['mask']  for s in all_scenes]),
                    dates=[s['date'] for s in all_scenes],
                    channels=_CHANNELS, footprint_m=PATCH_HALF_M * 2,
                    satellite=np.array([sfx] * len(all_scenes), dtype='S2'),
                )
                logger.info(f"  {sid} {sfx} {y}  {len(all_scenes)} scenes")

    logger.info(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == '__main__':
    main()
