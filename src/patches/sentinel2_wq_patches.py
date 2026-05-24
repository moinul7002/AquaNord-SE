#!/usr/bin/env python3
"""
Sentinel-2 MSI multispectral patches for WQ monitoring stations via GEE.

Sensor:     Sentinel-2 MSI  —  L1C TOA (2015–2016) + L2A SR (2017–2025)
Bands (10): B2  490 nm  blue     20 m   CDOM / OC4
            B3  560 nm  green    20 m   Chl-a / turbidity
            B4  665 nm  red      20 m   Chl-a / NDCI denominator
            B5  705 nm  red-edge 20 m   NDCI numerator / Chl fluorescence
            B6  740 nm  red-edge 20 m   Chl / phycocyanin
            B7  783 nm  red-edge 20 m   Chl / TSS
            B8  842 nm  NIR      10 m   water mask / AC  (resampled to 20 m)
            B8A 865 nm  NIR      20 m   AC reference
            B11 1610 nm SWIR-1   20 m   TSS / SPM
            B12 2190 nm SWIR-2   20 m   water mask / TSS
Footprint:  10 km × 10 km  (half_m = 5 000 m)
Scale:      20 m for sampleRectangle  →  native ~ 500 × 500 px
Canonical:  32 × 32 (bilinear resize from native)

Multiple S2 tiles overlap per day — images are mosaicked (median) before sampling.
Cloud mask:  L2A: SCL classes 8/9/10/3 = cloud/cirrus/cloud-shadow → mask=0
             L1C: QA60 bits 10/11 = cloud → mask=0

Output:
    data/processed/sentinel2_wq_patches/<station_id>/<YYYY-MM>_S2.npz
        patch       (n_scenes, 32, 32, 10)  float32  canonical tensor
        mask        (n_scenes, 32, 32)       uint8    cloud-free pixel flag
        dates       (n_scenes,)              |S10     one entry per clear overpass
        channels    (10,)                    object
        product_level  (n_scenes,)           |S3      'L1C' or 'L2A'
        station_id, latitude, longitude, footprint_m

Usage:
    python -m src.patches.sentinel2_wq_patches \\
        --start-year 2015 --end-year 2025 --gee-project <id>
"""

import argparse
import calendar
import datetime
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

_COLL_L2A    = 'COPERNICUS/S2_SR_HARMONIZED'   # 2017-03-28 +
_COLL_L1C    = 'COPERNICUS/S2_HARMONIZED'       # 2015-06-23 +  (TOA)
_L2A_START   = datetime.date(2017, 3, 28)

_BANDS       = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
_CHANNELS    = [
    'blue_490nm', 'green_560nm', 'red_665nm',
    're1_705nm', 're2_740nm', 're3_783nm',
    'nir_842nm', 'nira_865nm', 'swir1_1610nm', 'swir2_2190nm',
]
_S2_SCALE    = 1 / 10_000

PATCH_HALF_M = 5_000.0
GEE_SCALE_M  = 20          # sampleRectangle resolution (all S2 WQ bands at 20 m)
CANONICAL    = 32
SUFFIX       = 'S2'

# SCL classes that count as cloud / unusable for L2A
_SCL_CLOUD   = {3, 8, 9, 10}   # cloud shadow, med/high cloud prob, thin cirrus


# ── cloud masking ─────────────────────────────────────────────────────────────

def _apply_cloud_mask_l2a(ee, img):
    scl = img.select('SCL')
    cloud = (scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)))
    return img.updateMask(cloud.Not())


def _apply_cloud_mask_l1c(ee, img):
    qa = img.select('QA60')
    cloud = qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))
    return img.updateMask(cloud.eq(0))


# ── extraction ────────────────────────────────────────────────────────────────

def _mosaic_day(ee, coll):
    """Mosaic all S2 granules for a single day into one image (median composite)."""
    return coll.median()


def extract_month(ee, lat: float, lon: float, year: int, month: int,
                  max_retries: int = 3) -> dict:
    """
    Pull all clear S2 scenes for one station-month.

    Returns only days that have at least one valid pixel in the patch.
    L1C and L2A are selected automatically by date.
    """
    rect = station_rect(ee, lat, lon, PATCH_HALF_M)
    _, n_days = calendar.monthrange(year, month)
    month_start = datetime.date(year, month, 1)
    month_end   = datetime.date(year, month, n_days)

    # Choose collection based on year/month
    if month_end < _L2A_START:
        collections = [(_COLL_L1C, 'L1C')]
    elif month_start >= _L2A_START:
        collections = [(_COLL_L2A, 'L2A')]
    else:
        # Month straddles the L1C→L2A boundary
        collections = [(_COLL_L1C, 'L1C'), (_COLL_L2A, 'L2A')]

    patch_list, mask_list, date_list, level_list = [], [], [], []

    for date_d in (datetime.date(year, month, d) for d in range(1, n_days + 1)):
        date_iso   = date_d.strftime('%Y-%m-%d')
        coll_id, level = (_COLL_L2A, 'L2A') if date_d >= _L2A_START else (_COLL_L1C, 'L1C')
        mask_fn        = _apply_cloud_mask_l2a if level == 'L2A' else _apply_cloud_mask_l1c

        day_start = ee.Date(date_iso)
        coll = (ee.ImageCollection(coll_id)
                .filterDate(day_start, day_start.advance(1, 'day'))
                .filterBounds(rect))

        try:
            if coll.size().getInfo() == 0:
                continue
        except Exception:
            continue

        img = (_mosaic_day(ee, coll.map(mask_fn))
               .select(_BANDS)
               .multiply(_S2_SCALE))

        for attempt in range(1, max_retries + 1):
            try:
                info  = img.sampleRectangle(
                    region=rect, scale=GEE_SCALE_M, defaultValue=-9999
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

                native  = np.stack(band_arrs, axis=-1)   # (H, W, 10)
                valid   = np.all((native >= 0) & (native <= 1.5), axis=-1)
                if not valid.any():
                    break
                native[~valid] = np.nan

                canonical_patch = resize_to_canonical(native, CANONICAL)       # (32,32,10)
                canonical_mask  = resize_to_canonical(
                    valid.astype(np.float32), CANONICAL
                ).round().astype(np.uint8)

                patch_list.append(canonical_patch)
                mask_list.append(canonical_mask)
                date_list.append(date_iso)
                level_list.append(level)
                break

            except Exception as exc:
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                else:
                    logger.warning(f"  S2 {date_iso}: sampleRectangle failed — {exc}")

    if not patch_list:
        return {}

    return {
        'patch':         np.stack(patch_list),           # (n, 32, 32, 10)
        'mask':          np.stack(mask_list),             # (n, 32, 32)
        'dates':         date_list,
        'product_level': level_list,
        'channels':      _CHANNELS,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Sentinel-2 WQ patches (32×32, 10 MSI bands) via GEE.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--slu-station-file',    default='data/raw/slu_mvm/slu_mvm_measured_station_ids.csv')
    p.add_argument('--hydobs-station-file', default=None)
    p.add_argument('--fmi-station-file',    default=None)
    p.add_argument('--shard-dir',   default='data/processed/sentinel2_wq_patches')
    p.add_argument('--start-year',  type=int, default=2015)
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

            # Accumulate all observed scenes across 12 months
            all_patches, all_masks, all_dates, all_levels = [], [], [], []
            for m in range(1, 13):
                payload = extract_month(ee, lat, lon, y, m)
                if not payload:
                    continue
                all_patches.append(payload['patch'])
                all_masks.append(payload['mask'])
                all_dates.extend(payload['dates'])
                all_levels.extend(payload['product_level'])

            if not all_patches:
                continue
            save_annual_shard(
                shard_dir=shard_dir, station_id=sid, lat=lat, lon=lon,
                year=y, suffix=SUFFIX,
                patch=np.concatenate(all_patches), mask=np.concatenate(all_masks),
                dates=all_dates, channels=_CHANNELS,
                footprint_m=PATCH_HALF_M * 2,
                product_level=np.array(all_levels, dtype='S3'),
            )
            logger.info(f"  {sid}  {len(all_dates)} scenes")

    logger.info(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == '__main__':
    main()
