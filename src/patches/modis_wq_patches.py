#!/usr/bin/env python3
"""
MODIS MOD09GA / MYD09GA surface-reflectance patches for WQ monitoring stations.

NOT the snow-cover NDSI script (scripts/download_modis_patches.py, MOD10A1).
This extracts optical surface reflectance bands used as WQ proxies.

Sensor:     MODIS Terra (MOD09GA) + Aqua (MYD09GA)  via GEE
Bands (4):  sur_refl_b01  620–670 nm  red       500 m
            sur_refl_b02  841–876 nm  NIR       500 m
            sur_refl_b03  459–479 nm  blue      500 m
            sur_refl_b04  545–565 nm  green     500 m
Footprint:  16 km × 16 km  (half_m = 8 000 m → 32 × 32 pixels at 500 m native)
Canonical:  32 × 32  (no resampling — native pixel count already matches)
Coverage:   Terra 2000–2025  ·  Aqua 2002–2025  ·  all 12 months

Output:
    data/processed/modis_wq_patches/<station_id>/<YYYY>_MOD.npz   Terra
    data/processed/modis_wq_patches/<station_id>/<YYYY>_MYD.npz   Aqua
        patch       (n_valid_days, 32, 32, 4)  float32  sparse — only days with ≥1 valid pixel
        mask        (n_valid_days, 32, 32)      uint8
        dates       (n_valid_days,)             |S10
        channels    (4,)                         object
        station_id, latitude, longitude, footprint_m

Usage:
    python -m src.patches.modis_wq_patches \\
        --start-year 2000 --end-year 2025 --gee-project <id>

    # Pilot: 10 stations, one year
    python -m src.patches.modis_wq_patches \\
        --start-year 2020 --end-year 2020 --pilot 10 --gee-project <id>
"""

import argparse
import calendar
import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.patches._utils import (
    build_year_month_list, check_ee, crop_pad,
    load_wq_stations, save_annual_shard, annual_shard_exists, station_rect,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_TERRA       = 'MODIS/061/MOD09GA'
_AQUA        = 'MODIS/061/MYD09GA'
_SR_BANDS    = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04']
_CHANNELS    = ['red_620_670nm', 'nir_841_876nm', 'blue_459_479nm', 'green_545_565nm']
_SR_SCALE    = 0.0001
_SR_MIN      = -0.01   # valid surface reflectance after scaling
_SR_MAX      =  1.5
PATCH_HALF_M = 8_000.0
CANONICAL    = 32
SUFFIX       = {'MODIS/061/MOD09GA': 'MOD', 'MODIS/061/MYD09GA': 'MYD'}


# ── extraction ────────────────────────────────────────────────────────────────

def extract_month(ee, lat: float, lon: float, year: int, month: int,
                  collection_id: str, max_retries: int = 3) -> dict:
    """
    Pull one station-month of MODIS surface reflectance, one day at a time.

    Days with no valid overpass are stored as all-NaN patch + zero mask.
    """
    rect = station_rect(ee, lat, lon, PATCH_HALF_M)
    _, n_days = calendar.monthrange(year, month)
    dates = [f"{year}-{month:02d}-{d:02d}" for d in range(1, n_days + 1)]

    patch = np.full((n_days, CANONICAL, CANONICAL, len(_SR_BANDS)), np.nan, dtype=np.float32)
    mask  = np.zeros((n_days, CANONICAL, CANONICAL), dtype=np.uint8)

    for i, date_iso in enumerate(dates):
        day_start = ee.Date(date_iso)
        coll = (ee.ImageCollection(collection_id)
                .filterDate(day_start, day_start.advance(1, 'day'))
                .filterBounds(rect))
        try:
            if coll.size().getInfo() == 0:
                continue
        except Exception:
            continue

        img = ee.Image(coll.first()).select(_SR_BANDS).multiply(_SR_SCALE)

        for attempt in range(1, max_retries + 1):
            try:
                info  = img.sampleRectangle(region=rect, defaultValue=-9999).getInfo()
                props = info.get('properties', {})

                band_arrs, ok = [], True
                for b in _SR_BANDS:
                    raw = props.get(b)
                    if raw is None:
                        ok = False
                        break
                    band_arrs.append(crop_pad(np.asarray(raw, dtype=np.float32), CANONICAL))
                if not ok:
                    break

                stacked = np.stack(band_arrs, axis=-1)   # (32, 32, 4)
                valid   = np.all((stacked >= _SR_MIN) & (stacked <= _SR_MAX), axis=-1)
                stacked[~valid] = np.nan
                patch[i] = stacked
                mask[i]  = valid.astype(np.uint8)
                break

            except Exception as exc:
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                else:
                    logger.warning(f"  {date_iso}: sampleRectangle failed — {exc}")

    return {'patch': patch, 'mask': mask, 'dates': dates, 'channels': _CHANNELS}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='MODIS MOD09GA/MYD09GA WQ patches (32×32, 4 SR bands) via GEE.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--slu-station-file',
                   default='data/raw/slu_mvm/slu_mvm_measured_station_ids.csv')
    p.add_argument('--hydobs-station-file', default=None,
                   help='Also include SMHI HydObs stations (optional)')
    p.add_argument('--fmi-station-file', default=None,
                   help='Also include FMI stations (optional)')
    p.add_argument('--shard-dir',   default='data/processed/modis_wq_patches')
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

    sensors = [(_TERRA, 'MOD'), (_AQUA, 'MYD')]
    years   = list(range(args.start_year, args.end_year + 1))
    logger.info(f"Years: {years[0]}–{years[-1]}  stations: {len(stations):,}  sensors: Terra + Aqua")

    t0 = time.time()
    for yi, y in enumerate(years, 1):
        logger.info(f"\n[{yi}/{len(years)}] {y}")
        for coll_id, sfx in sensors:
            if sfx == 'MYD' and y < 2002:
                continue
            for _, row in stations.iterrows():
                sid = str(row['station_id'])
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                if annual_shard_exists(shard_dir, sid, y, sfx):
                    continue

                # Sparse accumulation: keep only days with ≥1 valid pixel
                valid_patches, valid_masks, valid_dates = [], [], []
                for m in range(1, 13):
                    payload = extract_month(ee, lat, lon, y, m, coll_id)
                    for i, date_iso in enumerate(payload['dates']):
                        if payload['mask'][i].any():   # at least one valid pixel
                            valid_patches.append(payload['patch'][i])
                            valid_masks.append(payload['mask'][i])
                            valid_dates.append(date_iso)

                if not valid_patches:
                    continue
                save_annual_shard(
                    shard_dir=shard_dir, station_id=sid, lat=lat, lon=lon,
                    year=y, suffix=sfx,
                    patch=np.stack(valid_patches),
                    mask=np.stack(valid_masks),
                    dates=valid_dates, channels=_CHANNELS,
                    footprint_m=PATCH_HALF_M * 2,
                )
                logger.info(f"  {sid} {sfx} {y}  {len(valid_dates)} valid days")

    logger.info(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == '__main__':
    main()
