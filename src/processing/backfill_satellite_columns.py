#!/usr/bin/env python3
"""
Backfill new metadata columns onto existing satellite Parquet files.

Applies three columns added in the v3 audit (2026-05-05) without re-extracting
from GEE.  Safe to run on partially-extracted archives; skips files that already
have all three columns.

New columns:
    owt_class_v1_provisional  — renamed from owt_class (MD-6)
    rw_blue_negative_flag     — 1 if blue-band reflectance < 0 (MD-1)
    ac_method                 — atmospheric correction method used (MD-1)

AC method by sensor:
    modis_terra / modis_aqua  → 'modis_ac'
    landsat_*                 → 'lasrc'
    sentinel1                 → 'sar'
    sentinel2 L2A             → 'sen2cor'   (derived from product_level column)
    sentinel2 L1C             → 'toa'
    sentinel3                 → 'c2rcc'

Usage:
    python -m src.processing.backfill_satellite_columns
    python -m src.processing.backfill_satellite_columns --dry-run
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAT_DIR = Path("data/raw/satellite/SE")

# Map filename pattern fragment → fixed ac_method value
# Sentinel-2 is handled dynamically via product_level column
_AC_METHOD = {
    "modis":     "modis_ac",
    "landsat":   "lasrc",
    "sentinel1": "sar",
    "sentinel3": "c2rcc",
}

# Blue-band column name per sensor family
_BLUE_COL = {
    "modis":     "rw_b03_blue",   # MOD09GA sur_refl_b03
    "landsat":   "rw_blue",
    "sentinel2": "rw_B2_blue",
    "sentinel3": "rw_oa04_490nm",
}


def _sensor_family(path: Path) -> str:
    name = path.stem.lower()
    for fam in ["sentinel1", "sentinel2", "sentinel3", "modis", "landsat"]:
        if fam in name:
            return fam
    return "unknown"


def backfill_file(path: Path, dry_run: bool = False) -> bool:
    """
    Add missing columns to one Parquet file in-place.
    Returns True if the file was modified.
    """
    df = pd.read_parquet(path)
    family = _sensor_family(path)
    modified = False

    # 1. owt_class → owt_class_v1_provisional
    if "owt_class" in df.columns and "owt_class_v1_provisional" not in df.columns:
        df = df.rename(columns={"owt_class": "owt_class_v1_provisional"})
        modified = True

    # 2. rw_blue_negative_flag
    if "rw_blue_negative_flag" not in df.columns:
        blue_col = _BLUE_COL.get(family)
        if blue_col and blue_col in df.columns:
            df["rw_blue_negative_flag"] = (df[blue_col] < 0).astype("int8")
        else:
            df["rw_blue_negative_flag"] = 0   # SAR / unknown — no blue band
        modified = True

    # 3. ac_method
    if "ac_method" not in df.columns:
        if family == "sentinel2" and "product_level" in df.columns:
            df["ac_method"] = df["product_level"].map({"L2A": "sen2cor", "L1C": "toa"})
        else:
            df["ac_method"] = _AC_METHOD.get(family, "unknown")
        modified = True

    if not modified:
        logger.info(f"  SKIP (already up to date): {path.name}")
        return False

    if dry_run:
        logger.info(f"  DRY-RUN (would update): {path.name}")
        return False

    df.to_parquet(path, index=False)
    logger.info(f"  Updated: {path.name}")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Backfill owt_class_v1_provisional, rw_blue_negative_flag, ac_method "
                    "onto existing satellite Parquet files.",
    )
    p.add_argument("--sat-dir", default=str(SAT_DIR),
                   help="Directory containing sweden_*_stations_*.parquet files")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change without writing anything")
    args = p.parse_args()

    sat_dir = Path(args.sat_dir)
    files = sorted(sat_dir.glob("sweden_*_stations_*.parquet"))

    if not files:
        logger.warning(f"No station Parquet files found in {sat_dir}")
        return

    logger.info(f"Found {len(files)} station Parquet files in {sat_dir}")
    n_updated = 0
    for f in files:
        if backfill_file(f, dry_run=args.dry_run):
            n_updated += 1

    logger.info(f"\nDone — {'would update' if args.dry_run else 'updated'} {n_updated}/{len(files)} files.")


if __name__ == "__main__":
    main()
