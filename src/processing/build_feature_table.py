#!/usr/bin/env python3
"""
SW-STEP-7 — Assemble the Sweden analysis-ready feature table.

Joins all data sources per (station_id, date) anchored on SLU MVM chemistry samples:

  Anchor  : SLU MVM chemistry       (one row per WQ sample)
  Exact   : ERA5 at stations         (station_id, date)
  Exact   : MESAN at stations        (station_id, date)  ← from station_grid_join.py
  ±3 days : Satellite point values   (5 sensors: modis, landsat, sentinel1, sentinel2, sentinel3)
  Static  : HydroATLAS covariates    (station_id only, no date)

Satellite matchup notes:
  - Default tolerance is ±3 days (compromise between fast-varying Chl-a and slow-varying TP/TN).
  - sat_days_offset_{sensor} column records actual days between anchor and matched obs;
    use this to apply per-parameter filtering downstream (e.g. ≤1 day for Chl-a, ≤7 for TP).
  - n_sat_obs_{sensor} records how many satellite obs existed within the window.
  - When multiple obs fall within the window, pd.merge_asof takes the temporally nearest one.

Quality summary:
  quality_summary integer (0–3) aggregates upstream data availability per row:
    0 = ERA5 present + ≥1 satellite match (complete)
    1 = ERA5 present, no satellite match
    2 = ERA5 missing, satellite present
    3 = ERA5 missing, no satellite match

Also adds:
  pre_wfd_method (bool)  — True for samples before 2007-01-01 (MD-8)

Output: data/interim/SE/feature_table/feature_table_SE_YYYY.parquet

Usage:
    python -m src.processing.build_feature_table
    python -m src.processing.build_feature_table --start-year 2015 --end-year 2020
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW         = Path("data/raw")
INTERIM     = Path("data/interim/SE")
FEATURE_DIR = INTERIM / "feature_table"

SENSORS = ["modis", "landsat", "sentinel1", "sentinel2", "sentinel3"]

# Derived WQ indices joined from derived_wq_timeseries.parquet (annual station aggregates).
# Prefixed with "derived_" to distinguish from raw measurements.
# NOTE: excluded from all ML feature tiers — TSI←TP, Chl-a:TP←Chl-a, etc. create target leakage.
DERIVED_WQ_COLS = [
    "np_molar_ratio", "chla_tp_ratio",
    "tsi", "tsi_chla", "tsi_tp", "tsi_secchi",
    "do_pct_sat", "do_deficit_mg_l",
    "n_samples",   # annual observation count — data density indicator
]
SATELLITE_TOLERANCE_DAYS = 7   # capture ±7d; use sat_days_offset_{sensor} / sat_match_*d_{sensor} flags to filter
MATCH_WINDOWS = [1, 3, 7]      # boolean flag columns added for each window


# ── loaders ───────────────────────────────────────────────────────────────────

def _load_slu_mvm(year: int) -> pd.DataFrame:
    p = RAW / "slu_mvm" / f"slu_mvm_chemistry_{year}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["sampling_date"]).dt.normalize()
    df["station_id"] = df["station_id"].astype(str)
    df["pre_wfd_method"] = df["date"] < pd.Timestamp("2007-01-01")
    return df


def _load_era5(year: int) -> pd.DataFrame:
    p = RAW / "era5" / f"era5_sweden_{year}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["station_id"] = df["station_id"].astype(str)
    # Prefix columns to avoid collision with WQ columns
    rename = {c: f"era5_{c}" for c in df.columns if c not in ("station_id", "date")}
    return df.rename(columns=rename)


def _load_mesan(year: int) -> pd.DataFrame:
    p = INTERIM / "mesan" / f"station_mesan_{year}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["station_id"] = df["station_id"].astype(str)
    rename = {c: f"mesan_{c}" for c in df.columns
              if c not in ("station_id", "date", "mesan_product")}
    return df.rename(columns=rename)


def _load_satellite_sensor(sensor: str, year: int) -> pd.DataFrame:
    """Load all available parquets for one sensor/year; add sensor prefix to columns."""
    frames = []
    # Try both YYYY and YYYY_stations filename patterns
    for pat in [f"sweden_{sensor}_{year}_stations.parquet",
                f"sweden_{sensor}_{year}.parquet"]:
        p = RAW / "satellite" / "SE" / pat
        if p.exists():
            frames.append(pd.read_parquet(p))
            break
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # Strip trailing '.0' from float-string station IDs (e.g. '100.0' → '100')
    df["station_id"] = df["station_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    # Keep station_id + date; prefix everything else
    id_cols = {"station_id", "date"}
    rename = {c: f"{sensor}_{c}" for c in df.columns if c not in id_cols}
    return df.rename(columns=rename)


def _load_hydroatlas() -> pd.DataFrame:
    p = RAW / "hydroatlas" / "hydroatlas_station_covariates_SE.parquet"
    if not p.exists():
        logger.warning("HydroATLAS not found — static covariates will be absent")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["station_id"] = df["station_id"].astype(str)
    return df


def _load_derived_wq() -> pd.DataFrame:
    """Load annual derived WQ indices (TSI, N:P ratio, etc.) per station × year."""
    p = INTERIM / "derived_wq_timeseries.parquet"
    if not p.exists():
        logger.warning("derived_wq_timeseries.parquet not found — derived indices will be absent")
        return pd.DataFrame()
    df = pd.read_parquet(p)
    keep = ["station_id", "year"] + [c for c in DERIVED_WQ_COLS if c in df.columns]
    df = df[keep].copy()
    df["station_id"] = df["station_id"].astype(str)
    rename = {c: f"derived_{c}" for c in df.columns if c not in ("station_id", "year")}
    return df.rename(columns=rename)


# ── temporal matchup ──────────────────────────────────────────────────────────

def _merge_satellite(anchor: pd.DataFrame, sat: pd.DataFrame,
                     sensor: str, tol_days: int = SATELLITE_TOLERANCE_DAYS) -> pd.DataFrame:
    """
    Left-join satellite observations to anchor rows within ±tol_days.
    Adds n_sat_obs_{sensor} (count within window) and sat_days_offset_{sensor}
    (days between anchor date and matched observation).
    """
    if sat.empty:
        return anchor

    tol = pd.Timedelta(days=tol_days)
    tol_ns = np.timedelta64(int(tol.total_seconds() * 1e9), "ns")
    result_chunks = []

    for sid, grp in anchor.groupby("station_id", sort=False):
        sat_grp = sat[sat["station_id"] == sid].sort_values("date")

        if sat_grp.empty:
            grp = grp.copy()
            grp[f"n_sat_obs_{sensor}"]      = 0
            grp[f"sat_days_offset_{sensor}"] = np.nan
            for w in MATCH_WINDOWS:
                grp[f"sat_match_{w}d_{sensor}"] = False
            result_chunks.append(grp)
            continue

        grp_sorted = grp.sort_values("date").copy()

        # Count obs within ±tol_days per anchor date (vectorized broadcast)
        anc_dates = grp_sorted["date"].values.astype("datetime64[ns]")
        sat_dates = sat_grp["date"].values.astype("datetime64[ns]")
        delta = np.abs(anc_dates[:, None] - sat_dates[None, :])
        grp_sorted[f"n_sat_obs_{sensor}"] = (delta <= tol_ns).sum(axis=1)

        # Rename satellite date to a temp column so we can compute the offset after merge
        sat_for_merge = (
            sat_grp
            .rename(columns={"date": f"_sat_date_{sensor}"})
            .drop(columns=["station_id"])
        )

        merged = pd.merge_asof(
            grp_sorted,
            sat_for_merge,
            left_on="date",
            right_on=f"_sat_date_{sensor}",
            direction="nearest",
            tolerance=tol,
        )

        merged[f"sat_days_offset_{sensor}"] = (
            (merged["date"] - merged[f"_sat_date_{sensor}"]).dt.days.abs()
        )
        merged = merged.drop(columns=[f"_sat_date_{sensor}"])
        for w in MATCH_WINDOWS:
            merged[f"sat_match_{w}d_{sensor}"] = merged[f"sat_days_offset_{sensor}"] <= w
        result_chunks.append(merged)

    if not result_chunks:
        return anchor

    return pd.concat(result_chunks, ignore_index=True)


# ── quality summary ───────────────────────────────────────────────────────────

def _compute_quality_summary(df: pd.DataFrame) -> pd.Series:
    """
    Row-level data quality score (0=complete, 1=minor gaps, 2=major gaps, 3=critical).
      0: ERA5 present AND ≥1 satellite match
      1: ERA5 present, no satellite match
      2: ERA5 missing, satellite present
      3: ERA5 missing, no satellite match
    """
    era5_col = next((c for c in df.columns if c.startswith("era5_t2m")), None)
    era5_ok = df[era5_col].notna() if era5_col else pd.Series(False, index=df.index)

    # Any satellite reflectance column present
    sat_rrs_cols = [c for c in df.columns
                    if any(c.startswith(f"{s}_rrs") or c.startswith(f"{s}_Rrs")
                           for s in SENSORS)]
    sat_count_cols = [f"n_sat_obs_{s}" for s in SENSORS if f"n_sat_obs_{s}" in df.columns]

    if sat_count_cols:
        sat_ok = df[sat_count_cols].gt(0).any(axis=1)
    elif sat_rrs_cols:
        sat_ok = df[sat_rrs_cols].notna().any(axis=1)
    else:
        sat_ok = pd.Series(False, index=df.index)

    score = pd.Series(0, index=df.index, dtype=np.int8)
    score[~era5_ok & sat_ok]  = 2
    score[~era5_ok & ~sat_ok] = 3
    score[era5_ok & ~sat_ok]  = 1
    # 0 already set for era5_ok & sat_ok
    return score


# ── year-level assembly ───────────────────────────────────────────────────────

def build_year(year: int, atlas: pd.DataFrame,
               derived_wq: pd.DataFrame = None) -> pd.DataFrame:
    anchor = _load_slu_mvm(year)
    if anchor.empty:
        logger.warning(f"  {year}: no SLU MVM chemistry — skipping")
        return pd.DataFrame()

    logger.info(f"  {year}: {len(anchor):,} WQ samples (anchor)")

    # Exact-date joins
    era5 = _load_era5(year)
    if not era5.empty:
        anchor = anchor.merge(era5, on=["station_id", "date"], how="left")
        era5_col = next((c for c in anchor.columns if c.startswith("era5_t2m")), None)
        if era5_col:
            n_era5 = anchor[era5_col].notna().sum()
            logger.info(f"  {year}: ERA5 joined — {n_era5:,} matches")

    mesan = _load_mesan(year)
    if not mesan.empty:
        anchor = anchor.merge(mesan, on=["station_id", "date"], how="left")
        logger.info(f"  {year}: MESAN joined")

    # Satellite ±3-day joins (use sat_days_offset_{sensor} for tighter filtering)
    for sensor in SENSORS:
        sat = _load_satellite_sensor(sensor, year)
        if sat.empty:
            logger.info(f"  {year}: {sensor} — no data available")
            continue
        anchor = _merge_satellite(anchor, sat, sensor, SATELLITE_TOLERANCE_DAYS)
        n_col = f"n_sat_obs_{sensor}"
        if n_col in anchor.columns:
            n_matched = (anchor[n_col] > 0).sum()
            logger.info(f"  {year}: {sensor} — {n_matched:,}/{len(anchor):,} within ±{SATELLITE_TOLERANCE_DAYS}d")

    # Static join (no date dimension)
    if not atlas.empty:
        anchor = anchor.merge(atlas, on="station_id", how="left")

    # Annual derived WQ indices (station × year) — excluded from ML tiers due to leakage
    if derived_wq is not None and not derived_wq.empty:
        year_slice = derived_wq[derived_wq["year"] == year].drop(columns="year")
        anchor = anchor.merge(year_slice, on="station_id", how="left")
        n_derived = anchor[[c for c in anchor.columns if c.startswith("derived_")]
                           ].notna().any(axis=1).sum()
        logger.info(f"  {year}: derived WQ indices joined — {n_derived:,} rows with coverage")

    # Row-level quality summary
    anchor["quality_summary"] = _compute_quality_summary(anchor)

    return anchor


# ── main ──────────────────────────────────────────────────────────────────────

def run(start_year: int = 2000, end_year: int = 2025) -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading HydroATLAS static covariates")
    atlas = _load_hydroatlas()

    logger.info("Loading derived WQ indices (TSI, N:P ratio, etc.)")
    derived_wq = _load_derived_wq()

    for year in range(start_year, end_year + 1):
        out_path = FEATURE_DIR / f"feature_table_SE_{year}.parquet"
        if out_path.exists():
            logger.info(f"  {year}: already exists, skipping (delete to regenerate)")
            continue

        logger.info(f"\n{'─'*50}")
        logger.info(f"  Year {year}")
        df = build_year(year, atlas, derived_wq)
        if df.empty:
            continue

        df.to_parquet(out_path, index=False)
        logger.info(f"  {year}: wrote {len(df):,} rows, {len(df.columns)} cols → {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Build Sweden feature table")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year",   type=int, default=2025)
    parser.add_argument("--country",    type=str, default="SE")
    parser.add_argument("--force",      action="store_true",
                        help="Overwrite existing parquets")
    args = parser.parse_args()

    if args.country != "SE":
        logger.warning("Only SE (Sweden) is supported in v1 — ignoring --country")

    if args.force:
        for p in FEATURE_DIR.glob("feature_table_SE_*.parquet"):
            p.unlink()
            logger.info(f"Removed {p.name}")

    run(args.start_year, args.end_year)


if __name__ == "__main__":
    main()
