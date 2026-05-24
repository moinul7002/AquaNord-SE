#!/usr/bin/env python3
"""
SHARED-STEP-2 — Build the multimodal aligned dataset.

For every WQ sample in the Sweden feature table (station_id, date, WQ targets),
locates the corresponding ERA5 12×12×9 spatial patch in the HDF5 archive and
writes everything to a structured ML-ready output.

This is the bridge between:
  - The tabular feature table  (one row per in-situ WQ measurement)
  - The ERA5 patch archive     (12×12 catchment-scale meteorology around each station)

Inputs:
  data/interim/SE/feature_table/feature_table_SE_YYYY.parquet  ← build_feature_table.py
  data/processed/patches_hdf5/era5_wq.h5               ← consolidate_patches.py
  data/raw/hydroatlas/hydroatlas_station_covariates_SE.parquet
  data/raw/smhi_hydrography/drainage_basins_4258.parquet

Outputs (data/processed/nordic_multimodal_dataset/):
  metadata.parquet      — one row per sample: station_id, date, WQ targets,
                          tabular features, fold_key, split (train/val/test)
  patches.zarr          — (n_samples, 12, 12, 9) float32  ERA5 patch tensors
  masks.zarr            — (n_samples, 12, 12)    uint8    validity masks
  channel_index.json    — ERA5 channel labels [{index, name, description}]
  split_index.parquet   — sample_id → split, temporal_split, lat_split

Spatial split strategy (prevents data leakage):
  Lake stations  (Hylak_id not null): fold key = Hylak_id (HydroLAKES ID)
  River/other stations:               fold key = SMHI drainage basin (111 Swedish basins)
                                       fallback: 1° lat/lon grid cell if spatial join fails
  80 % train / 10 % val / 10 % test — assigned by fold key (not sample)

Additional split indices in split_index.parquet:
  temporal_split — 2000-2020 = "train", 2021-2025 = "test" (tests temporal generalisation)
  lat_split      — <61°N = "south", ≥61°N = "north"        (tests latitudinal generalisation)

Framing note: ERA5 12×12 patches provide catchment-scale meteorological context; they
  capture spatial patterns in precipitation and temperature forcing rather than lake-
  surface optical or SAR signals. Frame as "spatiotemporal dataset with meteorological
  context" rather than "multimodal image fusion"; GEE satellite patches are planned v2.

Usage:
    python -m src.processing.build_multimodal_dataset
    python -m src.processing.build_multimodal_dataset --start-year 2010 --end-year 2020
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc as BloscCompressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HDF5_PATH    = Path("data/processed/patches_hdf5/era5_wq.h5")
INTERIM      = Path("data/interim/SE")
FEATURE_DIR  = INTERIM / "feature_table"
ATLAS_PATH   = Path("data/raw/hydroatlas/hydroatlas_station_covariates_SE.parquet")
BASINS_PATH  = Path("data/raw/smhi_hydrography/drainage_basins_4258.parquet")
OUT_DIR      = Path("data/processed/nordic_multimodal_dataset")

WQ_TARGETS  = ["tp_ug_l", "tn_ug_l", "chla_ug_l", "colour_mg_pt_l",
               "secchi_m", "turbidity_fnu", "ph", "do_mg_l", "toc_mg_l"]

ERA5_CHANNEL_DESC = {
    "t2m":  "2m air temperature (°C)",
    "tp":   "Total precipitation (mm day⁻¹)",
    "u10":  "10m eastward wind (m s⁻¹)",
    "v10":  "10m northward wind (m s⁻¹)",
    "ssrd": "Surface solar radiation downward (J m⁻²)",
    "d2m":  "2m dewpoint temperature (°C)",
    "sp":   "Surface pressure (Pa)",
    "sro":  "Surface runoff (m)",
    "ro":   "Runoff (m)",
}

# Per-channel normalization applied when writing to zarr.
# (offset, scale) → stored = (raw - offset) / scale  → roughly [-3, 3].
# None = channel overflows float16 (sp ~100 kPa, ssrd ~30 MJ m⁻²) → zeroed out.
ERA5_CHANNEL_NORM = {
    "t2m":  (0.0,    20.0),   # °C,      range ≈ -30..+30
    "tp":   (0.0,    10.0),   # mm day⁻¹, range ≈ 0..100
    "u10":  (0.0,    5.0),    # m s⁻¹
    "v10":  (0.0,    5.0),    # m s⁻¹
    "ssrd": None,             # float16 overflow → unusable
    "d2m":  (0.0,    20.0),   # °C
    "sp":   None,             # float16 overflow → unusable
    "sro":  (0.0,    0.001),  # m
    "ro":   (0.0,    0.002),  # m
}


# ── SMHI drainage basin spatial join ─────────────────────────────────────────

def _assign_smhi_basins(stations_df: pd.DataFrame) -> dict:
    """
    Return {station_id: basin_idx} by joining station points to 111 SMHI drainage
    basin polygons. Falls back to empty dict if geopandas or basin file unavailable.
    """
    if not BASINS_PATH.exists():
        logger.warning("SMHI basin file not found — river fold keys will use 1° grid")
        return {}
    try:
        import geopandas as gpd
        from shapely.wkb import loads as wkb_loads

        basin_df = pd.read_parquet(BASINS_PATH)
        geoms = [
            wkb_loads(bytes(g)) if isinstance(g, (bytes, bytearray, memoryview)) else None
            for g in basin_df["geometry"]
        ]
        basins = gpd.GeoDataFrame(basin_df, geometry=geoms, crs="EPSG:4258")
        basins["basin_idx"] = range(len(basins))

        pts = gpd.GeoDataFrame(
            stations_df[["station_id"]],
            geometry=gpd.points_from_xy(
                stations_df["longitude"], stations_df["latitude"]
            ),
            crs="EPSG:4258",
        )
        joined = gpd.sjoin(pts, basins[["basin_idx", "geometry"]], how="left", predicate="within")
        result = joined.set_index("station_id")["basin_idx"].dropna().astype(int).to_dict()
        logger.info(f"  SMHI basin join: {len(result):,} / {len(stations_df):,} stations assigned")
        return result
    except Exception as exc:
        logger.warning(f"SMHI basin join failed ({exc}) — river fold keys will use 1° grid")
        return {}


# ── Spatial split ─────────────────────────────────────────────────────────────

def _build_fold_keys(meta: pd.DataFrame, atlas: pd.DataFrame,
                     basin_lookup: dict) -> pd.Series:
    """
    Assign a spatial fold key to each sample.
    - Lake stations: Hylak_id (all obs from same lake stay in same fold)
    - River/other:   SMHI drainage basin index (111 Swedish basins)
                     fallback: 1° lat/lon grid cell
    """
    # lat/lon already in meta (from feature table); pull only fold-key cols from atlas
    df = meta[["station_id", "latitude", "longitude"]].drop_duplicates("station_id").merge(
        atlas[["station_id", "hydro_source", "Hylak_id"]].drop_duplicates("station_id"),
        on="station_id", how="left",
    )

    def _fold(row):
        if pd.notna(row.get("Hylak_id")):
            return f"lake_{int(row['Hylak_id'])}"
        bid = basin_lookup.get(str(row["station_id"]))
        if bid is not None:
            return f"smhi_basin_{int(bid)}"
        # Fallback: 1° lat/lon grid cell (more conservative than 0.5°)
        lat = round(float(row.get("latitude", 0) or 0))
        lon = round(float(row.get("longitude", 0) or 0))
        return f"grid_{lat:.0f}_{lon:.0f}"

    df["fold_key"] = df.apply(_fold, axis=1)
    return meta["station_id"].map(df.set_index("station_id")["fold_key"])


def _assign_splits(fold_keys: pd.Series,
                   train_frac=0.80, val_frac=0.10, seed=42) -> pd.Series:
    """
    Assign train/val/test splits by fold key (not sample) to prevent leakage.
    """
    unique_folds = np.array(sorted(fold_keys.dropna().unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_folds)

    n = len(unique_folds)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    fold_split = {}
    for i, f in enumerate(unique_folds):
        if i < n_train:
            fold_split[f] = "train"
        elif i < n_train + n_val:
            fold_split[f] = "val"
        else:
            fold_split[f] = "test"

    return fold_keys.map(fold_split).fillna("train")


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_feature_tables(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for yr in range(start_year, end_year + 1):
        p = FEATURE_DIR / f"feature_table_SE_{yr}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            logger.warning(f"  feature_table_SE_{yr}.parquet not found — skipping year")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["station_id"] = df["station_id"].astype(str)
    return df


def _load_atlas() -> pd.DataFrame:
    if not ATLAS_PATH.exists():
        logger.warning("HydroATLAS not found — spatial split will use grid cells only")
        return pd.DataFrame(columns=["station_id", "hydro_source", "Hylak_id"])
    df = pd.read_parquet(ATLAS_PATH,
                         columns=["station_id", "hydro_source", "Hylak_id"])
    df["station_id"] = df["station_id"].astype(str)
    return df


# ── Core assembly ─────────────────────────────────────────────────────────────

def build(start_year: int, end_year: int, force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = OUT_DIR / "metadata.parquet"
    if meta_path.exists() and not force:
        logger.info("metadata.parquet already exists — use --force to rebuild")
        return

    # ── 1. Load feature tables ──
    logger.info(f"Loading feature tables {start_year}–{end_year}")
    ft = _load_feature_tables(start_year, end_year)
    if ft.empty:
        logger.error("No feature tables found — run build_feature_table.py first")
        return
    logger.info(f"  {len(ft):,} WQ samples across {ft['station_id'].nunique():,} stations")

    # ── 2. Build HDF5 lookup indices ──
    logger.info("Building HDF5 lookup indices")
    with h5py.File(HDF5_PATH, "r") as f:
        h5_stations = f["station_ids"][:].astype(str)
        h5_dates    = f["dates"][:].astype(str)
        channels    = f["channels"][:].astype(str)

    station_to_idx = {s: i for i, s in enumerate(h5_stations)}
    date_to_idx    = {d: i for i, d in enumerate(h5_dates)}
    n_channels     = len(channels)

    # ── 3. Filter samples to those in HDF5 ──
    ft = ft[ft["station_id"].isin(station_to_idx)].copy()
    ft = ft[ft["date"].isin(date_to_idx)].copy()
    # Sort by station_id so each station's samples are contiguous — enables slice writes to zarr
    ft = ft.sort_values(["station_id", "date"]).reset_index(drop=True)
    ft["sample_id"] = ft.index
    logger.info(f"  {len(ft):,} samples have matching ERA5 patches")

    if ft.empty:
        logger.error("No samples matched — check station_ids and dates align with HDF5")
        return

    # ── 4. Spatial fold keys + train/val/test split ──
    logger.info("Assigning spatial fold keys and train/val/test splits")
    atlas = _load_atlas()

    # lat/lon come from feature table (SLU MVM); atlas adds Hylak_id + hydro_source
    ft_with_coords = ft.merge(
        atlas[["station_id", "Hylak_id", "hydro_source"]].drop_duplicates("station_id"),
        on="station_id", how="left",
    )

    # SMHI drainage basin spatial join for river stations
    stations_for_basin = ft_with_coords[["station_id", "latitude", "longitude"]].dropna().drop_duplicates("station_id")
    basin_lookup = _assign_smhi_basins(stations_for_basin)

    ft["fold_key"] = _build_fold_keys(ft_with_coords, atlas, basin_lookup)
    ft["split"]    = _assign_splits(ft["fold_key"])

    split_counts = ft["split"].value_counts().to_dict()
    logger.info(f"  Spatial split: {split_counts}")

    # ── 5. Temporal and latitudinal split indices ──
    ft["date_dt"] = pd.to_datetime(ft["date"])
    ft["temporal_split"] = ft["date_dt"].dt.year.apply(
        lambda y: "train" if y <= 2020 else "test"
    )

    lat_col = "latitude" if "latitude" in ft_with_coords.columns else None
    if lat_col:
        ft = ft.merge(
            ft_with_coords[["station_id", "latitude"]].drop_duplicates("station_id"),
            on="station_id", how="left", suffixes=("", "_geo"),
        )
        lat_series = ft.get("latitude_geo", ft.get("latitude"))
        ft["lat_split"] = lat_series.apply(
            lambda v: "south" if pd.notna(v) and float(v) < 61.0 else "north"
        )
        # Drop temporary geo lat if added
        ft = ft.drop(columns=["latitude_geo"], errors="ignore")
    else:
        ft["lat_split"] = "unknown"

    ft = ft.drop(columns=["date_dt"])

    logger.info(f"  Temporal split train/test: {ft['temporal_split'].value_counts().to_dict()}")
    logger.info(f"  Lat split south/north:     {ft['lat_split'].value_counts().to_dict()}")

    # ── 6. Write channel index ──
    channel_index = []
    for i, c in enumerate(channels):
        ch = str(c)
        norm = ERA5_CHANNEL_NORM.get(ch)
        entry = {
            "index": i, "name": ch,
            "description": ERA5_CHANNEL_DESC.get(ch, ""),
            "norm_offset": norm[0] if norm else None,
            "norm_scale":  norm[1] if norm else None,
            "corrupted":   norm is None,
        }
        channel_index.append(entry)
    (OUT_DIR / "channel_index.json").write_text(
        json.dumps(channel_index, indent=2), encoding="utf-8"
    )

    # ── 7. Create zarr stores ──
    n_samples = len(ft)
    logger.info(f"Creating zarr stores: {n_samples:,} samples × (12, 12, {n_channels})")

    _blosc = BloscCompressor(cname="zstd", clevel=3)
    patches_store = zarr.open(
        str(OUT_DIR / "patches.zarr"), mode="w",
        shape=(n_samples, 12, 12, n_channels),
        chunks=(512, 12, 12, n_channels),
        dtype="float32",
        compressor=_blosc,
    )
    masks_store = zarr.open(
        str(OUT_DIR / "masks.zarr"), mode="w",
        shape=(n_samples, 12, 12),
        chunks=(512, 12, 12),
        dtype="uint8",
        compressor=_blosc,
    )

    # ── 8. Extract patches station by station ──
    logger.info("Extracting ERA5 patches from HDF5")
    n_written = 0

    with h5py.File(HDF5_PATH, "r") as f:
        h5_patch = f["patch"]   # (18743, 9497, 12, 12, 9)  float16
        h5_mask  = f["mask"]    # (18743, 9497, 12, 12)      uint8

        for station_id, grp in ft.groupby("station_id", sort=False):
            st_idx = station_to_idx.get(station_id)
            if st_idx is None:
                continue

            sample_ids  = grp["sample_id"].values   # contiguous after sort above
            date_strs   = grp["date"].values
            date_idxs   = np.array([date_to_idx[d] for d in date_strs])

            # HDF5 requires strictly increasing unique indices
            unique_idxs, inverse = np.unique(date_idxs, return_inverse=True)
            patch_data = h5_patch[st_idx, unique_idxs, :, :, :].astype("float32")[inverse]
            mask_data  = h5_mask[st_idx,  unique_idxs, :, :][inverse]

            # Per-channel normalization: (raw - offset) / scale → ~[-3, 3].
            # Channels that overflowed float16 (sp, ssrd) are zeroed out.
            for ci, ch_entry in enumerate(channel_index):
                if ch_entry["corrupted"]:
                    patch_data[:, :, :, ci] = 0.0
                else:
                    raw = patch_data[:, :, :, ci]
                    normed = (raw - ch_entry["norm_offset"]) / ch_entry["norm_scale"]
                    patch_data[:, :, :, ci] = np.where(
                        np.isfinite(normed), normed, 0.0
                    ).astype("float32")

            # Use contiguous slice writes — avoids zarr v3 fancy-indexing async bugs
            s0, s1 = int(sample_ids[0]), int(sample_ids[-1]) + 1
            patches_store[s0:s1] = patch_data
            masks_store[s0:s1]   = mask_data
            n_written += len(sample_ids)

        if n_written % 5000 < len(grp):
            logger.info(f"  {n_written:,} / {n_samples:,} patches written")

    logger.info(f"  Done — {n_written:,} patches written")

    # ── 9. Write metadata and split index ──
    meta_cols = (["sample_id", "station_id", "date", "fold_key", "split",
                  "temporal_split", "lat_split"]
                 + [c for c in WQ_TARGETS if c in ft.columns]
                 + ["pre_wfd_method", "quality_summary"]
                 + [c for c in ft.columns if c.startswith("era5_")]
                 + [c for c in ft.columns if c.startswith("mesan_")])
    meta_cols = [c for c in meta_cols if c in ft.columns]
    meta_cols = list(dict.fromkeys(meta_cols))  # deduplicate, preserve order

    ft[meta_cols].to_parquet(OUT_DIR / "metadata.parquet", index=False)
    logger.info(f"Saved → {OUT_DIR / 'metadata.parquet'}")

    ft[["sample_id", "fold_key", "split", "temporal_split", "lat_split"]].to_parquet(
        OUT_DIR / "split_index.parquet", index=False
    )
    logger.info(f"Saved → {OUT_DIR / 'split_index.parquet'}")

    logger.info(f"\n{'='*50}")
    logger.info(f"Multimodal dataset complete:")
    logger.info(f"  Samples    : {n_samples:,}")
    logger.info(f"  Patch dim  : (n_samples, 12, 12, {n_channels})  float32")
    logger.info(f"  Channels   : {list(channels)}")
    logger.info(f"  Spatial    : {split_counts}")
    logger.info(f"  Temporal   : {ft['temporal_split'].value_counts().to_dict()}")
    logger.info(f"  Latitudinal: {ft['lat_split'].value_counts().to_dict()}")
    logger.info(f"  Output     : {OUT_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build multimodal ERA5-patch + WQ dataset")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year",   type=int, default=2025)
    parser.add_argument("--force",      action="store_true",
                        help="Overwrite existing output")
    args = parser.parse_args()
    build(args.start_year, args.end_year, force=args.force)


if __name__ == "__main__":
    main()
