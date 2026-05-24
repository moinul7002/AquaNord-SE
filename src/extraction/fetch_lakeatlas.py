"""
SW-STEP-5b / FI-STEP-5b: HydroATLAS static catchment covariates for WQ stations.

Two-phase spatial join:
  Phase 1 — LakeATLAS polygon join (point-in-polygon):
    WQ stations that fall inside a lake polygon receive full lake + upstream
    catchment attributes.  hydro_source = "lake".

  Phase 2 — RiverATLAS nearest-reach join (for unmatched stations):
    Remaining stations are matched to the nearest river reach within 5 km.
    These are river/stream monitoring sites.  hydro_source = "river".

  Stations matched by neither → hydro_source = "none", all atlas cols = NaN.

Sources:
  LakeATLAS v10 — data/raw/hydrolakes/LakeATLAS_Data_v10_shp/
  RiverATLAS v10 — data/raw/hydrolakes/RiverATLAS_Data_v10_shp/
  Both downloaded from https://www.hydrosheds.org/products/hydroatlas

Unit notes (HydroATLAS Technical Documentation):
  tmp_dc_*  — tenths of deg C  -> divided by 10 on output
  ero_kh_*  — t/ha/yr x 100   -> divided by 100 on output
  dis_m3_*  — m3/yr
  lkv_mc_*  — million m3

Usage:
  python -m src.extraction.fetch_lakeatlas --country SE
  python -m src.extraction.fetch_lakeatlas --country FI
  python -m src.extraction.fetch_lakeatlas --country ALL

Output:
  data/raw/hydrolakes/hydroatlas_station_covariates_SE.parquet
  data/raw/hydrolakes/hydroatlas_station_covariates_FI.parquet
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import RAW_DIR
from common.logging_utils import setup_logger

logger = setup_logger("LAKEATLAS")

HYDROLAKES_DIR = RAW_DIR / "hydroatlas"

BBOXES = {
    "SE": (10.5, 55.0, 24.5, 69.5),
    "FI": (19.0, 59.5, 31.5, 70.1),
}

STATION_CSVS = {
    "SE": [
        RAW_DIR / "hydroobs" / "smhi_hydroobs_stations.csv",
        RAW_DIR / "slu_mvm"  / "slu_mvm_stations.csv",
    ],
    "FI": [
        RAW_DIR / "syke" / "syke_stations.csv",
    ],
}

MIN_LAKE_KM2 = 0.5        # consistent with satellite stats-mode threshold
RIVER_MAX_DIST_M = 5000   # max snapping distance for nearest-reach join

# ── Shared upstream columns (identical names in both LakeATLAS and RiverATLAS) ──
SHARED_UPSTREAM_COLS = [
    "dis_m3_pyr",   # mean annual discharge (m3/yr)
    "dis_m3_pmn",   # min monthly discharge (m3/s)
    "dis_m3_pmx",   # max monthly discharge (m3/s)
    "lkv_mc_usu",   # upstream lake volume (million m3)
    "rev_mc_usu",   # upstream reservoir volume (million m3)
    "dor_pc_pva",   # degree of regulation (%)
    "inu_pc_umx",   # max inundated area upstream (%)
    "ele_mt_uav",   # mean upstream elevation (m)
    "slp_dg_uav",   # mean upstream slope (degrees)
    "tmp_dc_uyr",   # mean annual air temp upstream (tenths degC -> /10)
    "pre_mm_uyr",   # mean annual precip upstream (mm)
    "pet_mm_uyr",   # mean annual PET upstream (mm)
    "aet_mm_uyr",   # mean annual AET upstream (mm)
    "ari_ix_uav",   # aridity index upstream
    "cmi_ix_uyr",   # climate moisture index upstream
    "snw_pc_uyr",   # annual snow cover upstream (%)
    "for_pc_use",   # forest (%)
    "crp_pc_use",   # cropland (%)
    "pst_pc_use",   # pasture (%)
    "urb_pc_use",   # urban (%)
    "wet_pc_ug1",   # wetland GLWD tier 1+2 (%)
    "wet_pc_ug2",   # wetland all types (%)
    "gla_pc_use",   # glacier (%)
    "prm_pc_use",   # permafrost (%)
    "pac_pc_use",   # protected area (%)
    "cly_pc_uav",   # clay (%)
    "slt_pc_uav",   # silt (%)
    "snd_pc_uav",   # sand (%)
    "soc_th_uav",   # soil organic carbon (kg C/m2)
    "swc_pc_uyr",   # soil water content (%)
    "kar_pc_use",   # karst (%)
    "ero_kh_uav",   # soil erosion upstream (t/ha/yr x100 -> /100)
    "ppd_pk_uav",   # population density (people/km2)
    "nli_ix_uav",   # night-light index
    "rdd_mk_uav",   # road density (m/km2)
    "hft_ix_u09",   # human footprint (2009)
    "ire_pc_use",   # irrigated area (%)
    "gdp_ud_usu",   # GDP upstream (USD million)
    "hdi_ix_vav",   # Human Development Index
]

# ── LakeATLAS-specific columns ────────────────────────────────────────────────
LAKE_SPECIFIC_COLS = [
    "Hylak_id", "Lake_name", "Country", "Lake_type",
    "Lake_area",    # km2
    "Depth_avg",    # mean depth (m)
    "Vol_total",    # total volume (km3)
    "Res_time",     # residence time (days)
    "Wshd_area",    # watershed area (km2)
    "Elevation",    # elevation (m asl)
    "Shore_dev",    # shoreline development index
    "Grand_id",     # GRanD dam ID (0 = no dam)
    "run_mm_vyr",   # local runoff (mm/yr)
    "gwt_cm_vav",   # groundwater table depth (cm)
    "sgr_dk_vav",   # stream gradient (m/km)
    "lit_cl_vmj",   # dominant lithology class
    "tmp_dc_lmn",   # min monthly temp local (tenths degC)
    "tmp_dc_lmx",   # max monthly temp local (tenths degC)
]

# ── RiverATLAS-specific columns ───────────────────────────────────────────────
RIVER_SPECIFIC_COLS = [
    "HYRIV_ID",     # HydroRIVERS reach ID
    "ORD_STRA",     # Strahler stream order
    "CATCH_SKM",    # local catchment area (km2)
    "UPLAND_SKM",   # upstream catchment area (km2)
    "DIS_AV_CMS",   # mean annual discharge (m3/s)
    "LENGTH_KM",    # reach length (km)
    "DIST_DN_KM",   # distance to river mouth (km)
    "run_mm_cyr",   # catchment runoff (mm/yr)
    "gwt_cm_cav",   # groundwater table depth (cm)
    "sgr_dk_rav",   # reach gradient (m/km)
    "lit_cl_cmj",   # dominant lithology class
    "tmp_dc_cmn",   # min monthly temp catchment (tenths degC)
    "tmp_dc_cmx",   # max monthly temp catchment (tenths degC)
]

LAKEATLAS_COLS  = LAKE_SPECIFIC_COLS  + SHARED_UPSTREAM_COLS
RIVERATLAS_COLS = RIVER_SPECIFIC_COLS + SHARED_UPSTREAM_COLS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_lake_shapefile() -> Path:
    candidates = list(HYDROLAKES_DIR.glob("**/*pol_east.shp"))
    if not candidates:
        raise FileNotFoundError(
            f"LakeATLAS polygon shapefile not found under {HYDROLAKES_DIR}."
        )
    return candidates[0]


def _find_river_shapefile() -> Path:
    # RiverATLAS is split by continent; Sweden + Finland are in Europe (_eu)
    candidates = list(HYDROLAKES_DIR.glob("**/*_eu.shp"))
    if not candidates:
        raise FileNotFoundError(
            f"RiverATLAS EU shapefile not found under {HYDROLAKES_DIR}. "
            "Expected: RiverATLAS_Data_v10_shp/RiverATLAS_v10_shp/RiverATLAS_v10_eu.shp"
        )
    return candidates[0]


def _load_stations(country: str) -> pd.DataFrame:
    frames = []
    for p in STATION_CSVS[country]:
        if p.exists():
            frames.append(pd.read_csv(p, usecols=["station_id", "latitude", "longitude"]))
        else:
            logger.warning(f"Station file not found: {p}")
    if not frames:
        raise FileNotFoundError(f"No station CSVs found for {country}.")
    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates(subset=["station_id"])
        .reset_index(drop=True)
    )


def _apply_unit_conversions(df: pd.DataFrame) -> pd.DataFrame:
    for c in [col for col in df.columns if col.startswith("tmp_dc_")]:
        df[c] = df[c] / 10.0
    for c in [col for col in df.columns if col.startswith("ero_kh_")]:
        df[c] = df[c] / 100.0
    return df


def _stations_to_gdf(stations: pd.DataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs="EPSG:4326",
    ).to_crs(crs)


# ── Phase 1: LakeATLAS polygon join ──────────────────────────────────────────

def _join_lakes(bbox: tuple, stations: pd.DataFrame) -> pd.DataFrame:
    shp = _find_lake_shapefile()
    logger.info(f"Phase 1 — LakeATLAS polygon join ({shp.name})...")

    gdf = gpd.read_file(shp, bbox=bbox)
    gdf = gdf[gdf["Lake_area"] >= MIN_LAKE_KM2].reset_index(drop=True)
    logger.info(f"  {len(gdf)} lakes >= {MIN_LAKE_KM2} km2 in bbox")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    keep = [c for c in LAKEATLAS_COLS if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]]

    stations_gdf = _stations_to_gdf(stations)
    joined = gpd.sjoin(stations_gdf, gdf, how="left", predicate="within")
    joined = joined.drop(columns=["index_right", "geometry"], errors="ignore")

    # Keep largest lake when station overlaps multiple polygons
    if "Lake_area" in joined.columns:
        joined = (
            joined.sort_values("Lake_area", ascending=False)
            .drop_duplicates(subset=["station_id"])
        )

    joined = joined.reset_index(drop=True)
    lake_cols = [c for c in keep if c in joined.columns]
    joined[lake_cols] = _apply_unit_conversions(joined[lake_cols].copy())

    n = int(joined["Hylak_id"].notna().sum())
    logger.info(f"  {n}/{len(joined)} stations matched to a lake")
    return joined.drop(columns=["latitude", "longitude"], errors="ignore")


# ── Phase 2: RiverATLAS nearest-reach join ────────────────────────────────────

def _join_rivers(bbox: tuple, stations: pd.DataFrame) -> pd.DataFrame:
    """Nearest-reach join for river/unmatched stations. Projects to EPSG:3035."""
    shp = _find_river_shapefile()
    logger.info(f"Phase 2 — RiverATLAS nearest-reach join ({shp.name})...")

    river_gdf = gpd.read_file(shp, bbox=bbox)
    logger.info(f"  {len(river_gdf)} river reaches in bbox")

    if river_gdf.crs is None:
        river_gdf = river_gdf.set_crs("EPSG:4326")
    else:
        river_gdf = river_gdf.to_crs("EPSG:4326")

    keep = [c for c in RIVERATLAS_COLS if c in river_gdf.columns]
    river_gdf = river_gdf[keep + ["geometry"]]

    # Project to metric CRS for distance-based nearest join
    proj = "EPSG:3035"
    river_proj   = river_gdf.to_crs(proj)
    stations_gdf = _stations_to_gdf(stations, crs=proj)

    joined = gpd.sjoin_nearest(
        stations_gdf,
        river_proj,
        how="left",
        max_distance=RIVER_MAX_DIST_M,
        distance_col="reach_dist_m",
    ).drop(columns=["index_right", "geometry"], errors="ignore")

    joined = joined.drop_duplicates(subset=["station_id"]).reset_index(drop=True)

    river_cols = [c for c in keep if c in joined.columns]
    joined[river_cols] = _apply_unit_conversions(joined[river_cols].copy())

    n = int(joined["HYRIV_ID"].notna().sum())
    logger.info(
        f"  {n}/{len(joined)} stations matched to a river reach "
        f"(within {RIVER_MAX_DIST_M/1000:.0f} km)"
    )
    return joined.drop(columns=["latitude", "longitude"], errors="ignore")


# ── Main extraction ───────────────────────────────────────────────────────────

def extract(country: str) -> pd.DataFrame:
    """
    Full two-phase HydroATLAS join for all WQ stations in one country.

    Returns one row per station with:
      hydro_source  — "lake" | "river" | "none"
      Lake cols     — populated for lake stations, NaN for river/none
      River cols    — populated for river stations, NaN for lake/none
      Shared cols   — populated for lake and river stations, NaN for none
    """
    bbox = BBOXES[country]
    logger.info(f"HydroATLAS extraction for {country}")

    stations = _load_stations(country)
    logger.info(f"  {len(stations)} WQ stations loaded")

    # Phase 1 — lake join
    lake_df = _join_lakes(bbox, stations)
    lake_df["hydro_source"] = lake_df["Hylak_id"].notna().map(
        {True: "lake", False: "unmatched"}
    )

    # Phase 2 — river join for unmatched stations only
    unmatched_ids = set(lake_df.loc[lake_df["hydro_source"] == "unmatched", "station_id"])
    unmatched_stations = stations[stations["station_id"].isin(unmatched_ids)].copy()

    river_df = _join_rivers(bbox, unmatched_stations)
    river_df["hydro_source"] = river_df["HYRIV_ID"].notna().map(
        {True: "river", False: "none"}
    )
    river_df["reach_dist_m"] = river_df.get("reach_dist_m", float("nan"))

    # Combine: lake rows keep lake_df, river/none rows come from river_df
    lake_matched = lake_df[lake_df["hydro_source"] == "lake"].copy()
    lake_matched["reach_dist_m"] = float("nan")

    result = pd.concat([lake_matched, river_df], ignore_index=True)

    n_lake  = int((result["hydro_source"] == "lake").sum())
    n_river = int((result["hydro_source"] == "river").sum())
    n_none  = int((result["hydro_source"] == "none").sum())
    logger.info(
        f"  Result: {n_lake} lake | {n_river} river | {n_none} unmatched "
        f"out of {len(stations)} stations"
    )
    return result


def save(df: pd.DataFrame, country: str) -> None:
    out_path = HYDROLAKES_DIR / f"hydroatlas_station_covariates_{country}.parquet"
    HYDROLAKES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} rows -> {out_path}")


def run(country: str) -> None:
    countries = ["SE", "FI"] if country == "ALL" else [country]
    for c in countries:
        if c not in BBOXES:
            logger.error(f"Unknown country '{c}'. Choose SE, FI, or ALL.")
            continue
        df = extract(c)
        save(df, c)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SW-STEP-5b: HydroATLAS static covariates (lake + river) for WQ stations"
    )
    parser.add_argument("--country", choices=["SE", "FI", "ALL"], default="SE")
    args = parser.parse_args()
    run(args.country)


if __name__ == "__main__":
    main()
