"""
SLU Miljödata MVM Extractor — Swedish inland water chemistry.

Fetches observed water quality chemistry from the SLU Miljödata MVM REST API v2.
This is the Swedish national data host for lakes and watercourses — the direct
equivalent of Syke VESLA for Finland.

API base:  https://miljodata.slu.se/api/observations-service/v2
Auth:      token query parameter (public ticket, no approval needed)
Docs:      https://miljodata.slu.se/api/docs/index.html

Data:
  - Water chemistry in lakes (Sjö) and rivers (Vattendrag)
  - Parameters: TP, TN, NO3, PO4, DOC, pH, alkalinity, conductivity,
    colour, turbidity, chlorophyll-a, dissolved oxygen, temperature, metals
  - Quality flag: qualityCode "A" = authorized/approved data
  - Coordinates: SWEREF99 TM (EPSG:3006) → converted to WGS84

Token in .env as SLU_TOKEN (public ticket valid until 2026-10-31). No rate limits found, but we use retries and 1s delay between years just in case.

Usage:
    python -m src.extraction.fetch_slu_mvm --start-year 2000 --end-year 2025

Output:
    - data/raw/slu_mvm/slu_mvm_stations.csv: station metadata (ID, name, type, location)
    - data/raw/slu_mvm/slu_mvm_obs_YYYY.parquet: chemistry samples for year YYYY, one row per sample, columns for each parameter + quality + unit
"""

import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pyproj import Transformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import RAW_DIR
from common.logging_utils import setup_logger

load_dotenv()
logger = setup_logger("SLUMVM")

BASE_URL   = "https://miljodata.slu.se/api/observations-service/v2"
OUTPUT_DIR = RAW_DIR / "slu_mvm"

# Medium types for inland water bodies
INLAND_MEDIUM_TYPES = {"Sjö", "Vattendrag"}

# Property codes we want to extract (from MVM vocabulary)
# propertyCode -> output column name
PROPERTY_MAP = {
    # Nutrients — verified codes from API 2026-05-07
    "Tot_P":               "tp_ug_l",          # was: Ptot (wrong)
    "Tot_N_TNb":           "tn_ug_l",          # was: Ntot (wrong); TNb = total bound N
    "Tot_N_ps":            "tn_ps_ug_l",       # parallel sample method (store separately)
    "NO2_NO3_N":           "no3_ug_l",         # was: NO3 (wrong)
    "NO3_N":               "no3_n_ug_l",       # nitrate only
    "NH4_N":               "nh4_ug_l",         # was: NH4 (wrong)
    "PO4_P":               "po4_ug_l",         # was: PO4 (wrong)
    # Organic / optical
    "DOC":                 "doc_mg_l",
    "TOC":                 "toc_mg_l",
    "Farg":                "colour_mg_pt_l",   # correct already
    "Abs_F420":            "abs420",            # was: Abs420 (wrong)
    "Abs_436m":            "abs436",            # was: Abs436 (wrong)
    "Abs_F254":            "abs254",
    "Abs_F365":            "abs365",
    # Physical / general
    "pH":                  "ph",
    "Temp":                "water_temp_c",
    "O2":                  "do_mg_l",
    "SaO2":                "do_sat_pct",        # was: O2_sat (wrong)
    "Kond_25":             "conductivity_ms_m", # was: Kond (wrong)
    "Alk_Acid":            "alkalinity_mekv_l",
    "Alk":                 "alkalinity2_mekv_l",
    # Clarity / biology
    "Turb_FNU":            "turbidity_fnu",     # was: Turb (wrong)
    "Turb_NTU":            "turbidity_ntu",
    "Kfyll":               "chla_ug_l",         # was: Klorofyll (wrong) — Kfyll = Klorofyll
    "Siktdjup_kikare":     "secchi_m",          # with binocular; primary
    "Siktdjup_utan_kikare":"secchi_m_no_bino",  # without binocular
    "Siktdjup":            "secchi_m_generic",
    "Slamhalt":            "ss_mg_l",           # suspended solids; was: SS (wrong)
    "BOD7":                "bod7_mg_l",
    # Major ions
    "Fe":                  "fe_ug_l",
    "Mn":                  "mn_ug_l",
    "SO4_IC":              "so4_mg_l",          # was: SO4 (wrong)
    "Cl":                  "cl_mg_l",
    "Ca":                  "ca_mg_l",
    "Mg":                  "mg_mg_l",
    "Na":                  "na_mg_l",
    "K":                   "k_mg_l",
    "Si":                  "sio2_mg_l",         # was: SiO2 (wrong)
    "Abs420":              "abs420_old",         # keep old code as fallback for pre-2005 data
}

# Transformer from SWEREF99 TM (EPSG:3006) to WGS84
_SWEREF_TO_WGS84 = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)


def _get_session(token: str) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=3.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.params = {"token": token}  # attach token to every request
    return s


def _sweref_to_wgs84(easting: float, northing: float):
    """Convert SWEREF99 TM coordinates to (lon, lat) WGS84."""
    try:
        lon, lat = _SWEREF_TO_WGS84.transform(easting, northing)
        return lon, lat
    except Exception:
        return None, None


def _parse_value(value_str: str) -> Optional[float]:
    """Parse a Swedish decimal string ('3,5') to float."""
    if not value_str or value_str in ("", "None", "null"):
        return None
    try:
        return float(str(value_str).replace(",", ".").strip())
    except (ValueError, TypeError):
        return None


def get_stations(session: requests.Session) -> pd.DataFrame:
    """
    Fetch all stations (lakes + rivers) from MVM.
    Converts SWEREF99 TM coordinates to WGS84.
    """
    logger.info("Fetching MVM station list...")
    r = session.get(f"{BASE_URL}/stations/query", timeout=120)
    r.encoding = "utf-8"
    r.raise_for_status()
    d = r.json()

    rows = []
    for st in d.get("stations", []):
        st_type = st.get("stationType", "")
        if st_type not in INLAND_MEDIUM_TYPES:
            continue

        e = st.get("stationCoordinateE") or st.get("stationCoordinateX")
        n = st.get("stationCoordinateN") or st.get("stationCoordinateY")
        lon, lat = _sweref_to_wgs84(e, n) if e and n else (None, None)

        rows.append({
            "station_id":         str(st["stationId"]),
            "national_station_id": st.get("nationalStationId", ""),
            "name":               st.get("stationName", ""),
            "station_type":       st_type,
            "eu_id":              st.get("euId", ""),
            "county":             st.get("county", ""),
            "municipality":       st.get("municipality", ""),
            "latitude":           lat,
            "longitude":          lon,
            "easting_sweref":     e,
            "northing_sweref":    n,
            "country":            "SE",
            "source":             "slu_mvm",
        })

    df = pd.DataFrame(rows)
    logger.info(f"Found {len(df)} inland WQ stations (lakes: {(df['station_type']=='Sjö').sum()}, rivers: {(df['station_type']=='Vattendrag').sum()})")
    return df


def fetch_year(
    session: requests.Session,
    year: int,
    product_type: str = "Chemistry",
    medium_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetch all samples of the given product_type for one year.

    Args:
        product_type: "Chemistry" or "Biology"
        medium_types: medium type filter (default: inland lakes + rivers)

    Returns a wide DataFrame with one row per sample. For Chemistry, columns
    come from PROPERTY_MAP. For Biology, all observations are stored as
    bio_{propertyCode} columns (biovolume, indices, species counts).
    """
    params = {
        "fromYear":    year,
        "toYear":      year,
        "productType": product_type,
    }

    logger.info(f"  Querying {product_type} samples for {year}...")
    r = session.get(f"{BASE_URL}/full-samples/query", params=params, timeout=300)
    r.encoding = "utf-8"
    r.raise_for_status()
    d = r.json()

    samples = d.get("samples", [])
    n_total = d.get("numberOfSamples", 0)
    logger.info(f"  {n_total} samples returned")

    if not samples:
        return pd.DataFrame()

    rows = []
    for s in samples:
        medium = s.get("mediumTypeName", "")
        if medium_types and medium not in medium_types:
            continue

        # Coordinate: prefer samplingSite, fall back to station
        se = s.get("samplingSiteX") or s.get("stationCoordinateE") or s.get("stationCoordinateX")
        sn = s.get("samplingSiteY") or s.get("stationCoordinateN") or s.get("stationCoordinateY")
        lon, lat = _sweref_to_wgs84(se, sn) if se and sn else (None, None)

        row = {
            "sample_id":          s.get("sampleId"),
            "station_id":         str(s.get("stationId", "")),
            "national_station_id": s.get("nationalStationId", ""),
            "station_name":       s.get("stationName", ""),
            "station_type":       medium,
            "eu_id":              s.get("stationEUID", ""),
            "county":             s.get("county", ""),
            "municipality":       s.get("municipality", ""),
            "sampling_date":      s.get("samplingDate"),
            "min_depth_m":        s.get("minDepth"),
            "max_depth_m":        s.get("maxDepth"),
            "product_name":       s.get("productName", ""),
            "subprogram":         s.get("subProgramName", ""),
            "study":              s.get("studyName", ""),
            "latitude":           lat,
            "longitude":          lon,
            "easting_sweref":     se,
            "northing_sweref":    sn,
            "country":            "SE",
            "source":             "slu_mvm",
        }

        # Extract observations
        for obs in s.get("observations", []):
            prop_code = obs.get("propertyCode", "")
            quality   = obs.get("qualityCode", "")

            if product_type == "Chemistry":
                col_name = PROPERTY_MAP.get(prop_code)
                if col_name is None:
                    continue
                for val_entry in obs.get("observationValues", []):
                    if val_entry is None:
                        continue
                    val  = _parse_value(val_entry.get("value"))
                    unit = val_entry.get("unit", "")
                    row[col_name]              = val
                    row[col_name + "_quality"] = quality
                    row[col_name + "_unit"]    = unit
                    break
            else:
                # Biology: store all observations as bio_{propertyCode}
                # Key biological parameters:
                #   Phytoplankton: biovolume (µg/L), species richness
                #   Benthic: ASPT, MISA indices
                #   Macrophytes: cover index
                #   Zooplankton: abundance
                col_name = f"bio_{prop_code}"
                for val_entry in obs.get("observationValues", []):
                    if val_entry is None:
                        continue
                    raw_val = val_entry.get("value", "")
                    # Try numeric, keep string for codes/indices
                    val = _parse_value(raw_val)
                    if val is None:
                        val = str(raw_val).strip()
                    unit = val_entry.get("unit", "")
                    row[col_name]              = val
                    row[col_name + "_quality"] = quality
                    row[col_name + "_unit"]    = unit
                    break

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["sampling_date"] = pd.to_datetime(df["sampling_date"], errors="coerce")
    # MD-8: flag pre-WFD harmonisation samples (method change 2003–2007, effects through 2007)
    df["pre_wfd_method"] = df["sampling_date"].dt.year < 2007
    return df


def run(
    start_year: int = 2000,
    end_year:   int = 2025,
    medium_types: Optional[List[str]] = None,
    product_types: Optional[List[str]] = None,
) -> None:
    """
    Full extraction: fetch Chemistry + Biology samples year by year.

    Saves two Parquet files per year:
      slu_mvm_chemistry_YYYY.parquet  — TP, TN, NO3, DOC, pH, temp, turbidity, Chl-a…
      slu_mvm_biology_YYYY.parquet    — phytoplankton, benthic, macrophytes, zooplankton…

    Physical parameters (water temperature, conductivity, turbidity) are part of
    Chemistry samples and are already captured in the chemistry output.
    """
    if medium_types is None:
        medium_types = list(INLAND_MEDIUM_TYPES)
    if product_types is None:
        product_types = ["Chemistry", "Biology"]

    token = os.environ.get("SLU_TOKEN", "PUJD93023KAS943HD")
    session = _get_session(token)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: station metadata
    stations_path = OUTPUT_DIR / "slu_mvm_stations.csv"
    if not stations_path.exists():
        stations = get_stations(session)
        if not stations.empty:
            stations.to_csv(stations_path, index=False, encoding="utf-8")
            logger.info(f"Saved {len(stations)} stations -> {stations_path}")
    else:
        logger.info(f"Station file exists, skipping: {stations_path}")

    # Step 2: fetch each product type × year
    for product_type in product_types:
        pt_label = product_type.lower()
        logger.info(f"\n{'='*50}")
        logger.info(f"Product type: {product_type}")
        logger.info(f"{'='*50}")

        for year in range(start_year, end_year + 1):
            out_path = OUTPUT_DIR / f"slu_mvm_{pt_label}_{year}.parquet"
            if out_path.exists():
                logger.info(f"  {year}: already exists, skipping.")
                continue

            logger.info(f"=== SLU MVM {product_type}: year {year} ===")
            df = fetch_year(session, year, product_type=product_type, medium_types=medium_types)

            if df.empty:
                logger.warning(f"  No {product_type} data for {year}.")
                continue

            # Biology columns mix floats with detection-limit strings (e.g. "<0,000005")
            # and categorical indices. Cast all bio_* value columns to str so PyArrow
            # can serialize them; post-processing can parse numerics where needed.
            if product_type == "Biology":
                bio_val_cols = [c for c in df.columns
                                if c.startswith("bio_") and not c.endswith(("_quality", "_unit"))]
                df[bio_val_cols] = df[bio_val_cols].astype(str).replace("nan", pd.NA)

            # Ensure coordinate columns are float64 — API returns ints/None as object
            for coord_col in ("easting_sweref", "northing_sweref", "latitude", "longitude",
                              "min_depth_m", "max_depth_m"):
                if coord_col in df.columns:
                    df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce")

            df.to_parquet(out_path, index=False)
            n_lakes  = (df["station_type"] == "Sj\xf6").sum() if "station_type" in df.columns else 0
            n_rivers = (df["station_type"] == "Vattendrag").sum() if "station_type" in df.columns else 0
            logger.info(
                f"  Saved {len(df)} {product_type} samples for {year} "
                f"(lakes: {n_lakes}, rivers: {n_rivers}) -> {out_path}"
            )
            time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Download SLU MVM water quality observations (Sweden) — Chemistry + Biology"
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year",   type=int, default=2025)
    parser.add_argument(
        "--medium-types", nargs="+", default=None,
        help="Medium types to fetch (default: Sjo Vattendrag)",
    )
    parser.add_argument(
        "--product-types", nargs="+", default=None,
        help="Product types: Chemistry Biology (default: both)",
    )
    args = parser.parse_args()
    run(args.start_year, args.end_year, args.medium_types, args.product_types)


if __name__ == "__main__":
    main()
