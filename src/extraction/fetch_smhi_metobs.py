"""
SMHI MetObs Extractor — ALL Swedish meteorological station observations.

Fetches ALL available parameters dynamically — no pre-selection. Post-processing
determines which are relevant for water quality.

API base: https://opendata-download-metobs.smhi.se/api/version/1.0
Archive:  CSV only for corrected-archive period (not JSON/XML).

Output columns are named:  p{param_id}_{sanitised_title}
  e.g. p2_lufttemperatur, p5_nederbordsmangd, p8_snojup

Each column also gets a corresponding quality column: p{id}_{name}_quality

Known important params (for reference — all are fetched):
  2  Air temperature daily mean (°C)
  3  Wind direction (degrees)
  4  Wind speed (m/s)
  5  Precipitation daily total (mm)   ← use this not 14 (14=15-min)
  6  Relative humidity (%)            ← use this not 29 (29=cloud layer)
  8  Snow depth (m)
  9  Pressure sea level (hPa)
  10 Sunshine duration (s)
  11 Global radiation (W/m²)
  16 Total cloud cover (%)
  19 Air temp daily minimum (°C)
  20 Air temp daily maximum (°C)
  21 Wind gust (m/s)

Period semantics:
  Precipitation/snow = 06:00–06:00 UTC accumulation window
  Hourly params timestamp = END of aggregation window

Usage:
    python -m src.extraction.fetch_smhi_metobs --start-year 2000 --end-year 2025
    python -m src.extraction.fetch_smhi_metobs --start-year 2023 --end-year 2023
"""

import sys
import io
import re
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import RAW_DIR
from common.logging_utils import setup_logger

logger = setup_logger("SMHIMetObs")

BASE_URL   = "https://opendata-download-metobs.smhi.se/api/version/1.0"
OUTPUT_DIR = RAW_DIR / "smhi_metobs"


def _get_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _col_name(param_id: int, title: str) -> str:
    """Make a safe column name from param ID + title."""
    safe = re.sub(r"[^a-z0-9]+", "_", title.lower().split(":")[0].strip())
    safe = safe.strip("_")[:40]
    return f"p{param_id}_{safe}"


def discover_all_parameters(session: requests.Session) -> List[Dict]:
    """
    Fetch all available MetObs parameters at runtime.
    Returns list of dicts: {id, title, unit, col_name}.
    """
    url = f"{BASE_URL}.json"
    r = session.get(url, timeout=30)
    r.encoding = "utf-8"
    r.raise_for_status()
    resources = r.json().get("resource", [])

    params = []
    for res in resources:
        try:
            pid = int(res.get("key", 0))
        except (ValueError, TypeError):
            continue
        title = res.get("title", f"param_{pid}")
        unit  = res.get("unit", "")
        params.append({
            "id":       pid,
            "title":    title,
            "unit":     unit,
            "col_name": _col_name(pid, title),
        })

    logger.info(f"Discovered {len(params)} MetObs parameters")
    return params


def get_stations(session: requests.Session, param_id: int) -> List[Dict]:
    """Fetch station list for a MetObs parameter."""
    url = f"{BASE_URL}/parameter/{param_id}.json"
    try:
        r = session.get(url, timeout=30)
        r.encoding = "utf-8"
        r.raise_for_status()
        stations_raw = r.json().get("station", [])
    except Exception as exc:
        logger.warning(f"  Station list failed param {param_id}: {exc}")
        return []

    stations = []
    for st in stations_raw:
        lat = st.get("latitude")
        lon = st.get("longitude")
        # Fall back to position array if direct lat/lon missing
        if lat is None:
            positions = st.get("position", [])
            if positions:
                latest = sorted(positions, key=lambda p: p.get("from", 0))[-1]
                lat = latest.get("latitude")
                lon = latest.get("longitude")
        stations.append({
            "station_id":         str(st.get("id") or st.get("key", "")),
            "name":               st.get("name", ""),
            "owner":              st.get("owner", ""),
            "measuring_stations": st.get("measuringStations", ""),
            "active":             st.get("active", True),
            "latitude":           lat,
            "longitude":          lon,
            "from_ms":            st.get("from"),
            "to_ms":              st.get("to"),
            "country":            "SE",
            "param_id":           param_id,
        })
    return stations


def fetch_station_csv(
    session: requests.Session,
    param_id: int,
    station_id: str,
    start_year: int,
    end_year: int,
) -> Optional[pd.DataFrame]:
    """
    Download corrected-archive CSV for one station/parameter.
    Returns DataFrame [datetime_utc, value, quality] or None.
    """
    url = (
        f"{BASE_URL}/parameter/{param_id}"
        f"/station/{station_id}"
        f"/period/corrected-archive/data.csv"
    )
    try:
        r = session.get(url, timeout=120)
        r.encoding = "utf-8-sig"
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        logger.debug(f"  HTTP {station_id}/p{param_id}: {e}")
        return None
    except Exception as exc:
        logger.debug(f"  Fetch failed {station_id}/p{param_id}: {exc}")
        return None

    lines = r.text.splitlines()
    header_idx = next(
        (i for i, l in enumerate(lines) if l.strip().lower().startswith("datum")),
        None,
    )
    if header_idx is None:
        return None

    try:
        df = pd.read_csv(
            io.StringIO("\n".join(lines[header_idx:])),
            sep=";", decimal=".", on_bad_lines="skip", index_col=False,
            low_memory=False, dtype=str,
        )
    except Exception:
        return None

    if df.empty or len(df.columns) < 2:
        return None

    cols     = list(df.columns)
    date_col = cols[0]
    val_col  = cols[1]
    qual_col = cols[2] if len(cols) > 2 else None

    # Detect hourly format: "Datum;Tid [UTC];Value;Quality"
    time_col = None
    if len(cols) > 2:
        c1 = cols[1].strip().lower()
        if "tid" in c1 or "time" in c1 or "utc" in c1:
            time_col = cols[1]
            val_col  = cols[2] if len(cols) > 2 else None
            qual_col = cols[3] if len(cols) > 3 else None

    if val_col is None:
        return None

    try:
        if time_col:
            df["datetime_utc"] = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                format="mixed", dayfirst=False, errors="coerce", utc=True,
            )
        else:
            df["datetime_utc"] = pd.to_datetime(
                df[date_col].astype(str).str[:10],
                format="%Y-%m-%d", errors="coerce", utc=True,
            )
    except Exception:
        return None

    df = df.dropna(subset=["datetime_utc"])
    df = df[
        (df["datetime_utc"].dt.year >= start_year)
        & (df["datetime_utc"].dt.year <= end_year)
    ].copy()

    if df.empty:
        return None

    df["value"] = pd.to_numeric(
        df[val_col].astype(str).str.replace(",", ".").str.strip(),
        errors="coerce",
    )
    df["quality"] = df[qual_col].fillna("G").astype(str).str.strip() if qual_col else "G"

    return df[["datetime_utc", "value", "quality"]].dropna(subset=["value"])


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily mean (worst quality flag retained)."""
    df = df.copy()
    df["date"] = df["datetime_utc"].dt.date

    def worst_flag(flags):
        f = set(flags.fillna("G").apply(str))
        return "R" if "R" in f else ("Y" if "Y" in f else "G")

    return (
        df.groupby("date")
        .agg(value=("value", "mean"), quality=("quality", worst_flag))
        .reset_index()
    )


class SMHIMetObsExtractor:

    def __init__(self):
        self.session = _get_session()

    def run(
        self,
        param_ids: Optional[List[int]] = None,
        start_year: int = 2000,
        end_year:   int = 2025,
    ) -> None:
        """
        Fetch ALL MetObs parameters (or specified subset).
        Saves per-year Parquet with one column per parameter.
        Column naming: p{id}_{title} so all data is preserved for post-processing.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Discover all parameters
        all_params = discover_all_parameters(self.session)
        if param_ids:
            all_params = [p for p in all_params if p["id"] in param_ids]

        logger.info(f"Fetching {len(all_params)} parameters, {start_year}-{end_year}")

        # Save parameter catalogue for post-processing reference
        cat_path = OUTPUT_DIR / "smhi_metobs_parameters.csv"
        pd.DataFrame(all_params).to_csv(cat_path, index=False, encoding="utf-8")
        logger.info(f"Saved parameter catalogue -> {cat_path}")

        # (station_id, date) -> {col: value, col_quality: flag}
        obs: Dict[tuple, Dict] = {}
        station_meta: Dict[str, Dict] = {}

        for param in all_params:
            pid      = param["id"]
            col      = param["col_name"]
            qual_col = col + "_quality"
            logger.info(f"=== Param {pid}: {param['title'][:50]} | {param['unit']} -> {col} ===")

            stations = get_stations(self.session, pid)
            logger.info(f"  Stations: {len(stations)}")

            for st in stations:
                sid = st["station_id"]
                if not sid:
                    continue
                if sid not in station_meta:
                    station_meta[sid] = {
                        "station_id":         sid,
                        "name":               st["name"],
                        "owner":              st["owner"],
                        "measuring_stations": st["measuring_stations"],
                        "active":             st["active"],
                        "latitude":           st["latitude"],
                        "longitude":          st["longitude"],
                        "country":            "SE",
                        "source":             "smhi_metobs",
                    }

                df = fetch_station_csv(self.session, pid, sid, start_year, end_year)
                if df is None:
                    time.sleep(0.1)
                    continue

                # Always aggregate to daily (daily params: already daily; hourly: take mean)
                is_daily = df["datetime_utc"].dt.hour.nunique() <= 2
                if not is_daily:
                    daily = aggregate_to_daily(df)
                else:
                    daily = df.copy()
                    daily["date"] = daily["datetime_utc"].dt.date

                for _, row in daily.iterrows():
                    d   = row.get("date") or row["datetime_utc"].date()
                    key = (sid, d)
                    if key not in obs:
                        obs[key] = {}
                    if col not in obs[key]:
                        obs[key][col]      = row["value"]
                        obs[key][qual_col] = row["quality"]

                time.sleep(0.4)

        # Save station metadata
        if station_meta:
            pd.DataFrame(list(station_meta.values())).to_csv(
                OUTPUT_DIR / "smhi_metobs_stations.csv", index=False, encoding="utf-8"
            )
            logger.info(f"Saved {len(station_meta)} stations")

        # Save per-year Parquet — all params in wide format
        for year in range(start_year, end_year + 1):
            rows = []
            for (sid, d), cols in obs.items():
                d_year = d.year if hasattr(d, "year") else pd.Timestamp(d).year
                if d_year != year:
                    continue
                row = {
                    "station_id":         sid,
                    "date":               d,
                    "latitude":           station_meta.get(sid, {}).get("latitude"),
                    "longitude":          station_meta.get(sid, {}).get("longitude"),
                    "measuring_stations": station_meta.get(sid, {}).get("measuring_stations", ""),
                    "source":             "smhi_metobs",
                    "country":            "SE",
                }
                row.update(cols)
                rows.append(row)

            if not rows:
                logger.warning(f"No data for {year}.")
                continue

            year_df = (
                pd.DataFrame(rows)
                .sort_values(["station_id", "date"])
                .reset_index(drop=True)
            )
            year_df["date"] = pd.to_datetime(year_df["date"])
            out = OUTPUT_DIR / f"smhi_metobs_obs_{year}.parquet"
            year_df.to_parquet(out, index=False)
            logger.info(
                f"Saved {len(year_df)} station-days for {year} "
                f"({year_df['station_id'].nunique()} stations, "
                f"{len([c for c in year_df.columns if c.startswith('p')])} param cols) -> {out}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Download ALL SMHI MetObs parameters (Sweden) — filter in post-processing"
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year",   type=int, default=2025)
    parser.add_argument(
        "--params", type=int, nargs="+", default=None,
        help="Restrict to specific param IDs (default: all discovered)",
    )
    args = parser.parse_args()
    SMHIMetObsExtractor().run(args.params, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
