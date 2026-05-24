#!/usr/bin/env python3
"""
Compute derived WQ parameters from station data for evaluation and feature enrichment.

Derived parameters:
  1. Catchment nutrient loading flux  — TP × Q / catchment_area  (kg P / km² / yr)
  2. Ice-free growing season length   — from HydObs ice on/off dates  (days/yr)
  3. CDOM absorption α(440)           — from colour via de Wit (2005) calibration
  4. DO deficit (hypoxia risk)        — DO_sat(T) - DO_obs  (mg/L)
  5. Secchi depth from turbidity      — 1.7 / Kd; Kd = turb × 0.015 + 0.04
  6. Trophic State Index (TSI)        — Carlson (1977); needs Chl-a, TP, Secchi
  7. Water retention time (τ)         — Vol_total / mean_annual_discharge  (days)

Output:
    data/interim/SE/derived_wq_params.parquet   — one row per station
    data/interim/SE/derived_wq_timeseries.parquet — one row per (station, year)

Usage:
    python -m src.evaluation.derived_parameters
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW     = Path("data/raw")
INTERIM = Path("data/interim/SE")
OUT     = Path("data/outputs/figures")


# ── loaders ───────────────────────────────────────────────────────────────────

def load_chemistry(years=range(2000, 2026)) -> pd.DataFrame:
    frames = []
    for y in years:
        p = RAW / "slu_mvm" / f"slu_mvm_chemistry_{y}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["year"] = y
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_hydroobs_discharge() -> pd.DataFrame:
    """Load HydObs daily discharge (parameter 1) across all years."""
    frames = []
    for p in sorted((RAW / "hydroobs").glob("smhi_hydroobs_obs_*.parquet")):
        df = pd.read_parquet(p)
        # Keep only discharge (parameter 1) rows
        if "parameter_id" in df.columns:
            df = df[df["parameter_id"] == 1]
        elif "param_1_daily_discharge" in df.columns:
            df = df[["station_id", "date", "param_1_daily_discharge"]].rename(
                columns={"param_1_daily_discharge": "discharge_m3s"}
            )
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_hydroatlas() -> pd.DataFrame:
    path = RAW / "hydroatlas" / "hydroatlas_station_covariates_SE.parquet"
    cols = ["station_id", "hydro_source", "Lake_area", "Vol_total",
            "Depth_avg", "Wshd_area", "dis_m3_pyr", "for_pc_use", "crp_pc_use"]
    available = [c for c in cols
                 if c in pd.read_parquet(path, columns=["station_id"]).columns
                 or c == "station_id"]
    df = pd.read_parquet(path)
    return df[[c for c in cols if c in df.columns]].assign(
        station_id=lambda d: d["station_id"].astype(str)
    )


# ── derivations ───────────────────────────────────────────────────────────────

def do_saturation(temp_c: pd.Series) -> pd.Series:
    """Dissolved oxygen saturation (mg/L) at atmospheric pressure (Benson & Krause 1984)."""
    return 14.62 - 0.3898 * temp_c + 0.006969 * temp_c**2 - 5.596e-5 * temp_c**3


def do_pct_saturation(do_mg_l: pd.Series, temp_c: pd.Series) -> pd.Series:
    """DO % saturation — direct hypoxia indicator. < 50% = hypoxic."""
    sat = do_saturation(temp_c).replace(0, np.nan)
    return (do_mg_l / sat * 100).clip(0, 200)


def cdom_absorption_440(colour_mg_pt: pd.Series) -> pd.Series:
    """CDOM absorption at 440 nm from Pt-Co colour (de Wit et al. 2005):
    aCDOM(440) = 0.174 × colour^0.724"""
    return 0.174 * colour_mg_pt.clip(lower=0.1) ** 0.724


def cdom_e4_e6_ratio(abs420: pd.Series, abs436: pd.Series) -> pd.Series:
    """E4/E6 CDOM ratio (Abs420 / Abs436) — proxy for DOM molecular weight.
    High E4/E6 (>5): low MW, photodegraded.  Low E4/E6 (<3): high MW, terrestrial.
    Reference: Ågren et al. (2008); Chen et al. (2002)."""
    return (abs420 / abs436.replace(0, np.nan)).clip(0.5, 20)


def cdom_spectral_slope(abs420: pd.Series, abs436: pd.Series) -> pd.Series:
    """CDOM spectral slope S (nm⁻¹) from two wavelengths (420 and 436 nm).
    S = ln(A420/A436) / (436-420).
    Higher S → lower molecular weight, more photodegraded DOM.
    Reference: Helms et al. (2008) — typical range 0.01–0.025 nm⁻¹."""
    ratio = (abs420 / abs436.replace(0, np.nan)).clip(lower=1e-6)
    return (np.log(ratio) / (436 - 420)).clip(-0.05, 0.05)


def np_molar_ratio(tn_ug_l: pd.Series, tp_ug_l: pd.Series) -> pd.Series:
    """Molar N:P ratio from total nitrogen and total phosphorus.
    N_mol = TN(µg/L) / 14.007;  P_mol = TP(µg/L) / 30.974
    Redfield ratio = 16.  >16: P-limited;  10-16: co-limited;  <10: N-limited.
    Reference: Redfield (1958); Bergström & Jansson (2006)."""
    n_mol = tn_ug_l / 14.007
    p_mol = tp_ug_l.replace(0, np.nan) / 30.974
    return (n_mol / p_mol).clip(0, 500)


def nutrient_regime(np_ratio: pd.Series) -> pd.Series:
    """Nutrient limitation regime from N:P molar ratio (Bergström & Jansson 2006):
    P-limited: N:P > 16 | co-limited: 10-16 | N-limited: < 10"""
    return pd.cut(
        np_ratio,
        bins=[-np.inf, 10, 16, np.inf],
        labels=["N-limited", "co-limited", "P-limited"],
    )


def chla_tp_ratio(chla_ug_l: pd.Series, tp_ug_l: pd.Series) -> pd.Series:
    """Chl-a yield per unit TP (µg Chl-a / µg TP) — Dillon-Rigler (1974) index.
    Expected log-log slope ~1.45 (Dillon & Rigler 1974).
    Departures indicate CDOM light suppression or nutrient recycling."""
    return (chla_ug_l / tp_ug_l.replace(0, np.nan)).clip(0, 10)


def wfd_tp_status(tp_ug_l: pd.Series) -> pd.Series:
    """Approximate WFD ecological status class from total phosphorus.
    Uses simplified national boundary values (HVMFS 2019:25, type-averaged).
    Note: Type-specific thresholds vary; this is a first-order classification.
    High ≤8 | Good ≤24 | Moderate ≤42 | Poor ≤80 | Bad >80 µg/L"""
    return pd.cut(
        tp_ug_l,
        bins=[-np.inf, 8, 24, 42, 80, np.inf],
        labels=["High", "Good", "Moderate", "Poor", "Bad"],
    )


def trophic_state_index(chla_ug_l=None, tp_ug_l=None, secchi_m=None):
    """Carlson (1977) TSI — mean of available components.
    TSI < 30: oligotrophic  30–50: mesotrophic  50–70: eutrophic  >70: hypereutrophic"""
    components = []
    if chla_ug_l is not None:
        components.append(9.81 * np.log(chla_ug_l.clip(lower=0.01)) + 30.6)
    if tp_ug_l is not None:
        components.append(14.42 * np.log(tp_ug_l.clip(lower=0.1)) + 4.15)
    if secchi_m is not None:
        components.append(60 - 14.41 * np.log(secchi_m.clip(lower=0.01)))
    if not components:
        idx = (chla_ug_l if chla_ug_l is not None else
               tp_ug_l if tp_ug_l is not None else secchi_m).index
        return pd.Series(np.nan, index=idx)
    return pd.concat(components, axis=1).mean(axis=1)


def tsi_chla(chla_ug_l: pd.Series) -> pd.Series:
    """Carlson (1977) TSI from Chl-a only."""
    return 9.81 * np.log(chla_ug_l.clip(lower=0.01)) + 30.6


def tsi_tp(tp_ug_l: pd.Series) -> pd.Series:
    """Carlson (1977) TSI from TP only."""
    return 14.42 * np.log(tp_ug_l.clip(lower=0.1)) + 4.15


def tsi_secchi(secchi_m: pd.Series) -> pd.Series:
    """Carlson (1977) TSI from Secchi depth only."""
    return 60 - 14.41 * np.log(secchi_m.clip(lower=0.01))


def secchi_from_turbidity(turbidity_fnu: pd.Series) -> pd.Series:
    """Kratzer et al. (2008): Secchi = 1.7 / Kd; Kd = turb × 0.015 + 0.04"""
    kd = turbidity_fnu.clip(lower=0) * 0.015 + 0.04
    return (1.7 / kd).clip(upper=40)


# ── main derivation pipeline ──────────────────────────────────────────────────

def derive_station_parameters(chem: pd.DataFrame, atlas: pd.DataFrame) -> pd.DataFrame:
    """Compute per-station mean derived parameters across all years."""
    chem["station_id"] = chem["station_id"].astype(str)

    # ── DO parameters ────────────────────────────────────────────────────────
    if "do_mg_l" in chem.columns and "water_temp_c" in chem.columns:
        chem["do_sat"]         = do_saturation(chem["water_temp_c"])
        chem["do_deficit_mg_l"]= (chem["do_sat"] - chem["do_mg_l"]).clip(lower=0)
        chem["do_pct_sat"]     = do_pct_saturation(chem["do_mg_l"], chem["water_temp_c"])

    # ── CDOM parameters ──────────────────────────────────────────────────────
    if "colour_mg_pt_l" in chem.columns:
        chem["cdom_a440_m"] = cdom_absorption_440(chem["colour_mg_pt_l"])
    if "abs420" in chem.columns and "abs436" in chem.columns:
        chem["cdom_e4_e6"]          = cdom_e4_e6_ratio(chem["abs420"], chem["abs436"])
        chem["cdom_spectral_slope"]  = cdom_spectral_slope(chem["abs420"], chem["abs436"])

    # ── Nutrient parameters ───────────────────────────────────────────────────
    if "tp_ug_l" in chem.columns and "tn_ug_l" in chem.columns:
        chem["np_molar_ratio"] = np_molar_ratio(chem["tn_ug_l"], chem["tp_ug_l"])
    if "chla_ug_l" in chem.columns and "tp_ug_l" in chem.columns:
        chem["chla_tp_ratio"]  = chla_tp_ratio(chem["chla_ug_l"], chem["tp_ug_l"])
    if "tp_ug_l" in chem.columns:
        chem["wfd_tp_status"]  = wfd_tp_status(chem["tp_ug_l"])

    # ── Secchi from turbidity ─────────────────────────────────────────────────
    if "turbidity_fnu" in chem.columns:
        chem["secchi_from_turb"] = secchi_from_turbidity(chem["turbidity_fnu"])

    # ── TSI — individual components (needed for CDOM-offset analysis) ─────────
    if "chla_ug_l" in chem.columns:
        chem["tsi_chla"]   = tsi_chla(chem["chla_ug_l"])
    if "tp_ug_l" in chem.columns:
        chem["tsi_tp"]     = tsi_tp(chem["tp_ug_l"])
    if "secchi_m" in chem.columns:
        chem["tsi_secchi"] = tsi_secchi(chem["secchi_m"])
    tsi_components = [c for c in ["tsi_chla", "tsi_tp", "tsi_secchi"]
                      if c in chem.columns]
    if tsi_components:
        chem["tsi"] = chem[tsi_components].mean(axis=1)

    # Aggregate per station
    agg_cols = {c: "mean" for c in [
        "water_temp_c", "ph", "do_mg_l", "do_deficit_mg_l", "do_pct_sat",
        "colour_mg_pt_l", "cdom_a440_m", "cdom_e4_e6", "cdom_spectral_slope",
        "turbidity_fnu", "toc_mg_l",
        "secchi_m", "secchi_from_turb",
        "chla_ug_l", "tp_ug_l", "tn_ug_l",
        "np_molar_ratio", "chla_tp_ratio",
        "tsi", "tsi_chla", "tsi_tp", "tsi_secchi",
    ] if c in chem.columns}
    agg_cols["sample_id"] = "count"

    station_params = chem.groupby("station_id").agg(agg_cols).reset_index()
    station_params = station_params.rename(columns={"sample_id": "n_samples"})

    # Join atlas covariates
    station_params = station_params.merge(
        atlas[["station_id", "hydro_source", "Lake_area", "Depth_avg",
               "Vol_total", "Wshd_area", "dis_m3_pyr",
               "for_pc_use", "crp_pc_use"]].assign(
            station_id=lambda d: d["station_id"].astype(str)
        ),
        on="station_id", how="left"
    )

    # Water retention time (days) — Vol_total (m³) / dis_m3_pyr (m³/yr) → days
    if "Vol_total" in station_params.columns and "dis_m3_pyr" in station_params.columns:
        station_params["retention_time_days"] = (
            station_params["Vol_total"] / station_params["dis_m3_pyr"].replace(0, np.nan) * 365
        ).clip(upper=3650)

    # WFD TP status on station means (categorical from mean TP)
    if "tp_ug_l" in station_params.columns:
        station_params["wfd_tp_status"] = wfd_tp_status(
            station_params["tp_ug_l"]
        ).astype(str)

    # Nutrient regime from mean N:P ratio
    if "np_molar_ratio" in station_params.columns:
        station_params["nutrient_regime"] = nutrient_regime(
            station_params["np_molar_ratio"]
        ).astype(str)

    return station_params


def derive_annual_timeseries(chem: pd.DataFrame) -> pd.DataFrame:
    """Annual means per station — for trend analysis."""
    chem["station_id"] = chem["station_id"].astype(str)
    chem["year"] = pd.to_datetime(chem["sampling_date"]).dt.year

    if "do_mg_l" in chem.columns and "water_temp_c" in chem.columns:
        chem["do_deficit_mg_l"] = (do_saturation(chem["water_temp_c"]) - chem["do_mg_l"]).clip(lower=0)
        chem["do_pct_sat"]      = do_pct_saturation(chem["do_mg_l"], chem["water_temp_c"])
    if "colour_mg_pt_l" in chem.columns:
        chem["cdom_a440_m"] = cdom_absorption_440(chem["colour_mg_pt_l"])
    if "abs420" in chem.columns and "abs436" in chem.columns:
        chem["cdom_e4_e6"]         = cdom_e4_e6_ratio(chem["abs420"], chem["abs436"])
        chem["cdom_spectral_slope"] = cdom_spectral_slope(chem["abs420"], chem["abs436"])
    if "tp_ug_l" in chem.columns and "tn_ug_l" in chem.columns:
        chem["np_molar_ratio"] = np_molar_ratio(chem["tn_ug_l"], chem["tp_ug_l"])
    if "chla_ug_l" in chem.columns and "tp_ug_l" in chem.columns:
        chem["chla_tp_ratio"]  = chla_tp_ratio(chem["chla_ug_l"], chem["tp_ug_l"])
    if "chla_ug_l" in chem.columns:
        chem["tsi_chla"]   = tsi_chla(chem["chla_ug_l"])
    if "tp_ug_l" in chem.columns:
        chem["tsi_tp"]     = tsi_tp(chem["tp_ug_l"])
    if "secchi_m" in chem.columns:
        chem["tsi_secchi"] = tsi_secchi(chem["secchi_m"])
    tsi_cols = [c for c in ["tsi_chla", "tsi_tp", "tsi_secchi"] if c in chem.columns]
    if tsi_cols:
        chem["tsi"] = chem[tsi_cols].mean(axis=1)

    agg_cols = {c: "mean" for c in [
        "water_temp_c", "colour_mg_pt_l", "cdom_a440_m", "cdom_e4_e6",
        "cdom_spectral_slope", "turbidity_fnu", "toc_mg_l",
        "secchi_m", "chla_ug_l", "tp_ug_l", "tn_ug_l",
        "ph", "do_mg_l", "do_deficit_mg_l", "do_pct_sat",
        "np_molar_ratio", "chla_tp_ratio",
        "tsi", "tsi_chla", "tsi_tp", "tsi_secchi",
    ] if c in chem.columns}
    agg_cols["sample_id"] = "count"

    ts = chem.groupby(["station_id", "year"]).agg(agg_cols).reset_index()
    ts = ts.rename(columns={"sample_id": "n_samples"})
    return ts


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    INTERIM.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SLU MVM chemistry")
    chem = load_chemistry()
    if chem.empty:
        logger.error("No chemistry data found — run fetch_slu_mvm first")
        return

    logger.info(f"Chemistry: {len(chem):,} rows, {chem['station_id'].nunique():,} stations")

    # Check which key WQ columns are available
    key_cols = ["chla_ug_l", "tp_ug_l", "tn_ug_l", "turbidity_fnu", "secchi_m",
                "colour_mg_pt_l", "water_temp_c", "do_mg_l"]
    for col in key_cols:
        n = chem[col].notna().sum() if col in chem.columns else 0
        logger.info(f"  {col}: {n:,} non-null values")

    logger.info("Loading HydroATLAS")
    atlas = load_hydroatlas()

    logger.info("Deriving station-level parameters")
    station_params = derive_station_parameters(chem.copy(), atlas)
    out1 = INTERIM / "derived_wq_params.parquet"
    station_params.to_parquet(out1, index=False)
    logger.info(f"Saved {len(station_params):,} station records → {out1}")

    logger.info("Deriving annual time series")
    ts = derive_annual_timeseries(chem.copy())
    out2 = INTERIM / "derived_wq_timeseries.parquet"
    ts.to_parquet(out2, index=False)
    logger.info(f"Saved {len(ts):,} station-year records → {out2}")

    # Print availability summary for all derived WQ parameters
    logger.info("\nDerived WQ parameter availability:")
    derived_cols = [
        ("tsi",                 "TSI composite (Carlson 1977)"),
        ("tsi_chla",            "TSI from Chl-a"),
        ("tsi_tp",              "TSI from TP"),
        ("tsi_secchi",          "TSI from Secchi"),
        ("cdom_a440_m",         "CDOM α(440) (de Wit 2005)"),
        ("cdom_e4_e6",          "CDOM E4/E6 ratio (Ågren 2008)"),
        ("cdom_spectral_slope", "CDOM spectral slope (Helms 2008)"),
        ("do_deficit_mg_l",     "DO deficit (Benson & Krause 1984)"),
        ("do_pct_sat",          "DO % saturation"),
        ("np_molar_ratio",      "N:P molar ratio (Redfield)"),
        ("nutrient_regime",     "Nutrient regime class"),
        ("chla_tp_ratio",       "Chl:TP ratio (Dillon & Rigler 1974)"),
        ("secchi_from_turb",    "Secchi from turbidity (Kratzer 2008)"),
        ("wfd_tp_status",       "WFD TP status class (HVMFS 2019:25)"),
        ("retention_time_days", "Water retention time"),
    ]
    for col, label in derived_cols:
        if col in station_params.columns:
            n = station_params[col].notna().sum()
            pct = 100 * n / len(station_params)
            logger.info(f"  {label}: {n:,} stations ({pct:.0f}%)")


if __name__ == "__main__":
    main()
