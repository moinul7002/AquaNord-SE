#!/usr/bin/env python3
"""
Figure 2 — ERA5 physical consistency evaluation.

Tests whether ERA5-reanalysis variables are physically consistent with
independent in-situ observations across Sweden.  Four panels covering
the two primary WQ-relevant ERA5 variables (t2m, tp):

  A. ERA5 2 m air temperature vs SMHI MetObs (accuracy validation)
  B. ERA5 precipitation seasonal cycle vs water colour (CDOM proxy)
     — spring/autumn runoff peak matches CDOM peak
  C. ERA5 2 m temperature vs in-situ lake water temperature
     — confirms ERA5 t2m is a valid lake-heat proxy
  D. ERA5 annual precipitation vs total phosphorus (TP)
     — shows ERA5 tp captures nutrient-loading dimension (distinct from CDOM)

Note: ERA5 lmlt_c (FLake lake mixing-layer temperature) is validated
separately in the Tier 1 fidelity figure (fig_tier1_fidelity.png).

Output: data/outputs/figures/fig2_era5_consistency.png  (300 dpi)

Usage:
    python -m src.evaluation.era5_consistency
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW = Path("data/raw")
OUT = Path("data/outputs/figures")

# ── loaders ───────────────────────────────────────────────────────────────────

def _load_era5(years=range(2000, 2026)) -> pd.DataFrame:
    frames = []
    for y in years:
        p = RAW / "era5" / f"era5_sweden_{y}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df["date"]       = pd.to_datetime(df["date"])
    df["month"]      = df["date"].dt.month
    df["year"]       = df["date"].dt.year
    df["station_id"] = df["station_id"].astype(str)
    return df


def _load_metobs(years=range(2000, 2026)) -> pd.DataFrame:
    frames = []
    for y in years:
        p = RAW / "smhi_metobs" / f"smhi_metobs_obs_{y}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year

    # Find daily mean air temperature column (parameter 2 → p2_*)
    temp_col = next((c for c in df.columns
                     if c.startswith("p2_") and "quality" not in c), None)
    if temp_col is None:
        # Fallback: any lufttemperatur that is not dew point
        temp_col = next((c for c in df.columns
                         if "lufttemperatur" in c and "dagg" not in c
                         and "quality" not in c), None)
    prec_col = next((c for c in df.columns
                     if "nederb" in c and "quality" not in c
                     and "intensitet" not in c), None)
    df = df.rename(columns={temp_col: "air_temp_c", prec_col: "precip_mm"}
                   if temp_col and prec_col else {})
    df["station_id"] = df["station_id"].astype(str)
    return df


def _load_slu_wq(years=range(2000, 2026)) -> pd.DataFrame:
    frames = []
    for y in years:
        p = RAW / "slu_mvm" / f"slu_mvm_chemistry_{y}.parquet"
        if p.exists():
            df = pd.read_parquet(
                p,
                columns=["station_id", "sampling_date", "water_temp_c",
                         "colour_mg_pt_l", "toc_mg_l", "secchi_m", "ph", "do_mg_l"],
            )
            df["year"]  = y
            df["month"] = pd.to_datetime(df["sampling_date"]).dt.month
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["station_id"] = df["station_id"].astype(str)
    return df


# ── panel helpers ─────────────────────────────────────────────────────────────

def _add_regression(ax, x, y, color="steelblue"):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xi = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xi, slope * xi + intercept, color=color, linewidth=1.5, zorder=3)
    ax.text(0.05, 0.93, f"R² = {r**2:.3f}\nn = {mask.sum():,}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))


# ── main plots ────────────────────────────────────────────────────────────────

def panel_era5_vs_metobs_temp(ax, era5: pd.DataFrame, metobs: pd.DataFrame):
    """Panel A: ERA5 t2m vs MetObs air temperature (monthly station means).

    ERA5 and MetObs use different station networks.  We match each MetObs
    station to the nearest ERA5 extraction point (SLU MVM grid cell) using
    a KD-tree spatial join, then compare monthly means at matched pairs.
    """
    if "air_temp_c" not in metobs.columns:
        ax.text(0.5, 0.5, "MetObs temp column not found", ha="center", va="center",
                transform=ax.transAxes)
        return

    from scipy.spatial import cKDTree

    # Load SLU MVM station coordinates (ERA5 extraction points)
    slu_csv = RAW / "slu_mvm" / "slu_mvm_measured_station_ids.csv"
    slu = pd.read_csv(slu_csv).dropna(subset=["latitude", "longitude"])
    slu["station_id"] = slu["station_id"].astype(str)
    era5_ids = set(era5["station_id"].unique())
    era5_stations = slu[slu["station_id"].isin(era5_ids)].drop_duplicates("station_id")

    # Unique MetObs stations with coordinates
    metobs_coords = (metobs.dropna(subset=["latitude", "longitude"])
                           .groupby("station_id")[["latitude", "longitude"]]
                           .first().reset_index())

    # KD-tree: match each MetObs station to nearest ERA5 grid cell (≤ 25 km)
    tree = cKDTree(era5_stations[["latitude", "longitude"]].values)
    dists, idxs = tree.query(metobs_coords[["latitude", "longitude"]].values, k=1)
    valid = dists < 0.25   # ~25 km in degrees
    mapping = pd.DataFrame({
        "station_id":     metobs_coords["station_id"].values[valid],
        "era5_station_id": era5_stations["station_id"].values[idxs[valid]],
    })

    # Map MetObs rows to their nearest ERA5 station, then monthly aggregate
    m = (metobs.merge(mapping, on="station_id")
               .groupby(["era5_station_id", "year", "month"])["air_temp_c"]
               .mean().reset_index()
               .rename(columns={"era5_station_id": "station_id"}))
    e = era5.groupby(["station_id", "year", "month"])["t2m_mean_c"].mean().reset_index()
    merged = e.merge(m, on=["station_id", "year", "month"])
    logger.info(f"  Panel A: {len(mapping)} MetObs→ERA5 pairs, {len(merged):,} monthly matchups")

    # Subsample for readability
    if len(merged) > 50_000:
        merged = merged.sample(50_000, random_state=42)

    ax.scatter(
        merged["air_temp_c"], merged["t2m_mean_c"],
        alpha=0.15, s=2, color="steelblue", rasterized=True,
    )
    _add_regression(ax, merged["air_temp_c"].values, merged["t2m_mean_c"].values)

    lim = (-30, 35)
    ax.plot(lim, lim, "k--", linewidth=0.8, label="1:1")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SMHI MetObs air temperature (°C)", fontsize=10)
    ax.set_ylabel("ERA5 2 m air temperature (°C)", fontsize=10)
    ax.set_title("(A) ERA5 vs. SMHI MetObs observations", fontweight="bold")
    ax.legend(fontsize=8)


def panel_colour_seasonal(ax, era5: pd.DataFrame, wq: pd.DataFrame):
    """Panel B: seasonal cycle of ERA5 tp vs in-situ water colour (CDOM proxy)."""
    # Monthly climatology
    era5_monthly = era5.groupby("month")["tp_mm"].mean()
    wq_monthly   = wq.groupby("month")["colour_mg_pt_l"].median()

    months = list(range(1, 13))
    month_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]

    ax2 = ax.twinx()
    ax.bar(months, [era5_monthly.get(m, np.nan) for m in months],
           color="steelblue", alpha=0.6, label="ERA5 mean daily precipitation")
    ax2.plot(months, [wq_monthly.get(m, np.nan) for m in months],
             "o-", color="#c0392b", linewidth=2, markersize=6,
             label="Median water colour")

    ax.set_xticks(months); ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month of year", fontsize=10)
    ax.set_ylabel("ERA5 mean daily precipitation (mm day⁻¹)", color="steelblue", fontsize=10)
    ax2.set_ylabel("Median water colour (mg Pt L⁻¹)", color="#c0392b", fontsize=10)
    ax.set_title("(B) ERA5 precipitation seasonal cycle vs. water colour (CDOM proxy)", fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")


def panel_water_vs_air_temp(ax, era5: pd.DataFrame, wq: pd.DataFrame):
    """Panel C: ERA5 t2m vs in-situ water temperature (seasonal lag structure)."""
    e_monthly = era5.groupby(["station_id", "year", "month"])["t2m_mean_c"].mean().reset_index()
    w_monthly = (
        wq.dropna(subset=["water_temp_c"])
        .groupby(["station_id", "year", "month"])["water_temp_c"].mean()
        .reset_index()
    )
    merged = e_monthly.merge(w_monthly, on=["station_id", "year", "month"])

    if len(merged) > 30_000:
        merged = merged.sample(30_000, random_state=42)

    sc = ax.scatter(
        merged["t2m_mean_c"], merged["water_temp_c"],
        c=merged["month"], cmap="hsv", alpha=0.25, s=4,
        vmin=1, vmax=12, rasterized=True,
    )
    _add_regression(ax, merged["t2m_mean_c"].values, merged["water_temp_c"].values,
                    color="#333333")

    cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Month of year", fontsize=8)
    cb.set_ticks([1, 4, 7, 10, 12])
    cb.set_ticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"])

    ax.set_xlabel("ERA5 2 m air temperature (°C)", fontsize=10)
    ax.set_ylabel("In-situ lake water temperature (°C)", fontsize=10)
    ax.set_title("(C) ERA5 vs. in-situ", fontweight="bold")


def panel_era5_tp_vs_tp(ax, era5: pd.DataFrame, wq: pd.DataFrame):
    """Panel D: ERA5 annual precipitation vs in-situ total phosphorus.

    Shows ERA5 tp captures the nutrient-loading dimension of WQ — distinct
    from the CDOM/colour dimension shown in Panel B.  High-precipitation years
    flush more P from catchments, elevating lake TP concentrations.
    """
    tp_frames = []
    for y in range(2000, 2026):
        p = RAW / "slu_mvm" / f"slu_mvm_chemistry_{y}.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p, columns=["station_id", "sampling_date", "tp_ug_l"])
                df["year"] = y
                tp_frames.append(df)
            except Exception:
                pass
    if not tp_frames:
        ax.text(0.5, 0.5, "TP data not available", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("(D) ERA5 precipitation vs. TP (pending)")
        return

    tp_data = pd.concat(tp_frames, ignore_index=True)
    tp_data["station_id"] = tp_data["station_id"].astype(str)

    era5_ann = era5.groupby(["station_id", "year"])["tp_mm"].mean().reset_index()
    tp_ann   = tp_data.groupby(["station_id", "year"])["tp_ug_l"].median().reset_index()
    merged   = era5_ann.merge(tp_ann, on=["station_id", "year"]).dropna()
    merged   = merged[(merged["tp_ug_l"] > 0) & (merged["tp_mm"] > 0)]

    if len(merged) < 50:
        ax.text(0.5, 0.5, "Insufficient matchups", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("(D) ERA5 precipitation vs. TP")
        return

    if len(merged) > 30_000:
        merged = merged.sample(30_000, random_state=42)

    merged["decade"] = (merged["year"] // 10 * 10).astype(str) + "s"
    decade_colors    = {"2000s": "#2196F3", "2010s": "#4CAF50", "2020s": "#FF5722"}
    for decade, grp in merged.groupby("decade"):
        ax.scatter(grp["tp_mm"], grp["tp_ug_l"],
                   alpha=0.2, s=3,
                   color=decade_colors.get(decade, "gray"),
                   label=decade, rasterized=True)

    _add_regression(ax, merged["tp_mm"].values,
                    np.log1p(merged["tp_ug_l"].values), color="#333333")

    ax.set_xlabel("ERA5 mean daily precipitation (mm day⁻¹)", fontsize=10)
    ax.set_ylabel("In-situ total phosphorus (µg L⁻¹)", fontsize=10)
    ax.set_yscale("log")
    ax.set_title("(D) ERA5 precipitation vs. total phosphorus",
                 fontweight="bold")
    ax.legend(fontsize=8, title="Decade", markerscale=3)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ERA5 data")
    era5 = _load_era5()

    logger.info("Loading MetObs data")
    metobs = _load_metobs()

    logger.info("Loading SLU MVM WQ data")
    wq = _load_slu_wq()

    logger.info(
        f"ERA5: {len(era5):,} rows  |  MetObs: {len(metobs):,} rows  |  "
        f"WQ: {len(wq):,} rows"
    )

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    # fig.suptitle("ERA5 Reanalysis Physical Consistency Evaluation",
    #              fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_era5_vs_metobs_temp(ax_a, era5, metobs)
    panel_colour_seasonal(ax_b, era5, wq)
    panel_water_vs_air_temp(ax_c, era5, wq)
    panel_era5_tp_vs_tp(ax_d, era5, wq)

    out_path = OUT / "fig2_era5_consistency.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
