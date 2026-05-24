#!/usr/bin/env python3
"""
Figure 1 — Dataset coverage map.

Shows all 18,743 measured SLU MVM stations on a Sweden basemap, coloured by
data density (years with ≥1 measurement).  Lake stations are shown as circles
scaled to Lake_area; river stations as small markers.  Drainage basin boundaries
provide geographic context.

Output: data/outputs/figures/fig1_coverage_map.png  (300 dpi)

Usage:
    python -m src.evaluation.coverage_map
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW  = Path("data/raw")
OUT  = Path("data/outputs/figures")


# ── data loading ──────────────────────────────────────────────────────────────

def load_measured_stations() -> pd.DataFrame:
    """Load 18,743 measured SLU MVM stations with coordinates."""
    path = RAW / "slu_mvm" / "slu_mvm_measured_station_ids.csv"
    return pd.read_csv(path, usecols=["station_id", "latitude", "longitude"]).assign(
        station_id=lambda d: d["station_id"].astype(str)
    )


def compute_data_density() -> pd.DataFrame:
    """Count years with ≥1 measurement per station across all chemistry files."""
    files = sorted((RAW / "slu_mvm").glob("slu_mvm_chemistry_*.parquet"))
    logger.info(f"Computing data density from {len(files)} chemistry files")

    records = []
    for f in files:
        year = int(f.stem.split("_")[-1])
        df = pd.read_parquet(f, columns=["station_id"])
        ids = df["station_id"].astype(str).unique()
        records.append(pd.DataFrame({"station_id": ids, "year": year}))

    density = (
        pd.concat(records, ignore_index=True)
        .groupby("station_id")["year"]
        .agg(n_years="nunique", first_year="min", last_year="max")
        .reset_index()
    )
    return density


def load_hydroatlas() -> pd.DataFrame:
    """Load HydroATLAS for hydro_source and lake area."""
    path = RAW / "hydroatlas" / "hydroatlas_station_covariates_SE.parquet"
    cols = ["station_id", "hydro_source", "Lake_area", "Depth_avg", "Hylak_id"]
    return pd.read_parquet(path, columns=cols).assign(
        station_id=lambda d: d["station_id"].astype(str)
    )


def load_drainage_basins() -> gpd.GeoDataFrame:
    path = RAW / "smhi_hydrography" / "drainage_basins_3006.parquet"
    return gpd.read_parquet(path)


# ── coordinate projection ──────────────────────────────────────────────────────

def to_sweref99(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Project WGS84 lat/lon to SWEREF99 TM (EPSG:3006)."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
    x, y = t.transform(df["longitude"].values, df["latitude"].values)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs="EPSG:3006")


# ── plot ──────────────────────────────────────────────────────────────────────

def load_parameter_coverage() -> pd.DataFrame:
    """Count stations with each WQ parameter from chemistry parquets."""
    params = [
        ("water_temp_c",   "Water temperature"),
        ("do_mg_l",        "Dissolved oxygen"),
        ("tp_ug_l",        "Total phosphorus"),
        ("tn_ug_l",        "Total nitrogen"),
        ("colour_mg_pt_l", "Water colour (CDOM)"),
        ("turbidity_fnu",  "Turbidity"),
        ("ph",             "pH"),
        ("chla_ug_l",      "Chlorophyll-a"),
        ("secchi_m",       "Secchi depth"),
        ("toc_mg_l",       "Total organic carbon"),
        ("abs436",         "Absorbance at 436 nm"),
        ("do_mg_l",        "Dissolved oxygen"),   # already counted above; skip duplicate
    ]
    # Deduplicate
    params = list(dict.fromkeys(params))

    files = sorted(Path("data/raw/slu_mvm").glob("slu_mvm_chemistry_*.parquet"))
    all_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    all_data["station_id"] = all_data["station_id"].astype(str)

    rows = []
    total_stations = all_data["station_id"].nunique()
    for col, label in params:
        if col in all_data.columns:
            n = all_data.groupby("station_id")[col].apply(lambda x: x.notna().any()).sum()
            rows.append({"Parameter": label, "n_stations": int(n),
                         "pct": round(100 * n / total_stations, 1)})
    return pd.DataFrame(rows).sort_values("pct", ascending=True)


def make_figure(stations_gdf: gpd.GeoDataFrame, basins: gpd.GeoDataFrame) -> plt.Figure:
    # Taller figure + more vertical space between the two right panels
    fig = plt.figure(figsize=(20, 14), facecolor="white")
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1],
                  hspace=0.55, wspace=0.30,
                  left=0.06, right=0.96, top=0.94, bottom=0.06)
    ax_map = fig.add_subplot(gs[:, 0])   # map spans both rows
    ax_bar = fig.add_subplot(gs[0, 1])   # record length histogram
    ax_cov = fig.add_subplot(gs[1, 1])   # parameter coverage

    # ── basemap ──
    basins.plot(ax=ax_map, color="#f0f0f0", edgecolor="#aaaaaa", linewidth=0.5, zorder=1)

    # ── color: years of data ──
    cmap   = plt.cm.YlOrRd
    vmin, vmax = 1, stations_gdf["n_years"].quantile(0.95)
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ── river stations ──
    rivers = stations_gdf[stations_gdf["hydro_source"] == "river"]
    ax_map.scatter(
        rivers.geometry.x, rivers.geometry.y,
        c=rivers["n_years"], cmap=cmap, norm=norm,
        s=4, alpha=0.5, linewidths=0, zorder=2, label=f"River ({len(rivers):,})",
    )

    # ── lake stations — size by Lake_area ──
    lakes = stations_gdf[stations_gdf["hydro_source"] == "lake"]
    lake_size = np.clip(
        np.log1p(lakes["Lake_area"].fillna(1)) * 6, 10, 120
    )
    sc = ax_map.scatter(
        lakes.geometry.x, lakes.geometry.y,
        c=lakes["n_years"], cmap=cmap, norm=norm,
        s=lake_size, alpha=0.8, linewidths=0.3, edgecolors="white",
        zorder=3, label=f"Lake ({len(lakes):,})",
    )

    # ── colorbar ──
    cb = plt.colorbar(sc, ax=ax_map, fraction=0.025, pad=0.02)
    cb.set_label("Record length (years)", fontsize=11)

    ax_map.set_title("(A) Station coverage map", fontsize=12, fontweight="bold")
    ax_map.set_xlabel("Easting (SWEREF99 TM, m)", fontsize=10)
    ax_map.set_ylabel("Northing (SWEREF99 TM, m)", fontsize=10)
    # Legend in lower right — empty Baltic Sea area
    ax_map.legend(loc="lower right", fontsize=9, framealpha=0.88,
                  edgecolor="#aaaaaa")
    ax_map.tick_params(labelsize=8)

    # ── top-right: record length histogram ──
    ax_bar.hist(
        stations_gdf["n_years"], bins=26, range=(0, 26),
        color="#e06c00", edgecolor="white", linewidth=0.5,
    )
    ax_bar.set_xlabel("Record length (years with ≥1 measurement)", fontsize=10)
    ax_bar.set_ylabel("Number of stations", fontsize=10)
    ax_bar.set_title("(B) Record length distribution", fontsize=11, fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(labelsize=9)

    med = stations_gdf["n_years"].median()
    ax_bar.axvline(med, color="#333333", linestyle="--", linewidth=1.5,
                   label=f"Median = {med:.0f} yr")
    ax_bar.legend(fontsize=9)

    # ── bottom-right: WQ parameter availability ──
    try:
        cov = load_parameter_coverage()
        from matplotlib.patches import Patch

        colors_cov = ["#2196F3" if p >= 50 else "#FF9800" if p >= 20 else "#F44336"
                      for p in cov["pct"]]
        bars = ax_cov.barh(cov["Parameter"], cov["pct"], color=colors_cov,
                           edgecolor="white", linewidth=0.5, height=0.65)

        # Place percentage labels:
        #   — inside bar (white) when bar ≥ 30 % (enough room)
        #   — outside bar (black) when bar < 30 % (bar too short)
        for bar, pct in zip(bars, cov["pct"]):
            cx = bar.get_cy = bar.get_y() + bar.get_height() / 2
            if pct >= 30:
                ax_cov.text(bar.get_width() - 1.5, cx,
                            f"{pct:.0f}%", va="center", ha="right",
                            fontsize=8, color="white", fontweight="bold")
            else:
                ax_cov.text(bar.get_width() + 1.0, cx,
                            f"{pct:.0f}%", va="center", ha="left",
                            fontsize=8, color="#333333")

        # Y-axis: smaller font, enough padding
        ax_cov.tick_params(axis="y", labelsize=9, pad=4)
        ax_cov.tick_params(axis="x", labelsize=9)
        ax_cov.set_xlim(0, 108)   # capped so labels never clip
        ax_cov.axvline(50, color="gray", lw=0.8, linestyle="--", alpha=0.5,
                       label="50% threshold")
        ax_cov.set_xlabel("Stations with ≥1 measurement (%)", fontsize=10)
        ax_cov.set_title("(C) WQ parameter availability", fontsize=11,
                          fontweight="bold", pad=8)
        ax_cov.legend(handles=[
            Patch(color="#2196F3", label="≥50% of stations"),
            Patch(color="#FF9800", label="20–50% of stations"),
            Patch(color="#F44336", label="<20% of stations"),
        ], fontsize=8, loc="lower right", framealpha=0.85)
        ax_cov.spines["top"].set_visible(False)
        ax_cov.spines["right"].set_visible(False)

    except Exception as exc:
        logger.warning(f"Parameter coverage panel failed: {exc}")
        ax_cov.text(0.5, 0.5, "Parameter coverage\n(run after chemistry loaded)",
                    ha="center", va="center", transform=ax_cov.transAxes)

    # No tight_layout — manual GridSpec geometry controls spacing
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading measured stations")
    stations = load_measured_stations()

    logger.info("Computing data density")
    density = compute_data_density()

    logger.info("Loading HydroATLAS")
    atlas = load_hydroatlas()

    logger.info("Loading drainage basins")
    basins = load_drainage_basins()

    # Merge
    df = (
        stations
        .merge(density, on="station_id", how="left")
        .merge(atlas[["station_id", "hydro_source", "Lake_area"]], on="station_id", how="left")
    )
    df["n_years"]      = df["n_years"].fillna(0).astype(int)
    df["hydro_source"] = df["hydro_source"].fillna("none")
    df["Lake_area"]    = df["Lake_area"].fillna(0)

    logger.info(f"Stations: {len(df):,}  |  hydro_source: {df['hydro_source'].value_counts().to_dict()}")

    stations_gdf = to_sweref99(df)

    logger.info("Generating coverage map")
    fig = make_figure(stations_gdf, basins)

    out_path = OUT / "fig1_coverage_map.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
