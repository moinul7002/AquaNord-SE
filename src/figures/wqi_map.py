#!/usr/bin/env python3
"""
Figure 3 — Water quality spatial distribution map.

Computes a simplified Water Clarity Index (WCI) per lake station from
available in-situ parameters, then maps it over Sweden lake polygons.
A secondary panel shows ERA5 long-term mean precipitation overlay to
illustrate catchment forcing context.

WCI uses Secchi depth and water colour (mg Pt/L, a CDOM/humic proxy):
    clarity_score = mean_secchi_m / (1 + log1p(mean_colour_mg_pt_l))
    normalised 0–1 across all lakes: 0 = humic/turbid, 1 = clear/oligotrophic

Output: data/outputs/figures/fig3_wqi_map.png  (300 dpi)

Usage:
    python -m src.evaluation.wqi_map
"""

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import griddata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW = Path("data/raw")
OUT = Path("data/outputs/figures")


# ── data loading ──────────────────────────────────────────────────────────────

def load_wq_per_station() -> pd.DataFrame:
    """Aggregate key WQ parameters per station across all years."""
    files  = sorted((RAW / "slu_mvm").glob("slu_mvm_chemistry_*.parquet"))
    frames = []
    for f in files:
        want = ["station_id", "latitude", "longitude",
                "secchi_m", "colour_mg_pt_l", "toc_mg_l",
                "water_temp_c", "ph", "do_mg_l", "tp_ug_l", "chla_ug_l"]
        df = pd.read_parquet(f, columns=[c for c in want
                                          if c in pd.read_parquet(f, columns=["station_id"]).columns
                                          or c == "station_id"])
        # safe column read
        df = pd.read_parquet(f)
        df = df[[c for c in want if c in df.columns]]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["station_id"] = combined["station_id"].astype(str)

    agg_map = {
        "latitude":       ("latitude",        "first"),
        "longitude":      ("longitude",       "first"),
        "mean_secchi_m":  ("secchi_m",        "mean"),
        "mean_colour":    ("colour_mg_pt_l",  "mean"),
        "mean_toc":       ("toc_mg_l",        "mean"),
        "mean_temp":      ("water_temp_c",    "mean"),
        "mean_ph":        ("ph",              "mean"),
        "mean_do":        ("do_mg_l",         "mean"),
        "mean_tp":        ("tp_ug_l",         "mean"),
        "mean_chla":      ("chla_ug_l",       "mean"),
        "n_samples":      ("secchi_m",        "count"),
    }
    agg = combined.groupby("station_id").agg(
        **{k: v for k, v in agg_map.items() if v[0] in combined.columns}
    ).reset_index()
    return agg


def load_hydroatlas() -> pd.DataFrame:
    path = RAW / "hydroatlas" / "hydroatlas_station_covariates_SE.parquet"
    return pd.read_parquet(
        path,
        columns=["station_id", "hydro_source", "Lake_area", "Depth_avg", "Hylak_id"],
    ).assign(station_id=lambda d: d["station_id"].astype(str))


def load_era5_precip() -> pd.Series:
    """Long-term mean annual precipitation per station from ERA5."""
    files = sorted((RAW / "era5").glob("era5_sweden_*.parquet"))
    if not files:
        logger.warning("ERA5 Parquets not found — ERA5 re-extraction may still be running")
        return pd.Series(dtype=float, name="mean_tp_mm")
    frames = [pd.read_parquet(f, columns=["station_id", "tp_mm"]) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("station_id")["tp_mm"].mean().rename("mean_tp_mm")


def load_drainage_basins() -> gpd.GeoDataFrame:
    return gpd.read_parquet(RAW / "smhi_hydrography" / "drainage_basins_3006.parquet")


# ── Trophic State Index (Carlson 1977) ────────────────────────────────────────

def compute_tsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Carlson (1977) Trophic State Index — composite of available components.

    TSI(Chl-a) = 9.81 × ln(Chl-a) + 30.6
    TSI(TP)    = 14.42 × ln(TP) + 4.15
    TSI(Secchi)= 60 - 14.41 × ln(Secchi)
    Composite  = mean of available components

    TSI < 30: oligotrophic | 30–50: mesotrophic | 50–70: eutrophic | >70: hypereutrophic
    """
    components = []
    if "mean_chla" in df.columns:
        chla = df["mean_chla"].clip(lower=0.01)
        components.append(("tsi_chla",   9.81  * np.log(chla)               + 30.6))
    if "mean_tp" in df.columns:
        tp   = df["mean_tp"].clip(lower=0.1)
        components.append(("tsi_tp",     14.42 * np.log(tp)                  + 4.15))
    if "mean_secchi_m" in df.columns:
        sec  = df["mean_secchi_m"].clip(lower=0.01)
        components.append(("tsi_secchi", 60    - 14.41 * np.log(sec)))

    for name, vals in components:
        df[name] = vals

    if components:
        df["tsi"] = pd.concat([v for _, v in components], axis=1).mean(axis=1)
    else:
        df["tsi"] = np.nan

    return df


# ── coordinate projection ──────────────────────────────────────────────────────

def to_sweref99(df: pd.DataFrame) -> gpd.GeoDataFrame:
    t = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
    x, y = t.transform(df["longitude"].values, df["latitude"].values)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs="EPSG:3006")


# ── spatial interpolation for river station background ─────────────────────────

def interpolate_to_grid(gdf: gpd.GeoDataFrame, col: str,
                         res: float = 20_000,
                         clip_geom=None) -> tuple:
    """Bilinear interpolation of point values onto a regular grid.

    clip_geom: shapely geometry — grid cells outside it are set to NaN.
    """
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    v = gdf[col].values

    mask = np.isfinite(v)
    if mask.sum() < 10:
        return None, None, None

    xi = np.arange(x[mask].min(), x[mask].max(), res)
    yi = np.arange(y[mask].min(), y[mask].max(), res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x[mask], y[mask]), v[mask], (Xi, Yi), method="linear")

    if clip_geom is not None and Zi is not None:
        from shapely.geometry import Point
        from shapely.prepared import prep
        prepared = prep(clip_geom)
        flat_x, flat_y = Xi.ravel(), Yi.ravel()
        inside = np.array([prepared.contains(Point(px, py))
                           for px, py in zip(flat_x, flat_y)])
        Zi = Zi.copy()
        Zi[~inside.reshape(Xi.shape)] = np.nan

    return Xi, Yi, Zi


# ── figure ─────────────────────────────────────────────────────────────────────

def make_figure(lake_gdf: gpd.GeoDataFrame, all_gdf: gpd.GeoDataFrame,
                basins: gpd.GeoDataFrame) -> plt.Figure:
    # Extra vertical space so the bottom colorbar of Panel B has room
    fig, axes = plt.subplots(1, 2, figsize=(20, 15), facecolor="white")
    ax1, ax2 = axes
    fig.subplots_adjust(right=0.82)

    cmap_wci = plt.cm.RdYlGn
    cmap_tp  = plt.cm.Blues

    sweden_geom = (basins.geometry.union_all()
                   if hasattr(basins.geometry, "union_all")
                   else basins.geometry.unary_union)
    xmin, ymin, xmax, ymax = basins.total_bounds
    _pad = 30_000

    # ════════════════════════════════════════════════════════════════
    # Panel A — Water Clarity Index
    # ════════════════════════════════════════════════════════════════
    basins.plot(ax=ax1, color="#eeeeee", edgecolor="#bbbbbb", linewidth=0.5, zorder=1)

    # River stations background — interpolated TSI
    rivers = all_gdf[all_gdf["hydro_source"] == "river"].dropna(subset=["tsi"])
    Xi, Yi, Zi = interpolate_to_grid(rivers, "tsi", res=15_000, clip_geom=sweden_geom)
    if Zi is not None:
        ax1.pcolormesh(Xi, Yi, Zi, cmap=cmap_wci, alpha=0.20,
                       vmin=20, vmax=80, shading="auto", zorder=2)

    lakes = lake_gdf.dropna(subset=["tsi", "geometry"])
    sizes = np.clip(np.log1p(lakes["Lake_area"].fillna(1)) * 8, 15, 200)
    sc1 = ax1.scatter(
        lakes.geometry.x, lakes.geometry.y,
        c=lakes["tsi"], cmap=cmap_wci,
        vmin=20, vmax=80, s=sizes, alpha=0.85,
        linewidths=0.3, edgecolors="white", zorder=3,
    )

    # Panel A colorbar
    cb1 = fig.colorbar(sc1, ax=ax1, orientation="vertical",
                       fraction=0.030, pad=0.025, aspect=32, shrink=0.75)
    cb1.set_label("Trophic State Index (Carlson 1977)\n"
                  "< 30 oligotrophic · 30–50 mesotrophic · 50–70 eutrophic · > 70 hypereutrophic",
                  fontsize=9, labelpad=8)
    cb1.ax.tick_params(labelsize=9)
    # Trophic class tick marks
    for tsi_val, lbl in [(30, "Oligo"), (50, "Meso"), (70, "Eu")]:
        cb1.ax.axhline(tsi_val, color="white", lw=1.0, linestyle="--", alpha=0.8)

    ax1.set_title("(A) Trophic State Index", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Easting (SWEREF99 TM, m)", fontsize=10)
    ax1.set_ylabel("Northing (SWEREF99 TM, m)", fontsize=10)

    for label, color in [("Oligotrophic (TSI < 30)", "#1a9850"),
                          ("Mesotrophic (30–50)",     "#fee08b"),
                          ("Eutrophic (TSI > 50)",    "#d73027")]:
        ax1.scatter([], [], c=color, s=60, label=label)
    ax1.legend(fontsize=8, loc="lower right",
               framealpha=0.88, edgecolor="#aaaaaa")

    # ════════════════════════════════════════════════════════════════
    # Panel B — ERA5 precipitation  +  water colour
    # ════════════════════════════════════════════════════════════════
    basins.plot(ax=ax2, color="#eeeeee", edgecolor="#bbbbbb", linewidth=0.5, zorder=1)

    all_with_tp = (all_gdf.dropna(subset=["mean_tp_mm", "geometry"])
                   if "mean_tp_mm" in all_gdf.columns else pd.DataFrame())
    im2 = None
    if not all_with_tp.empty:
        Xi2, Yi2, Zi2 = interpolate_to_grid(all_with_tp, "mean_tp_mm",
                                             res=20_000, clip_geom=sweden_geom)
        if Zi2 is not None:
            Zi2_clip = np.clip(Zi2, 0, np.nanpercentile(Zi2, 98))
            im2 = ax2.pcolormesh(Xi2, Yi2, Zi2_clip, cmap=cmap_tp,
                                  alpha=0.65, shading="auto", zorder=2)

    lakes_colour = lake_gdf.dropna(subset=["mean_colour"])
    log_colour   = np.log1p(lakes_colour["mean_colour"].values)
    sc2 = ax2.scatter(
        lakes_colour.geometry.x, lakes_colour.geometry.y,
        c=log_colour, cmap="plasma",
        vmin=np.nanpercentile(log_colour, 2),
        vmax=np.nanpercentile(log_colour, 98),
        s=35, alpha=0.85,
        linewidths=0.35, edgecolors="#222222", zorder=3,
    )

    # Panel B — two separated colorbars with fixed figure-space geometry
    # The explicit axes keep a clear gap between the two legends.
    pos2 = ax2.get_position()
    bar_height = pos2.height * 0.86
    bar_bottom = pos2.y0 + pos2.height * 0.07
    bar_width = 0.012
    colour_x = pos2.x1 + 0.012
    precip_x = colour_x + bar_width + 0.028

    cax_colour = fig.add_axes([colour_x, bar_bottom, bar_width, bar_height])
    cax_precip = fig.add_axes([precip_x, bar_bottom, bar_width, bar_height])

    cax_colour.set_facecolor("white")
    cax_precip.set_facecolor("white")

    cb3 = fig.colorbar(sc2, cax=cax_colour, orientation="vertical")
    cb3.ax.yaxis.set_label_position("left")
    cb3.set_label("Water colour (mg Pt L⁻¹)", fontsize=9, labelpad=8)
    cb3.ax.tick_params(labelsize=9)
    tick_vals_real = [5, 20, 50, 100, 200, 400]
    tick_locs   = [np.log1p(v) for v in tick_vals_real
                   if cb3.vmin <= np.log1p(v) <= cb3.vmax]
    tick_labels = [str(v) for v in tick_vals_real
                   if cb3.vmin <= np.log1p(v) <= cb3.vmax]
    cb3.set_ticks(tick_locs)
    cb3.set_ticklabels(tick_labels)

    if im2 is not None:
        cb2 = fig.colorbar(im2, cax=cax_precip, orientation="vertical")
        cb2.set_label("ERA5 mean daily precipitation (mm day⁻¹)",
                      fontsize=9, labelpad=8)
        cb2.ax.tick_params(labelsize=9)
    else:
        cax_precip.set_visible(False)

    ax2.set_title("(B) ERA5 precipitation and water colour",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Easting (SWEREF99 TM, m)", fontsize=10)
    ax2.set_ylabel("")
    ax2.tick_params(labelleft=False)

    # Lock both panels to Sweden's bounding box
    for ax in (ax1, ax2):
        ax.set_xlim(xmin - _pad, xmax + _pad)
        ax.set_ylim(ymin - _pad, ymax + _pad)

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading WQ data per station")
    wq = load_wq_per_station()
    wq = compute_tsi(wq)

    logger.info("Loading HydroATLAS")
    atlas = load_hydroatlas()

    logger.info("Loading ERA5 precipitation")
    era5_tp = load_era5_precip()

    logger.info("Loading drainage basins")
    basins = load_drainage_basins()

    # Merge
    df = wq.merge(atlas, on="station_id", how="left")
    if not era5_tp.empty:
        era5_df = era5_tp.reset_index()
        era5_df["station_id"] = era5_df["station_id"].astype(str)
        df = df.merge(era5_df, on="station_id", how="left")
    df["hydro_source"] = df["hydro_source"].fillna("none")

    all_gdf  = to_sweref99(df.dropna(subset=["latitude", "longitude"]))
    lake_gdf = all_gdf[all_gdf["hydro_source"] == "lake"].copy()

    logger.info(
        f"Total stations: {len(all_gdf):,}  |  "
        f"Lake: {len(lake_gdf):,}  |  "
        f"With TSI: {df['tsi'].notna().sum():,}  |  "
        f"With colour: {df['mean_colour'].notna().sum():,}"
    )

    logger.info("Generating WQI map")
    fig = make_figure(lake_gdf, all_gdf, basins)

    out_path = OUT / "fig3_wqi_map.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
