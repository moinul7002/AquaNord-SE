#!/usr/bin/env python3
"""
Figure 5 — Gridded WQI spatial analysis (2000–2025).

Three output figures:

  fig5a_wqi_maps.png       — TSI mean map (A) + 25-year TSI change map (B)
  fig5c_wqi_multivar.png   — 3×3 overview maps: 2000–2025 mean for all 9 parameters
  fig5d_wqi_trends.png     — 3×3 annual spatial-mean trends for all 9 parameters

  table7_wqi_grid_summary.csv — year × parameter statistics

Output:
    data/outputs/figures/fig5a_wqi_maps.png
    data/outputs/figures/fig5c_wqi_multivar.png
    data/outputs/figures/fig5d_wqi_trends.png
    data/outputs/tables/table7_wqi_grid_summary.csv

Usage:
    python -m src.evaluation.wqi_grid_analysis
"""

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NC_PATH     = Path("data/outputs/nordic_water_quality_v1.nc")
BASINS_PATH = Path("data/raw/smhi_hydrography/drainage_basins_3006.parquet")
OUT_FIG     = Path("data/outputs/figures")
OUT_TAB     = Path("data/outputs/tables")

TSI_VAR     = "tsi"
TSI_LEVELS  = [0, 40, 50, 60, 70, 100]
TSI_COLOURS = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]

# Sweden extent in SWEREF99 TM — used as axis limits on every map panel
XLIM = (260_000, 920_000)
YLIM = (6_130_000, 7_680_000)

# Display metadata for each parameter
PARAM_META = {
    "tsi":            ("Carlson TSI",              "dimensionless", plt.cm.RdYlGn_r, (20, 80)),
    "tp_ug_l":        ("Total phosphorus (µg L⁻¹)", "µg L⁻¹",       plt.cm.YlOrRd,  (0, 100)),
    "colour_mg_pt_l": ("Water colour (mg Pt L⁻¹)", "mg Pt L⁻¹",    plt.cm.YlOrBr,  (0, 200)),
    "tn_ug_l":        ("Total nitrogen (µg L⁻¹)",  "µg L⁻¹",       plt.cm.YlOrRd,  (0, 2000)),
    "np_molar_ratio": ("N:P molar ratio",           "–",             plt.cm.PuOr,    (0, 50)),
    "secchi_m":       ("Secchi depth (m)",          "m",             plt.cm.Blues,   (0, 8)),
    "chla_ug_l":      ("Chlorophyll-a (µg L⁻¹)",   "µg L⁻¹",       plt.cm.Greens,  (0, 20)),
    "ph":             ("pH",                        "–",             plt.cm.RdBu,    (5, 8)),
    "do_pct_sat":     ("DO saturation (%)",         "%",             plt.cm.Blues_r, (60, 120)),
}

PANEL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _tsi_cmap_norm():
    cmap = mcolors.LinearSegmentedColormap.from_list("tsi", TSI_COLOURS, N=256)
    norm = mcolors.BoundaryNorm(TSI_LEVELS, cmap.N, clip=True)
    return cmap, norm


def _get_tsi_var(ds):
    return "tsi_mean" if "tsi_mean" in ds else TSI_VAR


# ── Fig 5a: TSI mean + change maps ───────────────────────────────────────────

def _map_panel(ax, easting, northing, data, cmap, norm,
               title, cb_label, basins=None, stats_text=None):
    """Shared map panel: basemap → pcolormesh → colorbar → stats box."""
    if basins is not None:
        basins.plot(ax=ax, color="#f0f0f0", edgecolor="#aaaaaa",
                    linewidth=0.4, zorder=1)
    ax.pcolormesh(easting, northing, data,
                  cmap=cmap, norm=norm, shading="auto",
                  rasterized=True, zorder=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.028, pad=0.02, shrink=0.92)
    cb.set_label(cb_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    if stats_text:
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Easting (SWEREF99 TM, m)", fontsize=9)
    ax.set_ylabel("Northing (SWEREF99 TM, m)", fontsize=9)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.tick_params(labelsize=8)


def panel_mean_map(ax, ds: xr.Dataset, basins=None):
    tsi_mean = ds[_get_tsi_var(ds)].mean(dim="year").values
    easting  = ds["easting"].values
    northing = ds["northing"].values
    cmap, norm = _tsi_cmap_norm()
    _map_panel(ax, easting, northing, tsi_mean, cmap, norm,
               title="(A) Mean TSI (2000–2025)",
               cb_label="Carlson TSI", basins=basins,
               stats_text=f"Sweden mean TSI = {float(np.nanmean(tsi_mean)):.1f}")


def panel_change_map(ax, ds: xr.Dataset, basins=None):
    tsi_var  = _get_tsi_var(ds)
    early    = ds[tsi_var].sel(year=slice(2000, 2005)).mean(dim="year").values
    late     = ds[tsi_var].sel(year=slice(2020, 2025)).mean(dim="year").values
    delta    = late - early
    easting  = ds["easting"].values
    northing = ds["northing"].values
    vmax     = np.nanpercentile(np.abs(delta[np.isfinite(delta)]), 95)
    norm     = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    improved = int(np.nansum(delta < -2))
    worsened = int(np.nansum(delta >  2))
    _map_panel(ax, easting, northing, delta, plt.cm.RdBu_r, norm,
               title="(B) 25-year TSI change",
               cb_label="ΔTSI (2020–25 minus 2000–05)", basins=basins,
               stats_text=(f"Improved (ΔTSI < −2): {improved:,} cells\n"
                           f"Worsened (ΔTSI > +2): {worsened:,} cells"))


# ── Fig 5b: standalone TSI trend ─────────────────────────────────────────────

# ── Fig 5c: 3×3 multi-parameter mean maps ────────────────────────────────────

def fig_multivar_maps(ds: xr.Dataset, basins: gpd.GeoDataFrame) -> plt.Figure:
    """3×3 mean maps for all 9 parameters, styled like coverage_map.py."""
    params   = [p for p in PARAM_META if p in ds.data_vars]
    ncols, nrows = 3, int(np.ceil(len(params) / 3))

    easting  = ds["easting"].values
    northing = ds["northing"].values

    # Sweden aspect ratio in SWEREF99 TM
    dx = XLIM[1] - XLIM[0]   # 660,000 m
    dy = YLIM[1] - YLIM[0]   # 1,550,000 m
    panel_w = 5.0
    panel_h = panel_w * dy / dx

    fig = plt.figure(figsize=(ncols * panel_w + 1.5, nrows * panel_h + 0.8),
                     facecolor="white")
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            hspace=0.08, wspace=0.28,
                            left=0.05, right=0.95, top=0.94, bottom=0.03)

    for i, var in enumerate(params):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        label, units, cmap, (vmin, vmax) = PARAM_META[var]

        # ── basemap (drainage basins) ──
        if basins is not None:
            basins.plot(ax=ax, color="#f0f0f0", edgecolor="#aaaaaa",
                        linewidth=0.4, zorder=1)

        # ── gridded parameter ──
        data = ds[var].mean(dim="year").values
        data_disp = np.where(np.isfinite(data), np.clip(data, vmin, vmax), np.nan)

        im = ax.pcolormesh(easting, northing, data_disp,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True, zorder=2)

        cb = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.02, shrink=0.9)
        cb.set_label(units, fontsize=7)
        cb.ax.tick_params(labelsize=6)

        sweden_mean = float(np.nanmean(data))
        coverage    = 100 * np.isfinite(data).sum() / data.size
        ax.text(0.02, 0.02,
                f"Mean={sweden_mean:.2g}  cov.{coverage:.0f}%",
                transform=ax.transAxes, fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

        ax.set_title(f"({PANEL_LABELS[i]}) {label}", fontweight="bold", fontsize=9)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xlabel("Easting (m)", fontsize=7)
        ax.set_ylabel("Northing (m)", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for j in range(len(params), nrows * ncols):
        fig.add_subplot(gs[j // ncols, j % ncols]).set_visible(False)

    fig.suptitle("Gridded Water Quality — 2000–2025 Mean (IDW, 5 km SWEREF99 TM)",
                 fontsize=12, fontweight="bold")
    return fig


# ── Fig 5d: 3×3 multi-parameter annual trends ────────────────────────────────

def fig_multivar_trends(ds: xr.Dataset) -> plt.Figure:
    """3×3 annual Sweden-wide spatial mean trends for all 9 parameters."""
    params = [p for p in PARAM_META if p in ds.data_vars]
    ncols, nrows = 3, int(np.ceil(len(params) / 3))
    years  = ds["year"].values.astype(int)

    fig = plt.figure(figsize=(18, nrows * 3.8), facecolor="white")
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            hspace=0.38, wspace=0.30,
                            left=0.06, right=0.97, top=0.93, bottom=0.06)

    for i, var in enumerate(params):
        ax    = fig.add_subplot(gs[i // ncols, i % ncols])
        label, units, cmap_obj, _ = PARAM_META[var]
        color = cmap_obj(0.65)

        data     = ds[var].values   # (year, y, x)
        ann_mean = np.array([float(np.nanmean(data[j])) for j in range(len(years))])
        ann_p25  = np.array([float(np.nanpercentile(data[j][np.isfinite(data[j])], 25))
                             if np.isfinite(data[j]).any() else np.nan
                             for j in range(len(years))])
        ann_p75  = np.array([float(np.nanpercentile(data[j][np.isfinite(data[j])], 75))
                             if np.isfinite(data[j]).any() else np.nan
                             for j in range(len(years))])

        ax.plot(years, ann_p25, "--", color=color, lw=0.8, alpha=0.6)
        ax.plot(years, ann_p75, "--", color=color, lw=0.8, alpha=0.6,
                label="P25–P75 (spatial spread)")
        ax.plot(years, ann_mean, "o-", color=color, lw=1.8, ms=4,
                label="Annual spatial mean")

        valid = np.isfinite(ann_mean)
        if valid.sum() > 5:
            slope, intercept, r, p_val, _ = stats.linregress(
                years[valid], ann_mean[valid])
            yr_r = np.array([years[valid].min(), years[valid].max()])
            ax.plot(yr_r, slope * yr_r + intercept, "--", color="#333333", lw=1.2,
                    label=f"{slope:+.3g} {units} yr⁻¹  p={p_val:.2f}")

        ax.set_title(f"({PANEL_LABELS[i]}) {label}", fontweight="bold", fontsize=9)
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel(units, fontsize=8)
        ax.legend(fontsize=6.5, loc="best", framealpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_facecolor("white")

    for j in range(len(params), nrows * ncols):
        fig.add_subplot(gs[j // ncols, j % ncols]).set_visible(False)

    fig.suptitle("Annual Sweden-wide Spatial Mean — All 9 Gridded WQ Parameters (2000–2025)",
                 fontsize=12, fontweight="bold")
    return fig


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary_table(ds: xr.Dataset) -> pd.DataFrame:
    years      = ds["year"].values.astype(int)
    param_vars = [v for v in ds.data_vars if not v.startswith("n_stations_")]

    rows = []
    for i, yr in enumerate(years):
        row = {"year": yr}
        for var in param_vars:
            layer = ds[var].values[i]
            valid = layer[np.isfinite(layer)]
            row[f"{var}_mean"]     = round(float(np.nanmean(layer)),             3) if valid.size else np.nan
            row[f"{var}_std"]      = round(float(np.nanstd(layer)),              3) if valid.size else np.nan
            row[f"{var}_p25"]      = round(float(np.nanpercentile(layer, 25)),   3) if valid.size else np.nan
            row[f"{var}_p75"]      = round(float(np.nanpercentile(layer, 75)),   3) if valid.size else np.nan
            row[f"{var}_coverage"] = round(100 * valid.size / layer.size,        1)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    if not NC_PATH.exists():
        logger.error(f"Not found: {NC_PATH} — run build_wqi_grid.py first")
        return

    logger.info(f"Loading {NC_PATH}")
    ds = xr.open_dataset(str(NC_PATH))
    logger.info(f"  dims: {dict(ds.sizes)}  variables: {[v for v in ds.data_vars if not v.startswith('n_stations_')]}")

    # ── Load drainage basins (basemap for fig5c) ──
    basins = None
    if BASINS_PATH.exists():
        logger.info("Loading drainage basins for basemap")
        basins = gpd.read_parquet(BASINS_PATH)
    else:
        logger.warning("Drainage basins not found — fig5c will have no basemap")

    # ── Fig 5a: TSI mean + change (side by side, tight) ──
    dx = XLIM[1] - XLIM[0]
    dy = YLIM[1] - YLIM[0]
    panel_w = 6.5
    panel_h = panel_w * dy / dx
    fig_maps = plt.figure(figsize=(panel_w * 2 + 2.5, panel_h + 0.8), facecolor="white")
    gs_maps  = gridspec.GridSpec(1, 2, figure=fig_maps,
                                 wspace=0.22, left=0.06, right=0.96,
                                 top=0.93, bottom=0.06)
    panel_mean_map(fig_maps.add_subplot(gs_maps[0, 0]),   ds, basins)
    panel_change_map(fig_maps.add_subplot(gs_maps[0, 1]), ds, basins)
    out_maps = OUT_FIG / "fig5a_wqi_maps.png"
    fig_maps.savefig(out_maps, dpi=300, bbox_inches="tight")
    logger.info(f"Saved → {out_maps}")
    plt.close(fig_maps)

    # ── Fig 5c: multi-parameter mean maps ──
    fig_mv = fig_multivar_maps(ds, basins)
    out_mv = OUT_FIG / "fig5c_wqi_multivar.png"
    fig_mv.savefig(out_mv, dpi=300, bbox_inches="tight")
    logger.info(f"Saved → {out_mv}")
    plt.close(fig_mv)

    # ── Fig 5d: multi-parameter trends ──
    fig_mt = fig_multivar_trends(ds)
    out_mt = OUT_FIG / "fig5d_wqi_trends.png"
    fig_mt.savefig(out_mt, dpi=300, bbox_inches="tight")
    logger.info(f"Saved → {out_mt}")
    plt.close(fig_mt)

    # ── Table 7 ──
    tbl = build_summary_table(ds)
    out_tbl = OUT_TAB / "table7_wqi_grid_summary.csv"
    tbl.to_csv(out_tbl, index=False)
    logger.info(f"Saved → {out_tbl}")

    ds.close()
    logger.info("All figures and table complete.")


if __name__ == "__main__":
    main()
