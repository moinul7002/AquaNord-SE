#!/usr/bin/env python3
"""
Figure — Evaluation split design (§3.7).

Panel A: GroupKFold(5) spatial fold assignment — stations coloured by fold (1–5)
         on a Sweden basemap matching fig1_coverage_map Panel A style.

Panel B: Latitudinal 61°N evaluation split — same basemap, stations coloured
         South (#e67e22, n=729,118) vs North (#2980b9, n=154,207), with a
         dashed split line at 61°N and annotated sample counts.

The feature tier ablation table belongs in the paper as a LaTeX table, not here.

Output: Reports/figures/fig_split_design.png  (300 dpi)

Usage:
    python -m src.evaluation.split_design
"""

import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW = Path("data/raw")
OUT = Path("Reports/figures")

FOLD_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]


# ── data loaders ──────────────────────────────────────────────────────────────

def _load_stations() -> pd.DataFrame:
    path = RAW / "slu_mvm" / "slu_mvm_measured_station_ids.csv"
    return pd.read_csv(path, usecols=["station_id", "latitude", "longitude"]).assign(
        station_id=lambda d: d["station_id"].astype(str)
    )


def _load_atlas() -> pd.DataFrame:
    path = RAW / "hydroatlas" / "hydroatlas_station_covariates_SE.parquet"
    return pd.read_parquet(
        path, columns=["station_id", "hydro_source", "Lake_area", "Hylak_id"]
    ).assign(station_id=lambda d: d["station_id"].astype(str))


def _load_basins() -> gpd.GeoDataFrame:
    return gpd.read_parquet(RAW / "smhi_hydrography" / "drainage_basins_3006.parquet")


def _to_sweref99(df: pd.DataFrame) -> gpd.GeoDataFrame:
    t = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
    x, y = t.transform(df["longitude"].values, df["latitude"].values)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs="EPSG:3006")


# ── fold assignment ───────────────────────────────────────────────────────────

def _compute_folds(stations_gdf: gpd.GeoDataFrame,
                   basins: gpd.GeoDataFrame,
                   atlas: pd.DataFrame,
                   n_splits: int = 5) -> pd.Series:
    """Assign GroupKFold fold (1–n_splits) to every station.

    Lakes  → grouped by Hylak_id.
    Rivers → grouped by SMHI drainage basin (spatial join).
    Groups sorted and assigned round-robin — matches training determinism.
    """
    fold = pd.Series(1, index=stations_gdf.index, dtype=int)

    # Lake stations
    lake_mask = stations_gdf["hydro_source"] == "lake"
    lake_ids = (
        atlas.set_index("station_id")["Hylak_id"]
             .reindex(stations_gdf.loc[lake_mask, "station_id"])
             .fillna(-1).astype(int).values
    )
    unique_lake = sorted(set(lake_ids))
    lake_map = {g: (i % n_splits) + 1 for i, g in enumerate(unique_lake)}
    fold.loc[lake_mask] = [lake_map.get(g, 1) for g in lake_ids]

    # River stations — spatial join to drainage basins
    river_mask = stations_gdf["hydro_source"] == "river"
    rivers_sub = stations_gdf.loc[river_mask, ["station_id", "geometry"]].copy()
    basin_id_col = next(c for c in basins.columns if c != "geometry")
    try:
        joined = gpd.sjoin(
            rivers_sub, basins[[basin_id_col, "geometry"]],
            how="left", predicate="within"
        )
        basin_keys = joined[basin_id_col].fillna("__unmatched__").astype(str)
        basin_keys.index = rivers_sub.index
    except Exception:
        basin_keys = pd.cut(
            rivers_sub.geometry.x, bins=n_splits,
            labels=[str(i) for i in range(n_splits)]
        ).astype(str)
        basin_keys.index = rivers_sub.index

    unique_basins = sorted(basin_keys.unique())
    basin_map = {b: (i % n_splits) + 1 for i, b in enumerate(unique_basins)}
    fold.loc[river_mask] = basin_keys.map(basin_map).fillna(1).astype(int).values

    return fold


# ── Panel A: fold map ─────────────────────────────────────────────────────────

def panel_fold_map(ax, stations_gdf: gpd.GeoDataFrame,
                   basins: gpd.GeoDataFrame,
                   fold: pd.Series,
                   n_splits: int = 5):
    """Panel A: 111 SMHI drainage basins coloured by GroupKFold fold assignment.

    Basin fold is the mode fold of river stations within that basin.
    Station points are overlaid on top with the same fold colours.
    """
    stations_gdf = stations_gdf.copy()
    stations_gdf["fold"] = fold.values

    # ── Assign each drainage basin its dominant fold ───────────────────────────
    basin_id_col = "versionID"
    rivers = stations_gdf[stations_gdf["hydro_source"] == "river"].copy()
    try:
        joined = gpd.sjoin(
            rivers[["fold", "geometry"]],
            basins[[basin_id_col, "geometry"]],
            how="left", predicate="within"
        )
        basin_fold = (
            joined.groupby(basin_id_col)["fold"]
                  .agg(lambda x: int(x.mode().iloc[0]) if len(x) else 1)
        )
        basins_colored = basins.copy()
        basins_colored["fold"] = (
            basins_colored[basin_id_col].map(basin_fold).fillna(1).astype(int)
        )
    except Exception:
        basins_colored = basins.copy()
        basins_colored["fold"] = 1

    # ── Basemap — identical to coverage_map Panel A ───────────────────────────
    basins.plot(ax=ax, color="#f0f0f0", edgecolor="#3f3f3f",
                linewidth=1.5, zorder=1)

    # ── River stations coloured by fold ───────────────────────────────────────
    # White edge lifts points off the basin fill background
    for f in range(1, n_splits + 1):
        sub = rivers[rivers["fold"] == f]
        if len(sub):
            ax.scatter(sub.geometry.x, sub.geometry.y,
                       s=8, alpha=0.75, linewidths=0.4,
                       edgecolors="white", zorder=3,
                       color=FOLD_COLORS[f - 1])

    # ── Lake stations — size by Lake_area, coloured by fold ───────────────────
    lakes = stations_gdf[stations_gdf["hydro_source"] == "lake"]
    lake_size = np.clip(np.log1p(lakes["Lake_area"].fillna(1)) * 6, 12, 130)
    for f in range(1, n_splits + 1):
        mask = lakes["fold"] == f
        if mask.any():
            ax.scatter(
                lakes.loc[mask].geometry.x, lakes.loc[mask].geometry.y,
                s=lake_size[mask], alpha=0.9, linewidths=0.5,
                edgecolors="white", zorder=4, color=FOLD_COLORS[f - 1],
            )

    legend_handles = [
        mpatches.Patch(color=FOLD_COLORS[i], label=f"Fold {i + 1}")
        for i in range(n_splits)
    ]
    legend_handles.append(
        mpatches.Patch(facecolor="none", edgecolor="#3f3f3f", linewidth=1.5,
                       label="SMHI drainage basin (n=111)")
    )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.88, edgecolor="#aaaaaa")
    ax.set_title("(A) Spatial GroupKFold with SMHI drainage basins",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Easting (SWEREF99 TM, m)", fontsize=10)
    ax.set_ylabel("Northing (SWEREF99 TM, m)", fontsize=10)
    ax.tick_params(labelsize=8)


# ── Panel B: latitudinal 61°N split ──────────────────────────────────────────

_LAT_SPLIT = 61.0
_N_SOUTH   = 729_118
_N_NORTH   = 154_207
_COL_SOUTH = "#e67e22"
_COL_NORTH = "#2980b9"


def panel_latitudinal_split(ax, stations_gdf: gpd.GeoDataFrame,
                             basins: gpd.GeoDataFrame):
    """Panel B: stations coloured by latitudinal split at 61°N.

    Background identical to Panel A (#f0f0f0 / #aaaaaa).
    """
    # Convert 61°N to SWEREF99 TM northing (central Swedish longitude ~15°E)
    _, y_split = Transformer.from_crs(
        "EPSG:4326", "EPSG:3006", always_xy=True
    ).transform(15.0, _LAT_SPLIT)

    # Basemap — identical to Panel A
    basins.plot(ax=ax, color="#f0f0f0", edgecolor="#aaaaaa",
                linewidth=0.5, zorder=1)

    # Station colours by zone
    south_mask = stations_gdf.geometry.y < y_split
    for mask, color, label in [
        (south_mask,  _COL_SOUTH, f"South (<61°N)  n={_N_SOUTH:,}"),
        (~south_mask, _COL_NORTH, f"North (≥61°N)  n={_N_NORTH:,}"),
    ]:
        # Rivers
        rivers = stations_gdf[mask & (stations_gdf["hydro_source"] == "river")]
        if len(rivers):
            ax.scatter(rivers.geometry.x, rivers.geometry.y,
                       s=4, alpha=0.5, linewidths=0, zorder=2, color=color)
        # Lakes
        lakes = stations_gdf[mask & (stations_gdf["hydro_source"] == "lake")]
        if len(lakes):
            lake_size = np.clip(np.log1p(lakes["Lake_area"].fillna(1)) * 6, 10, 120)
            ax.scatter(lakes.geometry.x, lakes.geometry.y,
                       s=lake_size, alpha=0.8, linewidths=0.3,
                       edgecolors="white", zorder=3, color=color, label=label)

    # Fix axis limits before drawing spans and line
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()

    # Shaded zones (drawn under stations via low zorder)
    ax.axhspan(ymin,    y_split, alpha=0.08, color=_COL_SOUTH, zorder=0)
    ax.axhspan(y_split, ymax,    alpha=0.08, color=_COL_NORTH,  zorder=0)

    # 61°N split line
    ax.axhline(y_split, color="#333333", lw=1.5, linestyle="--", zorder=4)

    # Annotate split line
    xmin, xmax = ax.get_xlim()
    ax.text(xmin + (xmax - xmin) * 0.03, y_split + (ymax - ymin) * 0.01,
            "61°N", fontsize=9, fontweight="bold", color="#333333", zorder=5)

    # n-count annotations inside each zone
    ax.text(xmax - (xmax - xmin) * 0.03,
            ymin + (y_split - ymin) * 0.85,
            f"South (<61°N)\nn = {_N_SOUTH:,}",
            ha="right", va="center", fontsize=9, color=_COL_SOUTH,
            fontweight="bold", zorder=5)
    ax.text(xmax - (xmax - xmin) * 0.03,
            y_split + (ymax - y_split) * 0.35,
            f"North (≥61°N)\nn = {_N_NORTH:,}",
            ha="right", va="center", fontsize=9, color=_COL_NORTH,
            fontweight="bold", zorder=5)

    ax.legend(loc="lower right", fontsize=9, framealpha=0.88, edgecolor="#aaaaaa")
    ax.set_title("(B) Latitudinal evaluation split (61°N)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Easting (SWEREF99 TM, m)", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(labelleft=False, labelsize=8)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading stations and atlas")
    stations = _load_stations()
    atlas    = _load_atlas()
    basins   = _load_basins()

    df = (
        stations
        .merge(atlas[["station_id", "hydro_source", "Lake_area", "Hylak_id"]],
               on="station_id", how="left")
    )
    df["hydro_source"] = df["hydro_source"].fillna("none")
    df["Lake_area"]    = df["Lake_area"].fillna(0)

    stations_gdf = _to_sweref99(df)

    logger.info("Computing fold assignments")
    fold = _compute_folds(stations_gdf, basins, atlas)
    logger.info(f"  {fold.value_counts().sort_index().to_dict()}")

    # Sweden's SWEREF99 TM aspect ≈ 0.43 (670 km wide / 1540 km tall).
    # At 14" figure height with ~12" usable, each map needs ~5.2" width.
    # Two maps + labels: ~12" total. wspace=0 + sharey collapses the middle gap.
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12, 14), sharey=True, facecolor="white",
        gridspec_kw={"wspace": 0.02, "left": 0.07, "right": 0.97,
                     "top": 0.95, "bottom": 0.06},
    )

    panel_fold_map(ax_a, stations_gdf, basins, fold)
    panel_latitudinal_split(ax_b, stations_gdf, basins)

    # sharey links the y-axes; align x-limits to the same extent so both maps
    # fill their cells symmetrically
    xl = ax_a.get_xlim()
    ax_b.set_xlim(xl)

    out_path = OUT / "fig_split_design.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
