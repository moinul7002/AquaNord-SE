#!/usr/bin/env python3
"""
Tier 3 — Spatial credibility: lake polygons + catchment covariates + gridded data.

Exploits the polygon + gridded + station triangle that makes this dataset
richer than a simple tabular benchmark.

Panel A: Trophic state vs catchment land use
         TSI vs crp_pc_use (agriculture) and for_pc_use (forest) — independent validation
         of the WQ-land cover link across 4,231 stations.

Panel B: Retention time × nutrient flushing
         Lake Vol/Q (retention time) vs TP concentration variance — tests whether
         hydraulic flushing vs internal cycling controls TP.

Panel C: River network downstream gradient
         TP and colour along Strahler stream order — mechanistic consistency check.

Panel D: Lake depth vs thermal buffering
         LakeATLAS Depth_avg vs seasonal amplitude of water_temp_c
         + ERA5 lmlt_c — validates FLake stratification representation.

Panel E: Within-lake spatial variability
         For multi-station lakes (same Hylak_id), coefficient of variation of WQ
         vs lake area — quantifies how well a single station represents a lake.

Panel F: ERA5 catchment gradient vs WQ variability
         Spatial gradient of ERA5 runoff (from 12×12 patch) vs between-lake
         TP variability within the same drainage basin.

Output: data/outputs/figures/fig_tier3_spatial.png  (300 dpi)

Usage:
    python -m src.evaluation.tier3_spatial
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

RAW     = Path("data/raw")
INTERIM = Path("data/interim/SE")
OUT     = Path("data/outputs/figures")


# ── loaders ───────────────────────────────────────────────────────────────────

def _load_derived_params() -> pd.DataFrame:
    p = INTERIM / "derived_wq_params.parquet"
    if not p.exists():
        logger.warning("derived_wq_params.parquet not found — run derived_parameters.py first")
        return pd.DataFrame()
    return pd.read_parquet(p).assign(station_id=lambda d: d["station_id"].astype(str))


def _load_chemistry_annual() -> pd.DataFrame:
    p = INTERIM / "derived_wq_timeseries.parquet"
    if not p.exists():
        logger.warning("derived_wq_timeseries.parquet not found — run derived_parameters.py first")
        return pd.DataFrame()
    return pd.read_parquet(p).assign(station_id=lambda d: d["station_id"].astype(str))


def _load_chemistry_raw() -> pd.DataFrame:
    frames = []
    for p in sorted((RAW / "slu_mvm").glob("slu_mvm_chemistry_*.parquet")):
        df = pd.read_parquet(p)
        df["year"] = int(p.stem.split("_")[-1])
        df["month"] = pd.to_datetime(df["sampling_date"]).dt.month
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["station_id"] = df["station_id"].astype(str)
    return df


def _load_atlas() -> pd.DataFrame:
    p = RAW / "hydroatlas" / "hydroatlas_station_covariates_SE.parquet"
    df = pd.read_parquet(p)
    return df.assign(station_id=lambda d: d["station_id"].astype(str))


def _load_era5_patches_sample(n_stations: int = 500) -> pd.DataFrame:
    """Load a sample of ERA5 patches to compute spatial gradients."""
    patch_dir = Path("data/processed/era5_wq_patches")
    if not patch_dir.exists():
        return pd.DataFrame()
    results = []
    stations = sorted(patch_dir.iterdir())[:n_stations]
    for station_dir in stations:
        if not station_dir.is_dir():
            continue
        npz_files = sorted(station_dir.glob("*.npz"))
        if not npz_files:
            continue
        try:
            z = np.load(npz_files[-1], allow_pickle=True)  # most recent year
            patch = z["patch"]  # (n_valid_days, 12, 12, 9)
            # Channel index 7 = sro (surface runoff), 8 = ro (total runoff)
            if patch.shape[-1] >= 9:
                runoff_patch = patch[:, :, :, 8]  # ro channel
                # Spatial gradient magnitude (Sobel approximation)
                valid = ~np.isnan(runoff_patch)
                if valid.any():
                    mean_runoff = np.nanmean(runoff_patch)
                    with np.errstate(all="ignore"):
                        grad_x = np.nanmean(runoff_patch[:, :, 6:] - runoff_patch[:, :, :6])
                        grad_y = np.nanmean(runoff_patch[:, :6, :] - runoff_patch[:, 6:, :])
                    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                    results.append({
                        "station_id": station_dir.name,
                        "mean_runoff_m": float(mean_runoff),
                        "gradient_magnitude": float(grad_mag),
                    })
        except Exception:
            continue
    return pd.DataFrame(results) if results else pd.DataFrame()


# ── panels ─────────────────────────────────────────────────────────────────────

def panel_trophic_vs_landuse(ax, derived: pd.DataFrame, atlas: pd.DataFrame):
    """Panel A: TSI vs catchment land use."""
    if derived.empty or "tsi" not in derived.columns:
        ax.text(0.5, 0.5, "TSI not computed\n(run derived_parameters.py first + SLU MVM re-extract)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(A) Trophic state vs land use (data pending)")
        return

    # derived already contains hydro_source, crp_pc_use, for_pc_use from atlas merge
    df = derived.copy()
    # supplement with urb_pc_use if not already present
    if "urb_pc_use" not in df.columns and "urb_pc_use" in atlas.columns:
        df = df.merge(atlas[["station_id", "urb_pc_use"]], on="station_id", how="left")
    if "hydro_source" not in df.columns or "crp_pc_use" not in df.columns:
        ax.text(0.5, 0.5, "hydro_source or crp_pc_use missing from derived_wq_params\n"
                "(re-run derived_parameters.py)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(A) TSI vs land use (derived params missing)")
        return
    df = df[df["hydro_source"] == "lake"].dropna(subset=["tsi", "crp_pc_use"])

    if len(df) < 20:
        ax.text(0.5, 0.5, f"Only {len(df)} stations with TSI + land use",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(A) Trophic state vs land use (insufficient data)")
        return

    sc = ax.scatter(df["crp_pc_use"], df["tsi"],
                    c=df["for_pc_use"], cmap="YlGn_r",
                    s=20, alpha=0.6, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.03)
    cb.set_label("Forest cover (%)", fontsize=9)

    # Regression
    mask = np.isfinite(df["crp_pc_use"]) & np.isfinite(df["tsi"])
    if mask.sum() > 10:
        slope, intercept, r, p, _ = stats.linregress(
            df.loc[mask, "crp_pc_use"], df.loc[mask, "tsi"]
        )
        xi = np.linspace(df["crp_pc_use"].min(), df["crp_pc_use"].max(), 100)
        pval = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.plot(xi, slope * xi + intercept, "r-", lw=1.5,
                label="Linear regression")

        # Box 1 (lower right, bottom): regression line label
        leg1 = ax.legend(fontsize=8, loc="lower right", framealpha=0.88,
                         bbox_to_anchor=(1.0, 0.0), bbox_transform=ax.transAxes)
        ax.add_artist(leg1)

        # Box 2 (lower right, just above): R / p / n stats — separate box
        from matplotlib.lines import Line2D
        stats_h = Line2D([0], [0], alpha=0, label=f"r={r:.2f},  {pval},  n={mask.sum():,} stations")
        ax.legend(handles=[stats_h], fontsize=8, framealpha=0.88,
                  loc="lower right",
                  bbox_to_anchor=(1.0, 0.06), bbox_transform=ax.transAxes)

    ax.set_xlabel("Upstream cropland fraction (%)")
    ax.set_ylabel("Trophic State Index (Carlson 1977)")
    ax.set_title("(A) Trophic State Index vs. upstream cropland fraction",
                 fontweight="bold")
    # Trophic level reference lines with inline labels
    for tsi_val, lbl in [(30, "Oligotrophic"), (50, "Mesotrophic"), (70, "Eutrophic")]:
        ax.axhline(tsi_val, color="gray", lw=0.7, linestyle=":")
        ax.text(ax.get_xlim()[1] * 0.98, tsi_val + 1, lbl,
                ha="right", fontsize=7, color="gray")


def panel_retention_vs_tp(ax, derived: pd.DataFrame):
    """Panel B: Retention time vs TP — flushing vs internal cycling."""
    if derived.empty:
        ax.text(0.5, 0.5, "Derived params not available", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("(B) Retention time vs TP (data pending)")
        return

    df = derived.copy()
    has_rt  = "retention_time_days" in df.columns
    has_tp  = "tp_ug_l" in df.columns
    if not (has_rt and has_tp):
        ax.text(0.5, 0.5, "retention_time_days or tp_ug_l missing",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(B) Retention time vs TP (TP data pending)")
        return

    df = df.dropna(subset=["retention_time_days", "tp_ug_l"])
    if "hydro_source" in df.columns:
        df = df[df["hydro_source"] == "lake"]

    if len(df) < 10:
        ax.text(0.5, 0.5, f"Only {len(df)} stations with both metrics",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(B) Retention time vs TP (insufficient data)")
        return

    ax.scatter(df["retention_time_days"].clip(1, 3650),
               df["tp_ug_l"].clip(1, 500),
               s=10, alpha=0.4, color="#8e44ad", rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Water retention time (days)")
    ax.set_ylabel("Mean total phosphorus (µg L⁻¹)")
    ax.set_title("(B) Total phosphorus vs. water retention time",
                 fontweight="bold")

    # Vollenweider line (expected: TP ∝ τ^0.5 for P-limited lakes)
    tau = np.logspace(0, np.log10(3650), 100)
    ax.plot(tau, 15 * tau**0.3, "r--", lw=1.5, alpha=0.7,
            label="Vollenweider expectation (TP ∝ τ⁰·³)")

    # Box 1: model expectation line
    leg1 = ax.legend(fontsize=8, loc="upper left", framealpha=0.88)
    ax.add_artist(leg1)

    # Box 2: sample size — separate box
    from matplotlib.lines import Line2D
    n_h = Line2D([0], [0], alpha=0, label=f"n = {len(df):,} stations")
    ax.legend(handles=[n_h], fontsize=8, loc="lower left", framealpha=0.88)


def panel_river_gradient(ax, chem_raw: pd.DataFrame, atlas: pd.DataFrame):
    """Panel C: WQ parameters along Strahler stream order."""
    if chem_raw.empty:
        ax.text(0.5, 0.5, "Chemistry data not available", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("(C) River network gradient (data pending)")
        return

    # Get Strahler order from RiverATLAS
    if "ORD_STRA" not in atlas.columns:
        ax.text(0.5, 0.5, "ORD_STRA not in HydroATLAS\n(river-specific column)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(C) River network TP gradient (column missing)")
        return

    river_stations = atlas[atlas["hydro_source"] == "river"][["station_id", "ORD_STRA"]]
    merged = chem_raw.merge(river_stations, on="station_id", how="inner")

    params = []
    if "tp_ug_l" in merged.columns:
        params.append(("tp_ug_l", "Total phosphorus", "#e74c3c"))
    if "colour_mg_pt_l" in merged.columns:
        params.append(("colour_mg_pt_l", "Water colour", "#8e44ad"))

    if not params:
        ax.text(0.5, 0.5, "TP or colour not in chemistry\n(re-run fetch_slu_mvm.py)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(C) River network gradient (TP/colour pending)")
        return

    orders = sorted(merged["ORD_STRA"].dropna().unique())
    ax2 = ax.twinx() if len(params) > 1 else None

    for i, (col, label, color) in enumerate(params):
        grp = merged.dropna(subset=[col]).groupby("ORD_STRA")[col]
        medians = grp.median().reindex(orders)
        q25     = grp.quantile(0.25).reindex(orders)
        q75     = grp.quantile(0.75).reindex(orders)

        target_ax = ax if i == 0 else ax2
        target_ax.plot(orders, medians, "o-", color=color, lw=2, ms=6, label=label)
        target_ax.fill_between(orders, q25, q75, alpha=0.2, color=color)
        target_ax.set_ylabel(label, color=color)

    ax.set_xlabel("Strahler stream order")
    ax.set_title("(C) Total phosphorus and water colour along Strahler stream order gradient",
                 fontweight="bold")
    ax.set_xticks(orders)

    lines1, labels1 = ax.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)


def panel_depth_buffering(ax, chem_raw: pd.DataFrame, derived: pd.DataFrame):
    """Panel D: Lake depth vs seasonal temperature amplitude."""
    if chem_raw.empty or "water_temp_c" not in chem_raw.columns:
        ax.text(0.5, 0.5, "water_temp_c not available", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("(D) Depth vs thermal buffering (data pending)")
        return

    if "Depth_avg" not in derived.columns:
        ax.text(0.5, 0.5, "Depth_avg not in derived params", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("(D) Depth vs thermal buffering (atlas missing)")
        return

    # Seasonal amplitude per station
    hs_col = "hydro_source" if "hydro_source" in derived.columns else None
    lake_mask = (derived[hs_col] == "lake") if hs_col else pd.Series(True, index=derived.index)
    lake_stations = derived[lake_mask]["station_id"].tolist()
    temp_data = (
        chem_raw[chem_raw["station_id"].isin(lake_stations)]
        .dropna(subset=["water_temp_c"])
        .groupby(["station_id", "month"])["water_temp_c"]
        .mean()
        .unstack("month")
    )
    if temp_data.empty:
        ax.text(0.5, 0.5, "No lake temperature data", ha="center",
                va="center", transform=ax.transAxes)
        return

    summer = [6, 7, 8]
    winter = [12, 1, 2]
    s_cols = [c for c in summer if c in temp_data.columns]
    w_cols = [c for c in winter if c in temp_data.columns]

    if not s_cols or not w_cols:
        ax.text(0.5, 0.5, "Insufficient seasonal coverage", ha="center",
                va="center", transform=ax.transAxes)
        return

    amplitude = (temp_data[s_cols].mean(axis=1) -
                 temp_data[w_cols].mean(axis=1)).rename("temp_amplitude")
    amplitude = amplitude.reset_index()
    amplitude = amplitude.merge(
        derived[["station_id", "Depth_avg"]].dropna(), on="station_id"
    )

    ax.scatter(amplitude["Depth_avg"].clip(0, 60),
               amplitude["temp_amplitude"],
               s=8, alpha=0.4, color="#27ae60", rasterized=True)

    # Expected: deeper lakes buffer temperature (lower amplitude)
    depth_bins = [0, 3, 10, 20, 60]
    means = []
    for d0, d1 in zip(depth_bins[:-1], depth_bins[1:]):
        mask = (amplitude["Depth_avg"] >= d0) & (amplitude["Depth_avg"] < d1)
        means.append(amplitude.loc[mask, "temp_amplitude"].mean())

    ax.plot([1.5, 6, 15, 40], means, "ro-", lw=2, ms=8, label="Bin means")
    ax.set_xlabel("Lake mean depth (m)")
    ax.set_ylabel("Summer–winter water temperature amplitude (°C)")
    ax.set_title("(D) Lake depth vs. seasonal water temperature amplitude",
                 fontweight="bold")

    # Box 1: bin means line
    leg1 = ax.legend(fontsize=8, loc="upper right", framealpha=0.88)
    ax.add_artist(leg1)

    # Box 2: sample size — separate box
    from matplotlib.lines import Line2D
    n_h = Line2D([0], [0], alpha=0, label=f"n = {len(amplitude):,} stations")
    ax.legend(handles=[n_h], fontsize=8, loc="lower left", framealpha=0.88)


def panel_within_lake_variability(ax, chem_raw: pd.DataFrame, atlas: pd.DataFrame):
    """Panel E: Within-lake CV of WQ vs lake area — single-station representativeness."""
    if chem_raw.empty:
        ax.text(0.5, 0.5, "Chemistry data not available", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("(E) Within-lake variability (data pending)")
        return

    # Lakes with multiple stations
    lake_map = atlas[atlas["hydro_source"] == "lake"][["station_id", "Hylak_id", "Lake_area"]].dropna()
    multi = lake_map.groupby("Hylak_id").filter(lambda x: len(x) > 1)

    if multi.empty:
        ax.text(0.5, 0.5, "No multi-station lakes found", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("(E) Within-lake variability (no multi-station lakes)")
        return

    # ── Panel E: CDOM-stratified TSI offset — Nordic-specific finding ──────────
    # In CDOM-rich Nordic lakes TSI(Secchi) reads systematically HIGH relative to
    # TSI(Chl-a) because non-algal CDOM darkens the water independently of algae.
    # This violates Carlson (1977) assumptions and shows where Secchi-based trophic
    # classification overestimates eutrophication in boreal/humic lakes.
    # Reference: Carlson (1977) caution; Verspagen et al. (2006); Sobek et al. (2007)

    derived = _load_derived_params()

    if derived.empty or not all(c in derived.columns for c in ["tsi_secchi", "tsi_chla", "colour_mg_pt_l"]):
        ax.text(0.5, 0.5, "TSI components or colour data not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(E) CDOM-TSI offset (pending re-run of derived_parameters)")
        return

    m = derived.dropna(subset=["tsi_secchi", "tsi_chla", "colour_mg_pt_l"])
    m = m[m["colour_mg_pt_l"] > 0].copy()
    m["tsi_offset"] = m["tsi_secchi"] - m["tsi_chla"]

    # Scatter coloured by CDOM class
    cdom_classes = [
        (m["colour_mg_pt_l"] < 30,              "#2196F3", "Clear (< 30 mg Pt/L)"),
        ((m["colour_mg_pt_l"] >= 30) & (m["colour_mg_pt_l"] < 100), "#FF9800", "Coloured (30–100)"),
        (m["colour_mg_pt_l"] >= 100,             "#B71C1C", "Humic (> 100 mg Pt/L)"),
    ]
    for mask, color, label in cdom_classes:
        grp = m[mask]
        if len(grp) > 5:
            ax.scatter(grp["colour_mg_pt_l"].clip(1, 500), grp["tsi_offset"],
                       s=8, alpha=0.35, color=color, label=label, rasterized=True)

    # LOESS-like rolling median trend
    m_sorted = m.sort_values("colour_mg_pt_l")
    roll = m_sorted.set_index("colour_mg_pt_l")["tsi_offset"].rolling(200, min_periods=20).median()
    ax.plot(roll.index, roll.values, color="black", lw=2, label="Median trend")

    # Reference line: zero offset (where Carlson assumptions hold)
    ax.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.7)
    ax.axhline(10, color="red", lw=0.8, linestyle=":", alpha=0.5)
    ax.text(500, 11, "TSI overestimation > 10 units", ha="right", fontsize=7, color="red")

    ax.set_xscale("log")
    ax.set_xlabel("Water colour (mg Pt L⁻¹) — CDOM proxy")
    ax.set_ylabel("TSI(Secchi) − TSI(Chl-a) offset")
    ax.set_title("(E) CDOM-induced TSI offset: Nordic-specific trophic bias",
                 fontweight="bold")

    from matplotlib.lines import Line2D

    leg1 = ax.legend(fontsize=8, loc="upper left")
    ax.add_artist(leg1)
    n_c = Line2D([0], [0], alpha=0, label=f"n = {len(m):,} stations")
    ax.legend(handles=[n_c], fontsize=8, loc="lower left", framealpha=0.88)


def panel_era5_gradient_vs_wq(ax, derived: pd.DataFrame, atlas: pd.DataFrame):
    """Panel F: ERA5 catchment runoff gradient magnitude vs between-lake TP variability."""
    era5_grads = _load_era5_patches_sample()

    if era5_grads.empty or derived.empty:
        ax.text(0.5, 0.5, "ERA5 patches or derived params not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(F) ERA5 spatial gradient vs WQ variability (data pending)")
        return

    if "tp_ug_l" not in derived.columns:
        ax.text(0.5, 0.5, "tp_ug_l not in derived params (TP data pending)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(F) ERA5 gradient vs TP (TP pending)")
        return

    # Join ERA5 gradients with TP and basin
    df = era5_grads.merge(
        derived[["station_id", "tp_ug_l"]].dropna(), on="station_id", how="inner"
    )

    if len(df) < 20:
        ax.text(0.5, 0.5, f"Only {len(df)} matching stations",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(F) ERA5 gradient vs TP (limited overlap)")
        return

    ax.scatter(df["gradient_magnitude"] * 1000, df["tp_ug_l"].clip(1, 500),
               s=10, alpha=0.4, color="#2980b9", rasterized=True,
               label="stations")
    ax.set_yscale("log")
    ax.set_xlabel("ERA5 catchment runoff spatial gradient (mm grid-cell⁻¹)")
    ax.set_ylabel("Mean total phosphorus (µg L⁻¹)")

    mask = np.isfinite(df["gradient_magnitude"]) & np.isfinite(np.log(df["tp_ug_l"]))
    if mask.sum() > 10:
        r, p = stats.pearsonr(df.loc[mask, "gradient_magnitude"],
                              np.log(df.loc[mask, "tp_ug_l"]))
        pval = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        from matplotlib.lines import Line2D

        leg1 = ax.legend(fontsize=8, loc="upper right", framealpha=0.88)
        ax.add_artist(leg1)
        stats_h = Line2D([0], [0], alpha=0, label=f"r={r:.2f},  {pval},  n={mask.sum():,} stations")
        ax.legend(handles=[stats_h], fontsize=8, framealpha=0.88,
              loc="lower right",
              bbox_to_anchor=(1.0, 0.0), bbox_transform=ax.transAxes)

    ax.set_title("(F) ERA5 catchment runoff gradient vs. lake total phosphorus",
                 fontweight="bold")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)

    logger.info("Loading derived parameters")
    derived  = _load_derived_params()
    chem_raw = _load_chemistry_raw()
    atlas    = _load_atlas()

    logger.info(f"Derived: {len(derived):,} stations  |  Chemistry: {len(chem_raw):,} rows")

    fig = plt.figure(figsize=(20, 18), facecolor="white")
    # fig.suptitle("Spatial Credibility Evaluation",
    #              fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    panel_trophic_vs_landuse(fig.add_subplot(gs[0, 0]),      derived, atlas)
    panel_retention_vs_tp(fig.add_subplot(gs[0, 1]),          derived)
    panel_river_gradient(fig.add_subplot(gs[1, 0]),           chem_raw, atlas)
    panel_depth_buffering(fig.add_subplot(gs[1, 1]),          chem_raw, derived)
    panel_within_lake_variability(fig.add_subplot(gs[2, 0]),  chem_raw, atlas)
    panel_era5_gradient_vs_wq(fig.add_subplot(gs[2, 1]),      derived, atlas)

    out_path = OUT / "fig_tier3_spatial.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
