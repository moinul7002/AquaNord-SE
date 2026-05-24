#!/usr/bin/env python3
"""
Tier 1 — Data fidelity against independent references.

Panel A: ERA5 lake_mix_layer_temperature vs SLU MVM water_temp_c
         stratified by lake depth class and season.
Panel B: ERA5 precipitation lag-correlogram vs water colour (CDOM) —
         peak lag identifies terrestrial DOM flushing response time.
Panel C: ERA5 2 m air temperature bias vs SMHI MetObs (2000–2025) —
         quantifies HIRLAM → AROME reanalysis transition effect.
Panel D: ERA5 lake mixing-layer temperature vs dissolved oxygen % saturation —
         seasonal hypoxia signal and thermal forcing of DO dynamics.

Output: data/outputs/figures/fig_tier1_fidelity.png  (300 dpi)

Usage:
    python -m src.evaluation.tier1_fidelity
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

def _load_era5(col: str) -> pd.DataFrame:
    frames = []
    for p in sorted((RAW / "era5").glob("era5_sweden_*.parquet")):
        df = pd.read_parquet(p)
        if col in df.columns:
            frames.append(df[["station_id", "date", col]])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["station_id"] = df["station_id"].astype(str)
    return df


def _load_slu(cols: list) -> pd.DataFrame:
    frames = []
    for p in sorted((RAW / "slu_mvm").glob("slu_mvm_chemistry_*.parquet")):
        df = pd.read_parquet(p)
        available = [c for c in cols if c in df.columns]
        if available:
            frames.append(df[["station_id", "sampling_date"] + available])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["sampling_date"])
    df["station_id"] = df["station_id"].astype(str)
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    return df


def _load_metobs_temp() -> pd.DataFrame:
    frames = []
    for p in sorted((RAW / "smhi_metobs").glob("smhi_metobs_obs_*.parquet")):
        df = pd.read_parquet(p)
        temp_col = next((c for c in df.columns
                         if c.startswith("p2_") and "quality" not in c), None)
        if temp_col is None:
            temp_col = next((c for c in df.columns
                             if "lufttemperatur" in c.lower()
                             and "dagg" not in c and "quality" not in c), None)
        if temp_col:
            keep = ["station_id", "date", temp_col]
            if "latitude"  in df.columns: keep.append("latitude")
            if "longitude" in df.columns: keep.append("longitude")
            frames.append(df[keep].rename(columns={temp_col: "air_temp_c"}))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["station_id"] = df["station_id"].astype(str)
    df["year"] = df["date"].dt.year
    return df


def _load_atlas() -> pd.DataFrame:
    p = RAW / "hydroatlas" / "hydroatlas_station_covariates_SE.parquet"
    df = pd.read_parquet(p, columns=["station_id", "Depth_avg", "hydro_source"])
    return df.assign(station_id=lambda d: d["station_id"].astype(str))


def _add_regression_line(ax, x, y, color="steelblue"):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xi = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
    ax.plot(xi, slope * xi + intercept, color=color, lw=1.5)
    ax.text(0.05, 0.93, f"R={r:.3f}  RMSE={np.sqrt(np.mean((y[mask]-x[mask])**2)):.2f}°C"
            f"\nbias={np.mean(y[mask]-x[mask]):.2f}°C  n={mask.sum():,}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))


# ── panels ─────────────────────────────────────────────────────────────────────

def panel_lake_temp(ax, era5_lmlt: pd.DataFrame, slu: pd.DataFrame, atlas: pd.DataFrame):
    """Panel A: ERA5 lmlt_c vs SLU water_temp_c by lake depth class."""
    if era5_lmlt.empty or "water_temp_c" not in slu.columns:
        ax.text(0.5, 0.5, "ERA5 lake mixing-layer temperature or\nin-situ water temperature not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(A) ERA5 lake temp vs in-situ (pending)")
        return

    # Match: merge ERA5 lmlt on (station_id, date within ±1 day)
    slu_temp = slu.dropna(subset=["water_temp_c"]).copy()
    merged = slu_temp.merge(
        era5_lmlt.rename(columns={"lmlt_c": "era5_lmlt"}),
        on=["station_id", "date"], how="inner"
    )
    merged = merged.merge(atlas, on="station_id", how="left")
    merged = merged[(merged["month"] >= 5) & (merged["month"] <= 10)]  # ice-free

    # Depth classes
    merged["depth_class"] = pd.cut(
        merged["Depth_avg"].fillna(0),
        bins=[0, 3, 10, 50, 9999],
        labels=["<3 m", "3–10 m", "10–50 m", ">50 m"],
    )

    colors = {"<3 m": "#e74c3c", "3–10 m": "#e67e22",
              "10–50 m": "#27ae60", ">50 m": "#2980b9"}

    for dc, grp in merged.groupby("depth_class", observed=True):
        ax.scatter(grp["water_temp_c"], grp["era5_lmlt"],
                   s=4, alpha=0.3, color=colors.get(str(dc), "gray"),
                   label=f"{dc} (n={len(grp):,})", rasterized=True)

    lim = (0, 30)
    ax.plot(lim, lim, "k--", lw=0.8, label="1:1")
    ax.set_xlim(lim); ax.set_ylim(lim)
    _add_regression_line(ax, merged["water_temp_c"].values,
                         merged["era5_lmlt"].values)
    ax.set_xlabel("In-situ water temperature (°C)")
    ax.set_ylabel("ERA5 lake mixing-layer temperature (°C)")
    ax.set_title("(A) ERA5 FLake lake mixing-layer temperature vs. in-situ water temperature",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=2)


def panel_precip_colour(ax, era5_tp: pd.DataFrame, slu: pd.DataFrame):
    """Panel B: lag-correlogram of ERA5 tp vs water colour (CDOM flushing signal)."""
    if era5_tp.empty or "colour_mg_pt_l" not in slu.columns:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(B) ERA5 precipitation → CDOM response (pending)")
        return

    # Monthly means per station, then cross-correlate at lags 0–8 weeks
    era5_m = era5_tp.groupby(["station_id", era5_tp["date"].dt.to_period("M")])["tp_mm"].sum().reset_index()
    slu_m  = slu.dropna(subset=["colour_mg_pt_l"]).copy()
    slu_m["period"] = pd.to_datetime(slu_m["date"]).dt.to_period("M")
    slu_m  = slu_m.groupby(["station_id", "period"])["colour_mg_pt_l"].mean().reset_index()

    era5_m["period"] = era5_m["date"]
    merged = era5_m.merge(slu_m, on=["station_id", "period"])

    if len(merged) < 50:
        ax.text(0.5, 0.5, "Insufficient coincident data", ha="center",
                va="center", transform=ax.transAxes)
        return

    # Cross-correlation at 0–6 month lags
    lags = range(0, 7)
    corrs = []
    for lag in lags:
        shifted_tp = era5_m.copy()
        shifted_tp["period"] = shifted_tp["period"] + lag
        m2 = shifted_tp.merge(slu_m, on=["station_id", "period"])
        if len(m2) > 20:
            r, _ = stats.pearsonr(m2["tp_mm"].values, m2["colour_mg_pt_l"].values)
            corrs.append((lag, r, len(m2)))
        else:
            corrs.append((lag, np.nan, 0))

    lag_vals, r_vals, n_vals = zip(*corrs)
    ax.bar(lag_vals, r_vals, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Pearson r (ERA5 precipitation vs. water colour)")
    ax.set_title("(B) ERA5 precipitation–CDOM colour lag correlogram",
                 fontweight="bold")
    ax.set_xticks(list(lag_vals))
    best_lag = lag_vals[np.nanargmax(r_vals)]
    ax.annotate(f"Peak lag: {best_lag} month(s)",
                xy=(best_lag, r_vals[best_lag]),
                xytext=(best_lag + 0.5, r_vals[best_lag] + 0.02),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="red"), color="red")


def panel_hirlam_arome(ax, era5_t2m: pd.DataFrame, metobs: pd.DataFrame):
    """Panel C: HIRLAM vs AROME discontinuity — ERA5 t2m bias before/after 2014."""
    if era5_t2m.empty or metobs.empty:
        ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(C) HIRLAM–AROME discontinuity (pending)")
        return

    from scipy.spatial import cKDTree

    # ERA5 and MetObs use different station networks — spatial match required.
    # Map each MetObs station to the nearest ERA5 extraction point (≤ 25 km).
    slu = (pd.read_csv(RAW / "slu_mvm" / "slu_mvm_measured_station_ids.csv")
             .dropna(subset=["latitude", "longitude"]))
    slu["station_id"] = slu["station_id"].astype(str)
    era5_ids      = set(era5_t2m["station_id"].unique())
    era5_stations = slu[slu["station_id"].isin(era5_ids)].drop_duplicates("station_id")

    if "latitude" not in metobs.columns:
        ax.text(0.5, 0.5, "MetObs lat/lon not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(C) HIRLAM–AROME discontinuity (pending)")
        return

    met_coords = (metobs.dropna(subset=["latitude", "longitude"])
                        .groupby("station_id")[["latitude", "longitude"]]
                        .first().reset_index())
    tree = cKDTree(era5_stations[["latitude", "longitude"]].values)
    dists, idxs = tree.query(met_coords[["latitude", "longitude"]].values, k=1)
    valid = dists < 0.25
    mapping = pd.DataFrame({
        "station_id":      met_coords["station_id"].values[valid],
        "era5_station_id": era5_stations["station_id"].values[idxs[valid]],
    })

    # Monthly means — map MetObs to ERA5 station then aggregate
    metobs_mapped = metobs.merge(mapping, on="station_id")
    era5_m = (era5_t2m.groupby(["station_id", era5_t2m["date"].dt.to_period("M")])
                      ["t2m_mean_c"].mean().reset_index())
    era5_m["period"] = era5_m["date"]
    met_m  = (metobs_mapped.groupby(
                  ["era5_station_id", pd.to_datetime(metobs_mapped["date"]).dt.to_period("M")])
                  ["air_temp_c"].mean().reset_index()
                  .rename(columns={"era5_station_id": "station_id"}))
    met_m["period"] = met_m["date"]

    merged = era5_m.merge(met_m, on=["station_id", "period"])
    merged["year"] = merged["period"].dt.year
    merged["bias"] = merged["t2m_mean_c"] - merged["air_temp_c"]
    merged["period_label"] = merged["year"].apply(
        lambda y: "HIRLAM (2000–2013)" if y < 2014 else "AROME (2014–2025)"
    )

    annual_bias = merged.groupby("year")["bias"].agg(["mean", "std"]).reset_index()

    ax.axvline(2014, color="red", lw=1.5, linestyle="--",
               label="HIRLAM → AROME transition (January 2014)")
    ax.errorbar(annual_bias["year"], annual_bias["mean"],
                yerr=annual_bias["std"],
                fmt="o-", color="steelblue", lw=1.5, ms=4, capsize=3,
                elinewidth=0.8, ecolor="steelblue", alpha=0.6,
                label="Annual mean bias ± 1 SD")
    ax.axhline(0, color="black", lw=0.8, linestyle=":")   # zero line — no legend entry

    # Period mean bias annotations — lower area (legend is at upper left, no conflict)
    for lbl, mask, xpos in [("HIRLAM", annual_bias["year"] < 2014,  0.02),
                              ("AROME",  annual_bias["year"] >= 2014, 0.58)]:
        mb = annual_bias.loc[mask, "mean"].mean()
        ax.text(xpos, 0.05, f"{lbl}: {mb:+.2f}°C",
                transform=ax.transAxes, fontsize=8, color="steelblue",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("ERA5 2 m temperature bias vs. MetObs (°C)", fontsize=10)
    ax.set_title("(C) ERA5 2 m air temperature bias vs. SMHI MetObs (2000–2025)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")


def panel_do_pct_saturation(ax, era5_lmlt: pd.DataFrame, slu: pd.DataFrame):
    """Panel D: DO % saturation vs ERA5 lake mixing-layer temperature by season.

    DO % saturation is the direct hypoxia indicator (< 50% = hypoxic conditions).
    Shows how ERA5 lake temperature drives oxygen dynamics across seasons.
    Reference hypoxia threshold: 50% saturation (Vaquer-Sunyer & Duarte 2008).
    """
    if era5_lmlt.empty or "do_mg_l" not in slu.columns or "water_temp_c" not in slu.columns:
        ax.text(0.5, 0.5, "Dissolved oxygen or ERA5 lake temperature data not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(D) DO% saturation vs ERA5 lake temperature (pending)")
        return

    slu_do = slu.dropna(subset=["do_mg_l", "water_temp_c"]).copy()
    T = slu_do["water_temp_c"].values
    do_sat = 14.62 - 0.3898*T + 0.006969*T**2 - 5.596e-5*T**3
    slu_do["do_pct"] = (slu_do["do_mg_l"].values / do_sat * 100).clip(0, 150)

    merged = slu_do.merge(
        era5_lmlt[["station_id", "date", "lmlt_c"]],
        on=["station_id", "date"], how="inner"
    ).dropna(subset=["lmlt_c", "do_pct"])

    if len(merged) < 50:
        ax.text(0.5, 0.5, "Insufficient matchups", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(D) DO% saturation vs ERA5 lake temperature")
        return

    seasons = {
        "Spring (MAM)": ([3, 4, 5],   "#27ae60"),
        "Summer (JJA)": ([6, 7, 8],   "#e74c3c"),
        "Autumn (SON)": ([9, 10, 11], "#e67e22"),
        "Winter (DJF)": ([12, 1, 2],  "#2980b9"),
    }

    for season, (months, color) in seasons.items():
        grp = merged[merged["month"].isin(months)]
        if len(grp) > 50:
            ax.scatter(grp["lmlt_c"], grp["do_pct"],
                       s=3, alpha=0.20, color=color, label=season, rasterized=True)

    # Hypoxia threshold line
    ax.axhline(50, color="red", lw=1.5, linestyle="--", alpha=0.8,
               label="Hypoxia threshold (50%)")
    ax.axhline(100, color="gray", lw=0.8, linestyle=":", alpha=0.6)

    n_hyp = int((merged["do_pct"] < 50).sum())
    pct_hyp = n_hyp / len(merged) * 100
    ax.text(0.03, 0.06, f"Hypoxic samples: {pct_hyp:.1f}% (n={n_hyp:,})",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel("ERA5 lake mixing-layer temperature (°C)")
    ax.set_ylabel("Dissolved oxygen saturation (%)")
    ax.set_title("(D) DO% saturation vs. ERA5 lake mixing-layer temperature",
                 fontweight="bold")
    ax.legend(fontsize=8, markerscale=3)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ERA5 variables")
    era5_lmlt = _load_era5("lmlt_c")
    era5_t2m  = _load_era5("t2m_mean_c")
    era5_tp   = _load_era5("tp_mm")
    logger.info(f"  lmlt_c rows: {len(era5_lmlt):,}  t2m rows: {len(era5_t2m):,}")

    logger.info("Loading SLU MVM chemistry")
    slu = _load_slu(["water_temp_c", "colour_mg_pt_l", "do_mg_l",
                     "turbidity_fnu", "tp_ug_l"])
    logger.info(f"  SLU rows: {len(slu):,}")

    logger.info("Loading MetObs")
    metobs = _load_metobs_temp()

    logger.info("Loading HydroATLAS")
    atlas = _load_atlas()

    fig = plt.figure(figsize=(18, 14), facecolor="white")
    # fig.suptitle("ERA5 Reanalysis Data Fidelity",
    #              fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    panel_lake_temp(fig.add_subplot(gs[0, 0]),       era5_lmlt, slu, atlas)
    panel_precip_colour(fig.add_subplot(gs[0, 1]),   era5_tp,   slu)
    panel_hirlam_arome(fig.add_subplot(gs[1, 0]),    era5_t2m,  metobs)
    panel_do_pct_saturation(fig.add_subplot(gs[1, 1]), era5_lmlt, slu)

    out_path = OUT / "fig_tier1_fidelity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
