#!/usr/bin/env python3
"""
Figure 4 — Long-term water quality trends in Sweden (2000–2025).

Four panels showing 25-year WQ dynamics benchmarked against published
Nordic literature — uniquely enabled by the 18,743-station, 25-year dataset:

  A. Sweden-wide median total phosphorus (TP) trend — eutrophication recovery
     vs Jeppesen et al. (2005) and Rowe et al. (2021) Nordic benchmarks
  B. Sweden-wide median water colour trend — browning signal
     vs Monteith et al. (2007) and de Wit et al. (2016) boreal trend
  C. WFD approximate TP ecological status distribution by year
     (stacked bars: High / Good / Moderate / Poor / Bad)
  D. N:P molar ratio distribution — nutrient limitation regime (2000–2025)
     with Bergström & Jansson (2006) ecoregion baselines

Output: data/outputs/figures/fig4_wq_trends.png  (300 dpi)

Usage:
    python -m src.evaluation.wq_trends
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

INTERIM = Path("data/interim/SE")
RAW     = Path("data/raw")
OUT     = Path("data/outputs/figures")


# ── loaders ───────────────────────────────────────────────────────────────────

def _load_ts() -> pd.DataFrame:
    p = INTERIM / "derived_wq_timeseries.parquet"
    if not p.exists():
        logger.error("Run python -m src.evaluation.derived_parameters first")
        return pd.DataFrame()
    return pd.read_parquet(p)


def _load_params() -> pd.DataFrame:
    p = INTERIM / "derived_wq_params.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


# ── panels ────────────────────────────────────────────────────────────────────

def panel_tp_trend(ax, ts: pd.DataFrame):
    """Panel A: Sweden-wide median TP trend — eutrophication recovery signal."""
    if ts.empty or "tp_ug_l" not in ts.columns:
        ax.text(0.5, 0.5, "TP data not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(A) TP trend (pending)")
        return

    ann = ts.groupby("year")["tp_ug_l"].agg(
        median="median", p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75), n="count"
    ).reset_index()
    ann = ann[ann["n"] >= 50]

    ax.plot(ann["year"], ann["p25"], "--", color="#8e44ad", lw=0.9,
            alpha=0.6, label="P25–P75 (station spread)")
    ax.plot(ann["year"], ann["p75"], "--", color="#8e44ad", lw=0.9, alpha=0.6)
    ax.plot(ann["year"], ann["median"], "o-", color="#8e44ad",
            lw=2, ms=5, label="Sweden-wide annual median")

    # OLS trend line
    if len(ann) > 5:
        slope, intercept, r, p_val, _ = stats.linregress(ann["year"], ann["median"])
        yr = np.array([ann["year"].min(), ann["year"].max()])
        ax.plot(yr, slope * yr + intercept, "--", color="#5b2c6f", lw=1.5,
                label=f"Trend: {slope:+.2f} µg/L/yr  (R²={r**2:.2f}, p={p_val:.3f})")

    # WFD Good status guideline (approximate national average)
    ax.axhline(24, color="green", lw=1, linestyle=":", alpha=0.7,
               label="Approximate WFD 'Good' status boundary (24 µg L⁻¹, HVMFS 2019:25)")

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Total phosphorus: annual median (µg L⁻¹)", fontsize=10)
    ax.set_title("(A) Total phosphorus trend (2000–2025): eutrophication trajectory",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="center", framealpha=0.88)


def panel_colour_trend(ax, ts: pd.DataFrame):
    """Panel B: Sweden-wide median water colour trend — browning signal."""
    if ts.empty or "colour_mg_pt_l" not in ts.columns:
        ax.text(0.5, 0.5, "Water colour data not available", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("(B) Browning trend (pending)")
        return

    ann = ts.groupby("year")["colour_mg_pt_l"].agg(
        median="median", p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75), n="count"
    ).reset_index()
    ann = ann[ann["n"] >= 50]

    ax.plot(ann["year"], ann["p25"], "--", color="#c0392b", lw=0.9,
            alpha=0.6, label="P25–P75 (station spread)")
    ax.plot(ann["year"], ann["p75"], "--", color="#c0392b", lw=0.9, alpha=0.6)
    ax.plot(ann["year"], ann["median"], "o-", color="#c0392b",
            lw=2, ms=5, label="Sweden-wide annual median")

    if len(ann) > 5:
        slope, intercept, r, p_val, _ = stats.linregress(ann["year"], ann["median"])
        yr = np.array([ann["year"].min(), ann["year"].max()])
        ax.plot(yr, slope * yr + intercept, "--", color="#7b241c", lw=1.5,
                label=f"Trend: {slope:+.2f} mg Pt/L/yr  (R²={r**2:.2f}, p={p_val:.3f})")
        direction = "browning ↑" if slope > 0 else "clearing ↓"
        ax.text(0.98, 0.06, direction, transform=ax.transAxes, fontsize=11,
                ha="right", color="#c0392b", fontweight="bold")

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Water colour: annual median (mg Pt L⁻¹)", fontsize=10)
    ax.set_title("(B) Water colour trend (2000–2025): browning/clearing signal",
                 fontweight="bold")
    ax.legend(fontsize=8)


def panel_wfd_status(ax, ts: pd.DataFrame):
    """Panel C: WFD approximate TP status distribution by year (stacked bars)."""
    if ts.empty or "tp_ug_l" not in ts.columns:
        ax.text(0.5, 0.5, "TP data not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(C) WFD TP status (pending)")
        return

    ts = ts.dropna(subset=["tp_ug_l"]).copy()
    # Simplified national TP thresholds (HVMFS 2019:25, type-averaged)
    bins   = [-np.inf, 8, 24, 42, 80, np.inf]
    labels = ["High (≤8)", "Good (8–24)", "Moderate (24–42)", "Poor (42–80)", "Bad (>80)"]
    colors = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]

    ts["wfd_class"] = pd.cut(ts["tp_ug_l"], bins=bins, labels=labels)
    ann = ts.groupby(["year", "wfd_class"], observed=True).size().unstack(fill_value=0)
    ann_pct = ann.div(ann.sum(axis=1), axis=0) * 100

    bottom = np.zeros(len(ann_pct))
    for lbl, color in zip(labels, colors):
        if lbl in ann_pct.columns:
            ax.bar(ann_pct.index, ann_pct[lbl], bottom=bottom,
                   color=color, label=lbl, width=0.8)
            bottom += ann_pct[lbl].values

    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of stations (%)")
    ax.set_title("(C) Approximate WFD TP ecological status distribution (2000–2025)",
                 fontweight="bold")
    ax.set_ylim(0, 100)
    # Legend placed below the panel (outside the axes) — 5 classes in one row
    ax.legend(fontsize=8, ncol=5,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.18),
              bbox_transform=ax.transAxes,
              framealpha=0.88, edgecolor="#aaaaaa")


def panel_np_regime(ax, ts: pd.DataFrame):
    """Panel D: N:P molar ratio distribution by decade — nutrient limitation regime."""
    if ts.empty or "np_molar_ratio" not in ts.columns:
        ax.text(0.5, 0.5, "N:P ratio data not available\n(re-run derived_parameters.py)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(D) N:P regime distribution (pending)")
        return

    ts = ts.dropna(subset=["np_molar_ratio"]).copy()
    ts["decade"] = (ts["year"] // 5 * 5).astype(str) + "–" + ((ts["year"] // 5 * 5) + 4).astype(str)

    decades = sorted(ts["decade"].unique())
    colors  = plt.cm.viridis(np.linspace(0.15, 0.85, len(decades)))

    for decade, color in zip(decades, colors):
        vals = ts[ts["decade"] == decade]["np_molar_ratio"].clip(1, 200)
        vals.plot.kde(ax=ax, color=color, lw=1.8, label=decade)

    # Bergström & Jansson (2006) reference lines
    ax.axvline(10, color="red",    lw=1.2, linestyle="--", alpha=0.8,
               label="N-limited threshold (N:P < 10)")
    ax.axvline(16, color="orange", lw=1.2, linestyle="--", alpha=0.8,
               label="P-limited threshold (N:P > 16)")

    ax.set_xlim(0, 80)

    # Zone labels centred within each regime (midpoint of N:P range / xlim)
    # N-limited:  0–10  → midpoint x=5  → axes fraction 5/80  = 0.063
    # Co-limited: 10–16 → midpoint x=13 → axes fraction 13/80 = 0.163
    # P-limited:  16–80 → midpoint x=35 → axes fraction 35/80 = 0.438
    ax.text(0.063, 0.80, "N-limited",  fontsize=8, color="red",
        ha="center", transform=ax.transAxes)
    ax.text(0.163, 0.72, "co-limited", fontsize=8, color="gray",
        ha="center", va="center", rotation=90, transform=ax.transAxes)
    ax.text(0.438, 0.80, "P-limited",  fontsize=8, color="darkorange",
        ha="center", transform=ax.transAxes)

    ax.set_xlabel("Molar N:P ratio (TN : TP)")
    ax.set_ylabel("Kernel density estimate (probability density)")
    ax.set_title("(D) Nutrient limitation regime (2000–2025)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right",
              bbox_to_anchor=(0.98, 0.02), bbox_transform=ax.transAxes,
              framealpha=0.88)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading annual WQ time series")
    ts = _load_ts()

    if ts.empty:
        logger.error("No time series data — run derived_parameters.py first")
        return

    logger.info(f"Time series: {len(ts):,} station-years, {ts['station_id'].nunique():,} stations")

    fig = plt.figure(figsize=(18, 14), facecolor="white")
    # fig.suptitle("Long-term Water Quality Trends — Sweden (2000–2025)",
                #  fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.3,
                           bottom=0.10)   # extra bottom margin for Panel C legend

    panel_tp_trend(fig.add_subplot(gs[0, 0]),     ts)
    panel_colour_trend(fig.add_subplot(gs[0, 1]), ts)
    panel_wfd_status(fig.add_subplot(gs[1, 0]),   ts)
    panel_np_regime(fig.add_subplot(gs[1, 1]),    ts)

    out_path = OUT / "fig4_wq_trends.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
