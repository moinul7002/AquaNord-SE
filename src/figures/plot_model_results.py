#!/usr/bin/env python3
"""
Generate fig_model_results.png — main model comparison figure for the paper.

Four panels:
  (A) R² by model × target — best feature tier per model (grouped bar)
  (B) Feature-tier ablation — LightGBM R² across 5 tiers × 3 targets (line)
  (C) Patch + multimodal vs tabular — R² comparison (dot plot)
  (D) Conformal calibration — PICP vs nominal coverage for LightGBM × target

Output: data/outputs/figures/fig_model_results.png
"""

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path("data/outputs/figures/fig_model_results.png")

TARGET_LABELS = {
    "chla_ug_l": "Chl-a (µg/L)",
    "tp_ug_l":   "TP (µg/L)",
    "secchi_m":  "Secchi (m)",
}
TARGETS = ["chla_ug_l", "tp_ug_l", "secchi_m"]

MODEL_ORDER = ["StationMean", "Ridge", "RF", "XGBoost", "LightGBM", "MLP", "FT-Transformer"]
TIER_ORDER  = ["static", "met", "met_mesan", "published_alg", "full"]
TIER_LABELS = {
    "static":       "Static\n(HydroATLAS)",
    "met":          "+ ERA5\nmet",
    "met_mesan":    "+ MESAN",
    "published_alg":"Published\nalgorithms",
    "full":         "Full\n(all sources)",
}

TARGET_COLORS = {"chla_ug_l": "#2980b9", "tp_ug_l": "#27ae60", "secchi_m": "#8e44ad"}


def _best_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Return the best R² row per (model, target) across all feature tiers."""
    idx = df.groupby(["model", "target"])["r2"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def panel_a(ax, tab: pd.DataFrame):
    """Grouped bar: R² by model × target (best tier)."""
    best = _best_tier(tab)
    models = [m for m in MODEL_ORDER if m in best["model"].values]
    x = np.arange(len(models))
    width = 0.25
    offsets = [-width, 0, width]

    for i, tgt in enumerate(TARGETS):
        vals = []
        for m in models:
            row = best[(best["model"] == m) & (best["target"] == tgt)]
            vals.append(float(row["r2"].iloc[0]) if len(row) else np.nan)
        bars = ax.bar(x + offsets[i], vals, width, label=TARGET_LABELS[tgt],
                      color=TARGET_COLORS[tgt], alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title("(A) Tabular baselines — best feature tier per model", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(-0.2, 0.75)


def panel_b(ax, tab: pd.DataFrame):
    """Line plot: LightGBM R² across feature tiers × target."""
    lgb = tab[tab["model"] == "LightGBM"]
    for tgt in TARGETS:
        sub = lgb[lgb["target"] == tgt].set_index("feature_set").reindex(TIER_ORDER)
        ax.plot(range(len(TIER_ORDER)), sub["r2"].values, "o-",
                color=TARGET_COLORS[tgt], lw=2, ms=6, label=TARGET_LABELS[tgt])

    ax.set_xticks(range(len(TIER_ORDER)))
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER], fontsize=8)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title("(B) LightGBM — feature tier ablation", fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(-0.1, 0.75)


def panel_c(ax, tab: pd.DataFrame, patch: pd.DataFrame, multi: pd.DataFrame):
    """Dot plot: R² across model families, chla only for clarity."""
    # LightGBM static as anchor
    records = []

    # Tabular best
    best_tab = _best_tier(tab)
    for _, row in best_tab[best_tab["target"] == "chla_ug_l"].iterrows():
        records.append({"label": row["model"], "r2": row["r2"], "family": "Tabular"})

    # Patch (chla)
    for _, row in patch[patch["target"] == "chla_ug_l"].iterrows():
        if row["model"] == "CNN+Tabular":
            continue  # exclude unstable
        records.append({"label": row["model"], "r2": row["r2"], "family": "Patch"})

    # Multimodal (chla)
    for _, row in multi[multi["target"] == "chla_ug_l"].iterrows():
        records.append({"label": row["model"], "r2": row["r2"], "family": "Multimodal"})

    df = pd.DataFrame(records)
    family_colors = {"Tabular": "#2980b9", "Patch": "#e67e22", "Multimodal": "#27ae60"}
    family_markers = {"Tabular": "o", "Patch": "s", "Multimodal": "^"}

    for i, row in enumerate(df.itertuples()):
        ax.scatter(row.r2, i, color=family_colors[row.family],
                   marker=family_markers[row.family], s=80, zorder=3)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"].tolist(), fontsize=8)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("R²  (Chl-a)", fontsize=10)
    ax.set_title("(C) All model families — Chl-a R² comparison", fontweight="bold")
    ax.set_facecolor("white")
    ax.grid(axis="x", alpha=0.3, zorder=0)

    patches = [mpatches.Patch(color=c, label=f) for f, c in family_colors.items()]
    ax.legend(handles=patches, fontsize=8, framealpha=0.9)


def panel_d(ax, conf: pd.DataFrame):
    """Calibration: PICP vs nominal coverage for LightGBM × target."""
    nominal = [0.80, 0.90, 0.95]
    lgb = conf[conf["model"] == "LightGBM"]

    for tgt in TARGETS:
        sub = lgb[lgb["target"] == tgt].sort_values("nominal_coverage")
        picps = sub["picp"].values
        ax.plot(nominal, picps, "o-", color=TARGET_COLORS[tgt],
                lw=2, ms=6, label=TARGET_LABELS[tgt])

    ax.plot([0.79, 0.96], [0.79, 0.96], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Nominal coverage", fontsize=10)
    ax.set_ylabel("Empirical coverage (PICP)", fontsize=10)
    ax.set_title("(D) Split conformal calibration — LightGBM", fontweight="bold")
    ax.set_xlim(0.78, 0.97); ax.set_ylim(0.70, 1.00)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_facecolor("white")
    ax.grid(alpha=0.3, zorder=0)
    ax.set_xticks(nominal)
    ax.set_xticklabels(["80%", "90%", "95%"])


def main():
    tab   = pd.read_csv("data/outputs/baseline_model_results.csv")
    patch = pd.read_csv("data/outputs/patch_baseline_results.csv")
    multi = pd.read_csv("data/outputs/multimodal_baseline_results.csv")
    conf  = pd.read_csv("data/outputs/baseline_conformal_results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.38, wspace=0.32)

    panel_a(axes[0, 0], tab)
    panel_b(axes[0, 1], tab)
    panel_c(axes[1, 0], tab, patch, multi)
    panel_d(axes[1, 1], conf)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
