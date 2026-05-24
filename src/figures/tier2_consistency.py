#!/usr/bin/env python3
"""
Tier 2 — Internal consistency across redundant modalities.

Panel A: Pairwise scatter matrix — in-situ Chl-a vs MODIS OC4 vs Landsat OC4 vs S2 NDCI
         at coincident station-dates within ±1 day matchup window.
Panel B: Negative reflectance rate at 443/490 nm by water colour class —
         diagnostic for AC failure over CDOM-rich waters (MD-1 quantification).
         Sensors shown: MODIS and Landsat (both carry rw_blue_negative_flag).
         Sentinel-2 excluded — no blue-band AC failure flag.
Panel C: Turbidity agreement — in-situ (Turb_FNU) vs Nechad NIR (MODIS).
Panel D: Derived Secchi vs measured Secchi — validates the 1.7/Kd approximation.
Panel E: Humic water challenge — satellite vs in-situ Chl-a bias stratified by
         water colour class (<30, 30–100, >100 mg Pt/L).
Panel F: Matchup window sensitivity — performance (R²) vs time window width (0–7 days).
Panel G: AC failure impact — RMSE and R² with vs without rw_blue_negative_flag=1
         per sensor. Quantifies MD-1: "RMSE doubles when atmospheric correction fails."
Panel H: Inter-sensor harmonization — co-observed Chl-a agreement between sensor pairs
         (MODIS vs Landsat vs S2) at coincident stations within ±3 days.

Additional outputs:
  data/outputs/table4_satellite_wq_benchmark_stratified.csv
         — RMSE/MAE/r/MBE/n per (sensor × variable × AC flag) for paper Table 4.

Usage:
    python -m src.evaluation.tier2_consistency
    (Requires: SLU MVM re-extracted with fixed codes + satellite stations Parquets)
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
OUT_CSV = Path("data/outputs")

MATCHUP_WINDOW_DAYS = 1   # ±1 day default


# ── loaders ───────────────────────────────────────────────────────────────────

def _load_slu_chemistry() -> pd.DataFrame:
    frames = []
    for p in sorted((RAW / "slu_mvm").glob("slu_mvm_chemistry_*.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["sampling_date"])
    df["station_id"] = df["station_id"].astype(str)
    return df


def _load_satellite(sensor: str) -> pd.DataFrame:
    """Load stations-mode satellite Parquet for a given sensor."""
    frames = []
    for p in sorted((RAW / "satellite" / "SE").glob(f"sweden_{sensor}_*_stations.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    # Satellite parquets store station_id as float ("100.0"); SLU MVM uses int string ("100")
    df["station_id"] = df["station_id"].astype(float).astype(int).astype(str)
    return df


def _matchup(slu: pd.DataFrame, sat: pd.DataFrame,
             slu_col: str, sat_col: str,
             window_days: int = 1) -> pd.DataFrame:
    """Match SLU MVM rows to satellite rows within ±window_days."""
    if slu_col not in slu.columns or sat_col not in sat.columns:
        return pd.DataFrame()

    slu_sub = slu.dropna(subset=[slu_col])[["station_id", "date", slu_col,
                                             "colour_mg_pt_l"]].copy()
    sat_sub = sat.dropna(subset=[sat_col])[["station_id", "date", sat_col,
                                             "rw_blue_negative_flag",
                                             "ac_method"]].copy() \
              if "rw_blue_negative_flag" in sat.columns else \
              sat.dropna(subset=[sat_col])[["station_id", "date", sat_col]].copy()

    # Merge then filter by date window
    merged = slu_sub.merge(sat_sub, on="station_id", suffixes=("_lab", "_sat"))
    merged["dt"] = (merged["date_sat"] - merged["date_lab"]).abs().dt.days
    return merged[merged["dt"] <= window_days].copy()


def _r2(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    r, _ = stats.pearsonr(x[mask], y[mask])
    return r**2


def _sat_matchup(sat_a: pd.DataFrame, col_a: str,
                 sat_b: pd.DataFrame, col_b: str,
                 window_days: int = 3) -> pd.DataFrame:
    """
    Match two satellite DataFrames on station + date within ±window_days.
    Uses year-month bucket to avoid cartesian explosion (MODIS has 82M rows).
    Always returns columns val_a / val_b / date_a / date_b / dt regardless
    of whether col_a == col_b, avoiding pandas suffix ambiguity.
    """
    if col_a not in sat_a.columns or col_b not in sat_b.columns:
        return pd.DataFrame()

    # Rename value columns to val_a / val_b before merge — no suffix needed
    a = (sat_a.dropna(subset=[col_a])[["station_id", "date", col_a]]
         .rename(columns={col_a: "val_a", "date": "date_a"}).copy())
    b = (sat_b.dropna(subset=[col_b])[["station_id", "date", col_b]]
         .rename(columns={col_b: "val_b", "date": "date_b"}).copy())

    # Integer 28-day bucket for efficient joining
    epoch = pd.Timestamp("2000-01-01")
    a["_m"] = ((a["date_a"] - epoch).dt.days // 28).astype("int32")
    b["_m"] = ((b["date_b"] - epoch).dt.days // 28).astype("int32")

    # Merge same-bucket + adjacent bucket to catch cross-boundary pairs
    frames = []
    for delta in (0, 1):
        tmp = a.merge(b.assign(_m=b["_m"] - delta), on=["station_id", "_m"])
        frames.append(tmp)
        if delta:
            tmp2 = a.assign(_m=a["_m"] - delta).merge(b, on=["station_id", "_m"])
            frames.append(tmp2)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True).drop(columns=["_m"])
    merged["dt"] = (merged["date_a"] - merged["date_b"]).abs().dt.days
    return merged[merged["dt"] <= window_days].drop_duplicates().dropna().copy()


def _rmse(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


# ── panels ─────────────────────────────────────────────────────────────────────

def panel_chla_pairwise(ax, slu: pd.DataFrame, modis: pd.DataFrame,
                         landsat: pd.DataFrame, s2: pd.DataFrame):
    """Panel A: Chl-a cross-sensor pairwise scatter."""
    if "chla_ug_l" not in slu.columns:
        ax.text(0.5, 0.5, "chla_ug_l missing from SLU MVM\n(re-run fetch_slu_mvm.py)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(A) Chl-a cross-sensor (SLU MVM data pending)")
        return

    pairs = []
    for name, sat, col in [
        ("MODIS OC4",    modis,   "chl_a_ug_l"),
        ("Landsat OC4",  landsat, "chl_a_ug_l"),
        ("S2 NDCI",      s2,      "chl_a_ndci"),
    ]:
        m = _matchup(slu, sat, "chla_ug_l", col)
        if len(m) > 10:
            pairs.append((name, m["chla_ug_l"].values, m[col].values))

    if not pairs:
        ax.text(0.5, 0.5, "No coincident matchups found yet\n(satellite extractions running)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(A) Chl-a cross-sensor (satellite running)")
        return

    colors = ["#27ae60", "#e74c3c", "#2980b9"]
    for (name, lab, sat_val), color in zip(pairs, colors):
        ax.scatter(lab, sat_val, s=8, alpha=0.4, color=color,
                   label=f"{name} (n={len(lab):,}, R²={_r2(lab, sat_val):.2f})",
                   rasterized=True)

    lim = (0.1, 200)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("In-situ Chl-a (µg/L)")
    ax.set_ylabel("Satellite Chl-a (µg/L)")
    ax.set_title("(A) Chl-a cross-sensor agreement (±1 day matchup)", fontweight="bold")
    ax.legend(fontsize=7)


def panel_negative_reflectance(ax, modis: pd.DataFrame, landsat: pd.DataFrame,
                                slu: pd.DataFrame):
    """Panel B: AC failure rate by water colour class — MODIS and Landsat only.

    Sentinel-2 is excluded: it does not carry rw_blue_negative_flag.
    """
    results = []

    for name, sat in [("MODIS", modis), ("Landsat", landsat)]:
        if sat.empty or "rw_blue_negative_flag" not in sat.columns:
            continue
        sat_col = sat.merge(
            slu[["station_id", "date", "colour_mg_pt_l"]].dropna(),
            on=["station_id", "date"], how="left"
        )
        sat_col["colour_class"] = pd.cut(
            sat_col["colour_mg_pt_l"],
            bins=[0, 30, 100, 9999],
            labels=["Clear\n(<30 mg Pt/L)", "Moderate\n(30–100 mg Pt/L)", "Humic\n(>100 mg Pt/L)"],
        )
        for cc, grp in sat_col.groupby("colour_class", observed=True):
            neg_rate = grp["rw_blue_negative_flag"].mean()
            results.append({"sensor": name, "colour_class": str(cc),
                            "neg_rate": neg_rate, "n": len(grp)})

    if not results:
        ax.text(0.5, 0.5, "Satellite data not yet available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(B) AC failure rate (satellite running)")
        return

    df = pd.DataFrame(results)
    colour_classes = ["Clear\n(<30 mg Pt/L)", "Moderate\n(30–100 mg Pt/L)", "Humic\n(>100 mg Pt/L)"]
    x = np.arange(len(colour_classes))
    width = 0.35
    sensor_order = [s for s in ["MODIS", "Landsat"] if s in df["sensor"].unique()]
    colors = {"MODIS": "#e74c3c", "Landsat": "#2980b9"}

    for i, sensor in enumerate(sensor_order):
        grp = df[df["sensor"] == sensor].set_index("colour_class").reindex(colour_classes)
        ax.bar(x + i * width, grp["neg_rate"].fillna(0) * 100, width,
               label=sensor, color=colors[sensor], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(colour_classes, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("AC failure rate (%)")
    ax.set_title("(B) AC failure rate by water colour class\n(MODIS vs Landsat)",
                 fontweight="bold")
    ax.legend(fontsize=8)


def panel_turbidity(ax, slu: pd.DataFrame, modis: pd.DataFrame):
    """Panel C: In-situ turbidity vs satellite Nechad NIR."""
    if "turbidity_fnu" not in slu.columns:
        ax.text(0.5, 0.5, "turbidity_fnu not in SLU MVM\n(check fetch_slu_mvm codes)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(C) Turbidity validation (SLU data pending)")
        return

    m = _matchup(slu, modis, "turbidity_fnu", "turbidity_fnu")
    if len(m) < 10:
        ax.text(0.5, 0.5, f"Only {len(m)} matchups found (satellite running)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(C) Turbidity: in-situ vs MODIS Nechad (satellite running)")
        return

    lab = m["turbidity_fnu_lab"].values
    sat = m["turbidity_fnu_sat"].values
    ax.scatter(lab, sat, s=8, alpha=0.4, color="#e67e22", rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    lim = (0.1, 500)
    ax.plot(lim, lim, "k--", lw=0.8, label="1:1")
    r2 = _r2(np.log(lab + 0.01), np.log(sat + 0.01))
    ax.text(0.05, 0.92, f"R²={r2:.3f}  n={len(m):,}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("In-situ turbidity (FNU)")
    ax.set_ylabel("MODIS Nechad turbidity (FNU)")
    ax.set_title("(C) Turbidity: in-situ vs Nechad NIR retrieval", fontweight="bold")
    ax.legend(fontsize=8)


def panel_secchi_validation(ax, slu: pd.DataFrame):
    """Panel D: Derived Secchi (1.7/Kd) vs measured Secchi."""
    if "secchi_m" not in slu.columns or "turbidity_fnu" not in slu.columns:
        ax.text(0.5, 0.5, "secchi_m or turbidity_fnu not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(D) Derived vs measured Secchi (data pending)")
        return

    df = slu.dropna(subset=["secchi_m", "turbidity_fnu"]).copy()
    df["kd"]             = df["turbidity_fnu"] * 0.015 + 0.04
    df["secchi_derived"] = (1.7 / df["kd"]).clip(upper=20)

    if len(df) < 10:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    ax.scatter(df["secchi_m"], df["secchi_derived"],
               s=6, alpha=0.3, color="#8e44ad", rasterized=True)
    lim = (0, 15)
    ax.plot(lim, lim, "k--", lw=0.8, label="1:1")

    mask = np.isfinite(df["secchi_m"]) & np.isfinite(df["secchi_derived"])
    r2 = _r2(df["secchi_m"].values, df["secchi_derived"].values)
    bias = (df["secchi_derived"] - df["secchi_m"]).mean()
    ax.text(0.05, 0.92, f"R²={r2:.3f}  bias={bias:+.2f} m  n={mask.sum():,}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    lim = (0, min(20, max(df["secchi_m"].max(), df["secchi_derived"].max()) * 1.05))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Measured Secchi depth (m)")
    ax.set_ylabel("Derived Secchi = 1.7/Kd (m)")
    ax.set_title("(D) Secchi derivation validation (Kratzer 2008)", fontweight="bold")
    ax.legend(fontsize=8)


def panel_humic_challenge(ax, slu: pd.DataFrame, s2: pd.DataFrame):
    """Panel E: Satellite vs in-situ Chl-a bias stratified by water colour."""
    if "chla_ug_l" not in slu.columns or s2.empty:
        ax.text(0.5, 0.5, "Chl-a or S2 data not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(E) Humic lake challenge (data pending)")
        return

    m = _matchup(slu, s2, "chla_ug_l", "chl_a_ug_l")
    if len(m) < 20:
        ax.text(0.5, 0.5, f"Only {len(m)} matchups — more satellite data needed",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(E) Humic lake challenge (satellite running)")
        return

    m["colour_class"] = pd.cut(
        m["colour_mg_pt_l"],
        bins=[0, 30, 100, 9999],
        labels=["Clear (<30)", "Moderate (30-100)", "Humic (>100)"],
    )
    m["log_bias"] = np.log10(m["chl_a_ug_l"].clip(0.01)) - np.log10(m["chla_ug_l"].clip(0.01))
    # log_bias > 0 means satellite overestimates relative to in-situ

    m.boxplot(column="log_bias", by="colour_class", ax=ax,
              patch_artist=True,
              boxprops=dict(facecolor="#2980b9", alpha=0.6))
    # pandas boxplot(by=...) adds an automatic suptitle — suppress it
    ax.get_figure().suptitle("")
    ax.axhline(0, color="red", lw=1.5, linestyle="--")
    ax.set_xlabel("Water colour class (CDOM indicator)")
    ax.set_ylabel("log₁₀(S2 Chl-a / in-situ Chl-a)")
    ax.set_title("(E) Humic lake challenge — S2 vs in-situ Chl-a bias", fontweight="bold")


def panel_matchup_sensitivity(ax, slu: pd.DataFrame, modis: pd.DataFrame):
    """Panel F: R² vs matchup window width (0–7 days)."""
    if "chla_ug_l" not in slu.columns or modis.empty:
        ax.text(0.5, 0.5, "Data not available for sensitivity analysis",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(F) Matchup window sensitivity (data pending)")
        return

    windows = range(0, 8)
    results = []
    for w in windows:
        m = _matchup(slu, modis, "chla_ug_l", "chl_a_ug_l", window_days=w)
        if len(m) > 10:
            r2 = _r2(np.log(m["chla_ug_l"].clip(0.01)),
                     np.log(m["chl_a_ug_l"].clip(0.01)))
            results.append({"window": w, "r2": r2, "n": len(m)})
        else:
            results.append({"window": w, "r2": np.nan, "n": len(m)})

    df = pd.DataFrame(results)
    ax2 = ax.twinx()
    ax.plot(df["window"], df["r2"], "o-", color="steelblue", lw=2, ms=6, label="R²")
    ax2.bar(df["window"], df["n"], alpha=0.3, color="gray", label="n matchups")
    ax.set_xlabel("Matchup window (±days)")
    ax.set_ylabel("R² (log Chl-a: in-situ vs MODIS)", color="steelblue")
    ax2.set_ylabel("Number of matchups", color="gray")
    ax.set_title("(F) Matchup window sensitivity — R² vs temporal tolerance",
                 fontweight="bold")
    ax.set_xticks(list(windows))
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labels1, fontsize=8)


# ── New panels G & H ─────────────────────────────────────────────────────────

def panel_ac_failure_impact(ax, slu: pd.DataFrame, modis: pd.DataFrame,
                             landsat: pd.DataFrame, s2: pd.DataFrame):
    """
    Panel G: RMSE (log-scale Chl-a) split by atmospheric correction (AC) outcome.
    Shows how AC failure affects retrieval accuracy per sensor.
    Sensors without rw_blue_negative_flag are shown as overall-only bars.
    """
    configs = [
        ("MODIS",      modis,   "chl_a_ug_l"),
        ("Landsat",    landsat, "chl_a_ug_l"),
        ("Sentinel-2", s2,      "chl_a_ndci"),
    ]
    results = []
    for sensor, sat, sat_col in configs:
        if sat.empty:
            continue
        m = _matchup(slu, sat, "chla_ug_l", sat_col, window_days=3)
        if len(m) < 20:
            continue
        has_flag = "rw_blue_negative_flag" in m.columns
        flag_groups = [(0, "AC OK"), (1, "AC failure")] if has_flag else [(-1, "All observations")]
        for flag_val, label in flag_groups:
            sub = m if flag_val == -1 else m[m["rw_blue_negative_flag"] == flag_val]
            if len(sub) < 5:
                continue
            la = np.log(sub["chla_ug_l"].clip(0.01).values)
            sv = np.log(sub[sat_col].clip(0.01).values)
            ok = np.isfinite(la) & np.isfinite(sv)
            if ok.sum() < 5:
                continue
            results.append({
                "sensor": sensor, "flag": label,
                "rmse_log": _rmse(la, sv),
                "r2": _r2(la, sv),
                "n": int(ok.sum()),
            })

    if not results:
        ax.text(0.5, 0.5, "No matchup data available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(G) Atmospheric correction failure impact (data pending)")
        return

    df = pd.DataFrame(results)
    sensors = df["sensor"].unique()
    all_flags = ["AC OK", "AC failure", "All observations"]
    present_flags = [f for f in all_flags if f in df["flag"].values]
    x = np.arange(len(sensors))
    width = 0.25
    colors = {"AC OK": "#2ecc71", "AC failure": "#e74c3c", "All observations": "#7f8c8d"}

    for i, flag_label in enumerate(present_flags):
        sub = df[df["flag"] == flag_label].set_index("sensor").reindex(sensors)
        bars = ax.bar(x + i * width, sub["rmse_log"].values, width,
                      label=flag_label, color=colors[flag_label], alpha=0.85,
                      edgecolor="white")
        for bar, row in zip(bars, sub.itertuples()):
            if not np.isnan(row.rmse_log):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"n={row.n:,}\nR²={row.r2:.2f}",
                        ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x + width * (len(present_flags) - 1) / 2)
    ax.set_xticklabels(sensors, fontsize=10)
    ax.set_ylabel("RMSE (log Chl-a, µg/L)", fontsize=10)
    ax.set_title("(G) Atmospheric correction failure impact on Chl-a retrieval",
                 fontweight="bold")
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=8, framealpha=0.9)


def panel_intersensor_harmonization(ax, modis: pd.DataFrame,
                                     landsat: pd.DataFrame, s2: pd.DataFrame):
    """
    Panel H: Satellite-to-satellite Chl-a agreement on co-observed stations (±3 days).
    Each point is a station observed by two sensors within 3 days of each other.
    """
    pairs = [
        ("MODIS",      modis,   "chl_a_ug_l", "Landsat",    landsat, "chl_a_ug_l"),
        ("MODIS",      modis,   "chl_a_ug_l", "Sentinel-2", s2,      "chl_a_ndci"),
        ("Landsat",    landsat, "chl_a_ug_l", "Sentinel-2", s2,      "chl_a_ndci"),
    ]
    colors = ["#e74c3c", "#2980b9", "#27ae60"]
    plotted = False
    lim_min, lim_max = np.inf, -np.inf

    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25, zorder=0)

    for (n_a, sa, c_a, n_b, sb, c_b), color in zip(pairs, colors):
        m = _sat_matchup(sa, c_a, sb, c_b, window_days=3)
        if len(m) < 10:
            continue
        va = np.log(m["val_a"].clip(0.01).values)
        vb = np.log(m["val_b"].clip(0.01).values)
        ok = np.isfinite(va) & np.isfinite(vb)
        if ok.sum() < 10:
            continue
        r2 = _r2(va[ok], vb[ok])
        rmse = _rmse(va[ok], vb[ok])
        ax.scatter(va[ok], vb[ok], s=4, alpha=0.3, color=color, rasterized=True,
                   label=f"{n_a} vs {n_b}  (R²={r2:.2f}, RMSE={rmse:.2f}, n={ok.sum():,})")
        lim_min = min(lim_min, va[ok].min(), vb[ok].min())
        lim_max = max(lim_max, va[ok].max(), vb[ok].max())
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No co-observed sensor pairs found (±3 days)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("(H) Satellite cross-sensor consistency (data pending)")
        return

    pad = 0.3
    lim = (lim_min - pad, lim_max + pad)
    ax.plot(lim, lim, "k--", lw=0.8, zorder=5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("log Chl-a (µg/L) — first sensor", fontsize=10)
    ax.set_ylabel("log Chl-a (µg/L) — second sensor", fontsize=10)
    ax.set_title("(H) Cross-sensor Chl-a consistency — co-observed stations (±3 days)",
                 fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9, loc="upper left")


# ── Stratified validation table ───────────────────────────────────────────────

def _save_stratified_table(slu: pd.DataFrame, modis: pd.DataFrame,
                            landsat: pd.DataFrame, s2: pd.DataFrame) -> None:
    """
    Write table4_satellite_wq_benchmark_stratified.csv:
    RMSE / MAE / Pearson-r / MBE / n per (sensor × variable × AC flag).
    AC flag: 'all' | 'ac_ok' (flag=0) | 'ac_failure' (flag=1).
    """
    configs = [
        ("MODIS",   modis,   [("chla",      "chla_ug_l",     "chl_a_ug_l"),
                               ("turbidity", "turbidity_fnu", "turbidity_fnu"),
                               ("secchi",    "secchi_m",      "secchi_m")]),
        ("Landsat", landsat, [("chla",      "chla_ug_l",     "chl_a_ug_l")]),
        ("S2",      s2,      [("chla_ndci", "chla_ug_l",     "chl_a_ndci"),
                               ("secchi",    "secchi_m",      "secchi_m")]),
    ]
    rows = []
    for sensor, sat, wq_pairs in configs:
        for var_name, lab_col, sat_col in wq_pairs:
            m = _matchup(slu, sat, lab_col, sat_col, window_days=3)
            if len(m) < 5:
                continue
            flag_groups = [("all", m)]
            if "rw_blue_negative_flag" in m.columns:
                flag_groups += [
                    ("ac_ok",      m[m["rw_blue_negative_flag"] == 0]),
                    ("ac_failure", m[m["rw_blue_negative_flag"] == 1]),
                ]
            for flag_label, sub in flag_groups:
                if len(sub) < 5:
                    continue
                # When lab_col == sat_col pandas adds _lab/_sat suffixes after merge
                actual_lab = (lab_col + "_lab") if (lab_col + "_lab") in sub.columns else lab_col
                actual_sat = (sat_col + "_sat") if (sat_col + "_sat") in sub.columns else sat_col
                if actual_lab not in sub.columns or actual_sat not in sub.columns:
                    continue
                lab = sub[actual_lab].values
                sv  = sub[actual_sat].values
                ok  = np.isfinite(lab) & np.isfinite(sv) & (lab > 0) & (sv > 0)
                if ok.sum() < 5:
                    continue
                lo, so = lab[ok], sv[ok]
                r, _   = stats.pearsonr(np.log(lo.clip(1e-3)), np.log(so.clip(1e-3)))
                rows.append({
                    "sensor":   sensor,
                    "variable": var_name,
                    "ac_flag":  flag_label,
                    "rmse":     round(float(np.sqrt(np.mean((lo - so) ** 2))), 3),
                    "mae":      round(float(np.mean(np.abs(lo - so))), 3),
                    "mbe":      round(float(np.mean(so - lo)), 3),
                    "pearson_r": round(float(r), 3),
                    "n":        int(ok.sum()),
                })

    if rows:
        out = OUT_CSV / "table4_satellite_wq_benchmark_stratified.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info(f"Saved → {out}")
    else:
        logger.warning("No matchup data for stratified table — satellite data may be missing")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SLU MVM chemistry")
    slu = _load_slu_chemistry()
    logger.info(f"  {len(slu):,} rows, chla_ug_l: "
                f"{slu['chla_ug_l'].notna().sum() if 'chla_ug_l' in slu.columns else 0:,}")

    logger.info("Loading satellite Parquets")
    modis   = _load_satellite("modis")
    landsat = _load_satellite("landsat")
    s2      = _load_satellite("sentinel2")
    logger.info(f"  MODIS: {len(modis):,}  Landsat: {len(landsat):,}  S2: {len(s2):,}")

    fig = plt.figure(figsize=(20, 24), facecolor="white")
    fig.suptitle("Tier 2 — Internal Consistency Across Redundant Modalities",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    panel_chla_pairwise(fig.add_subplot(gs[0, 0]),          slu, modis, landsat, s2)
    panel_negative_reflectance(fig.add_subplot(gs[0, 1]),   modis, landsat, slu)
    panel_turbidity(fig.add_subplot(gs[1, 0]),              slu, modis)
    panel_secchi_validation(fig.add_subplot(gs[1, 1]),      slu)
    panel_humic_challenge(fig.add_subplot(gs[2, 0]),        slu, s2)
    panel_matchup_sensitivity(fig.add_subplot(gs[2, 1]),    slu, modis)
    panel_ac_failure_impact(fig.add_subplot(gs[3, 0]),      slu, modis, landsat, s2)
    panel_intersensor_harmonization(fig.add_subplot(gs[3, 1]), modis, landsat, s2)

    out_path = OUT / "fig_tier2_consistency.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved → {out_path}")
    plt.close(fig)

    logger.info("Saving stratified validation table")
    _save_stratified_table(slu, modis, landsat, s2)


if __name__ == "__main__":
    main()
