#!/usr/bin/env python3
"""
SHARED-STEP-3 — Multi-parameter gridded water quality NetCDF via IDW and RF.

Interpolates 9 WQ parameters from SLU MVM station observations onto a 5 km
SWEREF99 TM regular grid using two methods:
  IDW : Inverse-distance weighting (power=2, r=50 km) — reference baseline,
        fully reproducible, requires no covariates.
  RF  : Random Forest spatial regression using 13 HydroATLAS catchment covariates
        (land use, climate, hydrology) + SWEREF99 easting/northing as predictors.
        Produces primary gridded product with spatial leave-group-out LOOCV RMSE.

Use --method [idw|rf|both] (default: both).

Parameters gridded:
  tsi          — Carlson Trophic State Index (composite)
  tp_ug_l      — Total phosphorus (µg L⁻¹)
  colour_mg_pt_l — Water colour / CDOM (mg Pt L⁻¹)
  tn_ug_l      — Total nitrogen (µg L⁻¹)
  np_molar_ratio — Molar N:P ratio
  secchi_m     — Secchi depth (m)
  chla_ug_l    — Chlorophyll-a (µg L⁻¹)
  ph           — pH
  do_pct_sat   — Dissolved oxygen saturation (%)

For RF grid prediction, HydroATLAS covariates at each grid cell are approximated
by nearest-station assignment (acceptable given large-scale spatial autocorrelation
of catchment attributes; noted as v1 limitation).

Input : data/interim/SE/derived_wq_timeseries.parquet
Output: data/outputs/nordic_water_quality_v1.nc

Usage:
    python -m src.processing.build_wqi_grid
    python -m src.processing.build_wqi_grid --method rf
    python -m src.processing.build_wqi_grid --resolution-km 5 --radius-km 50
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INTERIM   = Path("data/interim/SE")
OUT_DIR   = Path("data/outputs")
ATLAS_PATH = Path("data/raw/hydroatlas/hydroatlas_station_covariates_SE.parquet")

SWEDEN_BBOX_3006 = {
    "x_min": 260_000, "x_max": 920_000,
    "y_min": 6_130_000, "y_max": 7_680_000,
}

# Parameter registry — (ts_column, long_name, units, valid_range, min_stations)
PARAMETERS = [
    ("tsi",            "Carlson Trophic State Index (composite)",    "dimensionless",  [0,   100], 50),
    ("tp_ug_l",        "Total phosphorus",                           "µg L⁻¹",         [0,  5000], 50),
    ("colour_mg_pt_l", "Water colour (CDOM proxy)",                  "mg Pt L⁻¹",      [0,  2000], 30),
    ("tn_ug_l",        "Total nitrogen",                             "µg L⁻¹",         [0, 50000], 30),
    ("np_molar_ratio", "Molar N:P ratio",                            "dimensionless",  [0,   500], 30),
    ("secchi_m",       "Secchi depth",                               "m",              [0,    20], 10),
    ("chla_ug_l",      "Chlorophyll-a",                              "µg L⁻¹",         [0,   500], 10),
    ("ph",             "pH",                                         "1",              [2,    12], 20),
    ("do_pct_sat",     "Dissolved oxygen saturation",                "%",              [0,   200], 20),
]

# HydroATLAS features used as RF predictors (catchment attributes + spatial position)
HYDROATLAS_FEATURES = [
    "for_pc_use", "crp_pc_use", "pst_pc_use", "pre_mm_uyr", "ele_mt_uav",
    "soc_th_uav", "dis_m3_pyr", "tmp_dc_uyr", "wet_pc_ug1", "ari_ix_uav",
    "snw_pc_uyr", "slp_dg_uav", "cly_pc_uav",
]


# ── Grid helpers ──────────────────────────────────────────────────────────────

def _to_sweref99(lon: np.ndarray, lat: np.ndarray):
    t = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
    return t.transform(lon, lat)


def _make_grid(res_m: float):
    bb = SWEDEN_BBOX_3006
    xs = np.arange(bb["x_min"], bb["x_max"] + res_m, res_m)
    ys = np.arange(bb["y_min"], bb["y_max"] + res_m, res_m)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return xs, ys, grid_x, grid_y


# ── IDW ───────────────────────────────────────────────────────────────────────

def _idw(tree: cKDTree, values: np.ndarray,
         grid_xy: np.ndarray, k: int = 12, power: float = 2.0,
         radius: float = 50_000.0) -> np.ndarray:
    k = min(k, len(values))   # cap to available stations
    dists, idxs = tree.query(grid_xy, k=k, workers=-1)
    dists = np.where(dists <= radius, dists, np.inf)
    weights = np.where(dists == 0, np.inf,
                       np.where(np.isinf(dists), 0.0, 1.0 / dists ** power))
    w_sum = weights.sum(axis=1, keepdims=True)
    result = np.full(len(grid_xy), np.nan)
    valid = w_sum.ravel() > 0
    w_norm = np.where(w_sum > 0, weights / w_sum, 0.0)
    result[valid] = (w_norm[valid] * values[idxs[valid]]).sum(axis=1)
    return result


# ── RF regression ─────────────────────────────────────────────────────────────

def _load_atlas() -> pd.DataFrame:
    if not ATLAS_PATH.exists():
        logger.warning("HydroATLAS not found — RF method unavailable, falling back to IDW")
        return pd.DataFrame()
    df = pd.read_parquet(ATLAS_PATH)
    df["station_id"] = df["station_id"].astype(str)
    return df


def _build_feature_matrix(param_df: pd.DataFrame,
                           ha_df: pd.DataFrame,
                           use_cols: list) -> tuple:
    """
    Build (X, y, station_x, station_y) for RF training.
    param_df must have: station_id, value, latitude, longitude.
    ha_df must have: station_id + columns in use_cols.
    Returns (X, y, sx, sy) or None if too few stations.
    """
    merged = param_df.merge(ha_df[["station_id"] + use_cols], on="station_id", how="left")
    sx, sy = _to_sweref99(merged["longitude"].values, merged["latitude"].values)

    feat_cols = []
    for c in use_cols:
        col = merged[c].fillna(merged[c].median() if merged[c].notna().any() else 0.0)
        feat_cols.append(col.values.astype(np.float32))
    feat_cols += [sx.astype(np.float32), sy.astype(np.float32)]

    X = np.column_stack(feat_cols)
    y = merged["value"].values.astype(np.float32)
    return X, y, sx, sy


def _rf_predict_grid(param_df: pd.DataFrame,
                     ha_df: pd.DataFrame,
                     grid_pts: np.ndarray,
                     use_cols: list,
                     n_folds: int = 5) -> tuple:
    """
    Fit RF on station WQ observations, predict to grid.
    Returns (grid_values, loocv_rmse).
    Grid-cell HydroATLAS covariates assigned by nearest station.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold

    result = _build_feature_matrix(param_df, ha_df, use_cols)
    X, y, sx, sy = result

    # Spatial cross-validation groups: 1° lat bins
    merged_lats = param_df["latitude"].values
    groups = (merged_lats * 1.0).astype(int)  # ~111 km groups along lat
    unique_groups = np.unique(groups)
    n_splits = min(n_folds, len(unique_groups))

    rf = RandomForestRegressor(
        n_estimators=100, max_features="sqrt",
        min_samples_leaf=5, random_state=42, n_jobs=-1,
    )

    loocv_rmse = np.nan
    if len(X) >= 10 and n_splits >= 2:
        preds = np.full(len(y), np.nan)
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf.split(X, y, groups):
            rf.fit(X[train_idx], y[train_idx])
            preds[val_idx] = rf.predict(X[val_idx])
        loocv_rmse = float(np.sqrt(np.nanmean((preds - y) ** 2)))

    # Refit on all data for grid prediction
    rf.fit(X, y)

    # Build grid feature matrix: nearest-station HydroATLAS covariates per cell
    station_pts = np.column_stack([sx, sy])
    tree_st = cKDTree(station_pts)
    _, nn_idx = tree_st.query(grid_pts, k=1)

    grid_ha = ha_df.set_index("station_id").reindex(
        param_df["station_id"].values[nn_idx]
    )
    feat_grid = []
    for c in use_cols:
        col = grid_ha[c].fillna(grid_ha[c].median() if grid_ha[c].notna().any() else 0.0)
        feat_grid.append(col.values.astype(np.float32))
    feat_grid += [grid_pts[:, 0].astype(np.float32), grid_pts[:, 1].astype(np.float32)]
    X_grid = np.column_stack(feat_grid)

    return rf.predict(X_grid).astype(np.float32), loocv_rmse


# ── Core ──────────────────────────────────────────────────────────────────────

def run(res_km: float = 5.0, radius_km: float = 50.0, method: str = "both") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    do_idw = method in ("idw", "both")
    do_rf  = method in ("rf",  "both")

    ts_path = INTERIM / "derived_wq_timeseries.parquet"
    if not ts_path.exists():
        logger.error(f"Not found: {ts_path} — run derived_parameters.py first")
        return

    # Load station coordinates
    coords_path = Path("data/raw/slu_mvm/slu_mvm_measured_station_ids.csv")
    if not coords_path.exists():
        logger.error(f"Station coords not found: {coords_path}")
        return
    coords = pd.read_csv(coords_path, usecols=["station_id", "latitude", "longitude"])
    coords["station_id"] = coords["station_id"].astype(str)

    logger.info("Loading annual WQ timeseries")
    ts = pd.read_parquet(ts_path)
    ts["station_id"] = ts["station_id"].astype(str)
    ts = ts.merge(coords, on="station_id", how="left")

    years = sorted(ts["year"].dropna().unique().astype(int))
    logger.info(f"  {len(ts):,} station-years, {len(years)} years: {years[0]}–{years[-1]}")

    # Load HydroATLAS for RF
    ha_df = pd.DataFrame()
    if do_rf:
        ha_df = _load_atlas()
        if ha_df.empty:
            logger.warning("RF method disabled — HydroATLAS not found")
            do_rf = False
        else:
            # Keep only columns that exist
            available = [c for c in HYDROATLAS_FEATURES if c in ha_df.columns]
            logger.info(f"  HydroATLAS: {len(ha_df):,} stations, {len(available)}/{len(HYDROATLAS_FEATURES)} features available")
            HYDROATLAS_FEATURES[:] = available  # mutate in-place so helpers see updated list

    res_m    = res_km * 1000.0
    radius_m = radius_km * 1000.0

    logger.info(f"Building {res_km:.0f} km SWEREF99 grid (IDW radius {radius_km:.0f} km)")
    xs, ys, grid_x, grid_y = _make_grid(res_m)
    nx, ny = len(xs), len(ys)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    logger.info(f"  Grid size: {ny} × {nx} = {ny*nx:,} cells")

    # Storage per method
    param_layers_idw    = {col: [] for col, *_ in PARAMETERS}
    param_layers_rf     = {col: [] for col, *_ in PARAMETERS}
    n_station_layers    = {col: [] for col, *_ in PARAMETERS}
    loocv_rmse          = {col: {} for col, *_ in PARAMETERS}
    valid_years         = []

    for yr in years:
        grp_yr = ts[ts["year"] == yr]
        any_param_ok = False

        for col, long_name, units, valid_range, min_st in PARAMETERS:
            nan_layer = np.full((ny, nx), np.nan, dtype=np.float32)
            zero_cnt  = np.zeros((ny, nx), dtype=np.int16)

            if col not in grp_yr.columns:
                if do_idw: param_layers_idw[col].append(nan_layer)
                if do_rf:  param_layers_rf[col].append(nan_layer)
                n_station_layers[col].append(zero_cnt)
                continue

            valid = grp_yr.dropna(subset=[col, "latitude", "longitude"])
            valid = valid[(valid[col] >= valid_range[0]) &
                          (valid[col] <= valid_range[1])]

            if len(valid) < min_st:
                logger.debug(f"  {yr}/{col}: only {len(valid)} stations < {min_st} min — NaN layer")
                if do_idw: param_layers_idw[col].append(nan_layer)
                if do_rf:  param_layers_rf[col].append(nan_layer)
                n_station_layers[col].append(zero_cnt)
                continue

            s_x, s_y = _to_sweref99(valid["longitude"].values, valid["latitude"].values)
            s_pts = np.column_stack([s_x, s_y])
            tree  = cKDTree(s_pts)

            # IDW
            if do_idw:
                grid_vals = _idw(tree, valid[col].values.astype(float),
                                 grid_pts, radius=radius_m).reshape(ny, nx)
                param_layers_idw[col].append(grid_vals.astype(np.float32))

            # RF
            if do_rf and not ha_df.empty:
                try:
                    param_df = valid[["station_id", col, "latitude", "longitude"]].rename(
                        columns={col: "value"}
                    )
                    rf_vals, cv_rmse = _rf_predict_grid(
                        param_df, ha_df, grid_pts, HYDROATLAS_FEATURES
                    )
                    param_layers_rf[col].append(rf_vals.reshape(ny, nx))
                    loocv_rmse[col][yr] = cv_rmse
                except Exception as exc:
                    logger.warning(f"  {yr}/{col}: RF failed ({exc}) — NaN layer")
                    param_layers_rf[col].append(nan_layer)
            elif do_rf:
                param_layers_rf[col].append(nan_layer)

            # Station count grid
            cnt_dists, _ = tree.query(grid_pts, k=min(20, len(s_pts)), workers=-1)
            cnt_grid = (cnt_dists <= radius_m).sum(axis=1).reshape(ny, nx).astype(np.int16)
            n_station_layers[col].append(cnt_grid)
            any_param_ok = True

        if any_param_ok:
            valid_years.append(yr)

        if yr % 5 == 0 or yr == years[-1]:
            n_ok = sum(1 for c, *_ in PARAMETERS
                       if param_layers_idw.get(c) and
                       np.isfinite(param_layers_idw[c][-1]).any())
            logger.info(f"  {yr}: {n_ok} parameters OK")

    if not valid_years:
        logger.error("No valid years produced")
        return

    # ── Build xarray Dataset ──
    year_arr = np.array(valid_years, dtype=np.int32)
    dims_3d  = ["year", "y", "x"]

    xr_coords = {
        "year":     (["year"], year_arr),
        "easting":  (["y", "x"], grid_x.astype(np.float32),
                     {"units": "m", "long_name": "SWEREF99 TM easting",
                      "standard_name": "projection_x_coordinate"}),
        "northing": (["y", "x"], grid_y.astype(np.float32),
                     {"units": "m", "long_name": "SWEREF99 TM northing",
                      "standard_name": "projection_y_coordinate"}),
        "x": (["x"], xs.astype(np.float32), {"units": "m"}),
        "y": (["y"], ys.astype(np.float32), {"units": "m"}),
    }

    data_vars = {}
    methods_used = []

    for col, long_name, units, valid_range, _ in PARAMETERS:
        cnt = np.stack(n_station_layers[col], axis=0)
        data_vars[f"n_stations_{col}"] = xr.Variable(
            dims_3d, cnt,
            {"long_name": f"Stations within {radius_km:.0f} km contributing to {long_name}",
             "units": "count"},
        )

        if do_idw:
            arr = np.stack(param_layers_idw[col], axis=0)
            data_vars[col] = xr.Variable(
                dims_3d, arr,
                {"long_name": long_name,
                 "units": units,
                 "valid_min": float(valid_range[0]),
                 "valid_max": float(valid_range[1]),
                 "interpolation": f"IDW power=2 radius={radius_km:.0f}km",
                 "_FillValue": np.nan},
            )
            if "idw" not in methods_used:
                methods_used.append("idw")

        if do_rf:
            arr_rf = np.stack(param_layers_rf[col], axis=0)
            # Per-parameter mean LOOCV RMSE across years
            cv_vals = [v for v in loocv_rmse[col].values() if not np.isnan(v)]
            mean_cv = float(np.mean(cv_vals)) if cv_vals else np.nan
            data_vars[f"{col}_rf"] = xr.Variable(
                dims_3d, arr_rf,
                {"long_name": f"{long_name} (RF spatial regression)",
                 "units": units,
                 "valid_min": float(valid_range[0]),
                 "valid_max": float(valid_range[1]),
                 "method": "RandomForest_HydroATLAS_spatial_covariates",
                 "loocv_rmse": mean_cv,
                 "_FillValue": np.nan},
            )
            if "rf" not in methods_used:
                methods_used.append("rf")

    global_attrs = {
        "title":                "Sweden Gridded Water Quality — Multi-parameter",
        "institution":          "HydroImaging 2026 dataset",
        "source":               "SLU MVM chemistry (2000-2025)",
        "history":              "Produced by src/processing/build_wqi_grid.py",
        "Conventions":          "CF-1.8",
        "crs":                  "EPSG:3006 (SWEREF99 TM)",
        "methods":              ", ".join(methods_used),
        "idw_method":           f"power=2 radius={radius_km:.0f}km grid={res_km:.0f}km",
        "rf_method":            "RandomForestRegressor(n_estimators=100) + HydroATLAS covariates",
        "rf_covariates":        ", ".join(HYDROATLAS_FEATURES),
        "rf_cv_strategy":       "GroupKFold(5) on 1° latitude bins",
        "grid_resolution_m":    res_m,
        "parameters":           ", ".join(col for col, *_ in PARAMETERS),
        "seasonal_note":        "Annual layers only in v1; monthly/seasonal layers planned for v2",
    }

    ds_out = xr.Dataset(data_vars, coords=xr_coords, attrs=global_attrs)

    out_path = OUT_DIR / "nordic_water_quality_v1.nc"
    ds_out.to_netcdf(str(out_path), format="NETCDF4", engine="netcdf4")
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved → {out_path}  ({size_mb:.1f} MB)")
    logger.info(f"  Variables: {list(data_vars.keys())}")
    logger.info(f"  Dims: year={len(year_arr)}, y={ny}, x={nx}")
    logger.info(f"  Methods: {methods_used}")


def main():
    parser = argparse.ArgumentParser(description="Build multi-parameter gridded WQ NetCDF")
    parser.add_argument("--resolution-km", type=float, default=5.0)
    parser.add_argument("--radius-km",     type=float, default=50.0)
    parser.add_argument("--method",        type=str,   default="both",
                        choices=["idw", "rf", "both"],
                        help="Interpolation method: idw, rf, or both (default: both)")
    parser.add_argument("--country",       type=str,   default="SE")
    args = parser.parse_args()
    run(res_km=args.resolution_km, radius_km=args.radius_km, method=args.method)


if __name__ == "__main__":
    main()
