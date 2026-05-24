#!/usr/bin/env python3
"""
Tabular baseline models for NordWQ-26 dataset paper (CIKM 2026).

Feature-set tiers (ablation axis 1 — data sources):
  static       — HydroATLAS catchment attributes only
  met          — static + ERA5 daily meteorology
  met_mesan    — static + ERA5 + MESAN daily meteorology
  published_alg— satellite-derived WQ algorithm outputs only
                 (chl_a_oc4, chl_a_ndci, turbidity_fnu, secchi_m, cdom_m per sensor)
                 Situates the dataset against current remote-sensing practice without
                 reimplementation risk of Pahlevan 2020 / Toming 2016 architectures.
  full         — all of the above + raw satellite spectral bands (5 sensors)

Models:
  StationMean   — per-station historical mean                 [floor baseline]
  Ridge         — L2 linear regression                       [linearity baseline]
  RF            — Random Forest                              [interpretable]
  XGBoost       — Gradient Boosting (SOTA tree)              [SOTA]
  LightGBM      — Gradient Boosting (SOTA tree, fast)        [SOTA]
  MLP           — sklearn MLPRegressor                       [neural, no PyTorch]
  FT-Transformer— Gorishniy et al. NeurIPS 2021              [SOTA neural tabular]
  QuantileLGB   — LightGBM quantile regression (10/50/90th)  [uncertainty quantification]

Evaluation dimensions:
  1. Cross-validated RMSE / R² / MAE per model × feature tier × target
  2. Trophic class breakdown (oligo / meso / eu / hyper) — shows where satellite-only fails
  3. Rolling-origin temporal evaluation — train 2000-2010/test 2011-2015 etc.
  4. Sensor-coverage ablation — RMSE vs number of co-located sensors (0–5)

Targets (log1p-transformed): chla_ug_l, tp_ug_l, secchi_m

CV: GroupKFold on SMHI drainage basin fold keys.  Falls back to 1° lat bins.

Outputs:
  data/outputs/baseline_model_results.csv          — main results table
  data/outputs/baseline_model_results.tex          — LaTeX for paper
  data/outputs/baseline_trophic_results.csv        — breakdown by trophic class
  data/outputs/baseline_rolling_origin.csv         — rolling-origin temporal results
  data/outputs/baseline_sensor_coverage.csv        — RMSE vs n_sensors
  data/outputs/baseline_quantile_coverage.csv      — UQ calibration (interval coverage %)
  data/outputs/baseline_feature_importance.parquet — top-20 features (RF/XGB/LGB)

Usage:
    python -m src.evaluation.baseline_models
    python -m src.evaluation.baseline_models --models rf xgb lgb
    python -m src.evaluation.baseline_models --max-rows 100000   # FT-Trans subsample
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INTERIM      = Path("data/interim/SE")
FEATURE_DIR  = INTERIM / "feature_table"
ATLAS_PATH   = Path("data/raw/hydroatlas/hydroatlas_station_covariates_SE.parquet")
DERIVED_PATH = Path("data/interim/SE/derived_wq_params.parquet")
SPLIT_PATH   = Path("data/processed/nordic_multimodal_dataset/split_index.parquet")
OUT_DIR      = Path("data/outputs")

TARGETS = ["chla_ug_l", "tp_ug_l", "secchi_m"]

HYDROATLAS_FEATURES = [
    # Morphometry / hydrology
    "Lake_area", "Depth_avg", "Vol_total", "Res_time", "Wshd_area",
    "Elevation", "Shore_dev", "CATCH_SKM", "UPLAND_SKM",
    "ORD_STRA", "LENGTH_KM", "DIST_DN_KM", "DIS_AV_CMS",
    "dis_m3_pyr", "dis_m3_pmn", "dis_m3_pmx",
    "lkv_mc_usu", "rev_mc_usu", "dor_pc_pva", "inu_pc_umx",
    # Climate / water balance
    "ele_mt_uav", "slp_dg_uav", "tmp_dc_uyr", "tmp_dc_cmn", "tmp_dc_cmx",
    "tmp_dc_lmn", "tmp_dc_lmx", "pre_mm_uyr", "pet_mm_uyr", "aet_mm_uyr",
    "ari_ix_uav", "cmi_ix_uyr", "snw_pc_uyr", "swc_pc_uyr",
    "run_mm_cyr", "run_mm_vyr", "gwt_cm_cav", "gwt_cm_vav",
    "sgr_dk_rav", "sgr_dk_vav",
    # Land cover
    "for_pc_use", "crp_pc_use", "pst_pc_use", "urb_pc_use",
    "wet_pc_ug1", "wet_pc_ug2", "gla_pc_use", "prm_pc_use", "pac_pc_use",
    "ire_pc_use",
    # Soil / geology
    "cly_pc_uav", "slt_pc_uav", "snd_pc_uav", "soc_th_uav",
    "kar_pc_use", "ero_kh_uav", "lit_cl_cmj", "lit_cl_vmj",
    # Anthropogenic pressure
    "ppd_pk_uav", "nli_ix_uav", "rdd_mk_uav", "hft_ix_u09",
    "gdp_ud_usu", "hdi_ix_vav",
]
ERA5_PATTERNS = ["era5_"]
MESAN_PATTERNS = ["mesan_"]
SAT_PATTERNS   = ["modis_", "landsat_", "sentinel1_", "sentinel2_", "sentinel3_"]

# Satellite WQ algorithm outputs — Pahlevan/Toming published algorithms used as features
# (avoids reimplementation risk; better scientific argument: "our dataset makes published
#  algorithms strictly more useful when combined with catchment + met covariates")
PUBLISHED_ALG_PATTERNS = [
    "chl_a_ug_l", "chl_a_oc4", "chl_a_ndci",   # Chl-a algorithm outputs
    "turbidity_fnu",                              # Nechad NIR
    "cdom_m", "secchi_m",                         # CDOM + Secchi derived
    "owt_class",                                  # OWT classification
]

# In-situ derived WQ indices present in the feature table but excluded from ALL ML tiers.
# Reason: target leakage — TSI←TP, tsi_chla←Chl-a, chla_tp_ratio←both, etc.
# They are retained as dataset columns for scientific use; see evaluation §4.
DERIVED_WQ_EXCL_PREFIX = "derived_"

ROLLING_ORIGINS = [
    (2000, 2010, 2011, 2015, "2011-15"),
    (2000, 2015, 2016, 2020, "2016-20"),
    (2000, 2020, 2021, 2025, "2021-25"),
]

SENSORS = ["modis", "landsat", "sentinel1", "sentinel2", "sentinel3"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_feature_tables(start_year: int, end_year: int) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for yr in range(start_year, end_year + 1)
              if (p := FEATURE_DIR / f"feature_table_SE_{yr}.parquet").exists()]
    if not frames:
        logger.error("No feature tables found — run build_feature_table.py first")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).copy()
    df["station_id"] = df["station_id"].astype(str)
    df["date"]       = pd.to_datetime(df["date"])
    df["year"]       = df["date"].dt.year
    return df


def _load_atlas() -> pd.DataFrame:
    if not ATLAS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ATLAS_PATH)
    df["station_id"] = df["station_id"].astype(str)
    return df


def _add_trophic_class(ft: pd.DataFrame) -> pd.DataFrame:
    """Classify each row by trophic state from TP (µg/L) — widely used Nordic thresholds."""
    if "tp_ug_l" not in ft.columns:
        ft["trophic_class"] = "unknown"
        return ft
    tp = ft["tp_ug_l"]
    ft["trophic_class"] = pd.cut(
        tp,
        bins=[0, 10, 35, 100, np.inf],
        labels=["oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic"],
    ).astype(str)
    return ft


def _sensor_count(ft: pd.DataFrame) -> pd.Series:
    """Count how many sensors have ≥1 valid observation per row."""
    counts = pd.Series(0, index=ft.index)
    for s in SENSORS:
        col = f"n_sat_obs_{s}"
        if col in ft.columns:
            counts += (ft[col].fillna(0) > 0).astype(int)
    return counts


def _get_cv_groups(ft: pd.DataFrame) -> np.ndarray:
    if SPLIT_PATH.exists():
        try:
            split = pd.read_parquet(SPLIT_PATH, columns=["sample_id", "fold_key"])
            if "sample_id" in ft.columns:
                merged = ft[["sample_id"]].merge(split, on="sample_id", how="left")
                return pd.Categorical(merged["fold_key"].fillna("unknown")).codes
        except Exception:
            pass
    atlas = _load_atlas()
    if not atlas.empty and "latitude" in atlas.columns:
        lat_map = atlas.set_index("station_id")["latitude"].to_dict()
        lats = ft["station_id"].map(lat_map).fillna(60.0)
    else:
        lats = pd.Series(60.0, index=ft.index)
    return lats.apply(lambda v: int(float(v))).values


def _select_features(ft: pd.DataFrame, feature_set: str,
                     ha_cols, era5_cols, mesan_cols,
                     pub_alg_cols, sat_cols) -> list:
    if feature_set == "static":
        cols = list(ha_cols)
    elif feature_set == "met":
        cols = list(ha_cols) + [c for c in era5_cols if c in ft.columns]
    elif feature_set == "met_mesan":
        cols = list(ha_cols) + [c for c in era5_cols if c in ft.columns] + \
               [c for c in mesan_cols if c in ft.columns]
    elif feature_set == "published_alg":
        cols = [c for c in pub_alg_cols if c in ft.columns]
    elif feature_set == "full":
        cols = list(ha_cols) + \
               [c for c in era5_cols if c in ft.columns] + \
               [c for c in mesan_cols if c in ft.columns] + \
               [c for c in sat_cols if c in ft.columns]
    else:
        cols = []
    return [c for c in cols if c in ft.columns]


def _fill(X_df: pd.DataFrame) -> np.ndarray:
    return X_df.apply(
        lambda c: c.fillna(c.median() if c.notna().any() else 0.0)
    ).values.astype(np.float32)


def _metrics(y_true_orig, y_pred_orig, ok=None):
    if ok is None:
        ok = np.isfinite(y_pred_orig) & np.isfinite(y_true_orig)
    a, b = y_true_orig[ok], y_pred_orig[ok]
    if len(a) < 5:
        return {"rmse": np.nan, "r2": np.nan, "mae": np.nan, "n": 0}
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(a, b))), 3),
        "r2":   round(float(r2_score(a, b)), 3),
        "mae":  round(float(mean_absolute_error(a, b)), 3),
        "n":    int(ok.sum()),
    }


# ── Naive baselines ────────────────────────────────────────────────────────────

class StationMeanBaseline:
    """Per-station historical mean — simplest possible baseline (the floor)."""
    def fit(self, X, y, station_ids):
        self._means = dict(zip(station_ids, y))
        self._global = float(np.mean(y))

    def predict(self, X, station_ids):
        return np.array([self._means.get(s, self._global) for s in station_ids],
                        dtype=np.float32)


# ── Model registry ─────────────────────────────────────────────────────────────

def _get_models(requested: list) -> dict:
    models = {}

    if "mean" in requested:
        models["StationMean"] = ("mean", False)

    if "linear" in requested:
        models["Ridge"] = (lambda: Ridge(alpha=1.0), True)

    if "rf" in requested:
        models["RF"] = (lambda: RandomForestRegressor(
            n_estimators=300, max_features="sqrt",
            min_samples_leaf=5, random_state=42, n_jobs=-1), False)

    if "xgb" in requested:
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = (lambda: XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                tree_method="hist", device="cpu", verbosity=0,
                random_state=42), False)
        except ImportError:
            logger.warning("xgboost not installed — skipping")

    if "lgb" in requested:
        try:
            import lightgbm as lgb
            models["LightGBM"] = (lambda: lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, num_leaves=63,
                min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1), False)
        except ImportError:
            logger.warning("lightgbm not installed — skipping")

    if "mlp" in requested:
        try:
            import torch, torch.nn as nn
            from sklearn.base import BaseEstimator, RegressorMixin

            class _TorchMLP(BaseEstimator, RegressorMixin):
                def __init__(self, hidden=(256, 128, 64), lr=1e-3,
                             epochs=200, batch=512, patience=10):
                    self.hidden   = hidden
                    self.lr       = lr
                    self.epochs   = epochs
                    self.batch    = batch
                    self.patience = patience

                def fit(self, X, y):
                    import torch, torch.nn as nn
                    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dims  = [X.shape[1]] + list(self.hidden)
                    layers = []
                    for i in range(len(dims) - 1):
                        layers += [nn.Linear(dims[i], dims[i+1]),
                                   nn.BatchNorm1d(dims[i+1]), nn.ReLU()]
                    layers.append(nn.Linear(dims[-1], 1))
                    self.model_ = nn.Sequential(*layers).to(dev)
                    self.dev_   = dev
                    opt     = torch.optim.AdamW(self.model_.parameters(),
                                                lr=self.lr, weight_decay=1e-4)
                    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(
                                opt, T_max=self.epochs)
                    loss_fn = nn.HuberLoss()
                    Xt = torch.FloatTensor(X).to(dev)
                    yt = torch.FloatTensor(y).unsqueeze(1).to(dev)
                    ds = torch.utils.data.TensorDataset(Xt, yt)
                    dl = torch.utils.data.DataLoader(
                        ds, batch_size=self.batch, shuffle=True, drop_last=True)
                    best, wait = 1e9, 0
                    for _ in range(self.epochs):
                        self.model_.train()
                        for xb, yb in dl:
                            opt.zero_grad()
                            loss_fn(self.model_(xb), yb).backward()
                            opt.step()
                        sch.step()
                        with torch.no_grad():
                            val_loss = loss_fn(self.model_(Xt), yt).item()
                        if val_loss < best - 1e-4:
                            best = val_loss; wait = 0
                        else:
                            wait += 1
                            if wait >= self.patience:
                                break
                    return self

                def predict(self, X):
                    import torch
                    self.model_.eval()
                    with torch.no_grad():
                        return self.model_(
                            torch.FloatTensor(X).to(self.dev_)
                        ).squeeze(1).cpu().numpy()

            models["MLP"] = (lambda: _TorchMLP(), True)
        except ImportError:
            models["MLP"] = (lambda: MLPRegressor(
                hidden_layer_sizes=(256, 128, 64), activation="relu",
                max_iter=500, early_stopping=True, validation_fraction=0.1,
                random_state=42), True)

    if "fttrans" in requested:
        ft_factory = _build_ft_transformer_factory()
        if ft_factory:
            models["FT-Transformer"] = (ft_factory, True)

    if "quantile" in requested:
        try:
            import lightgbm as lgb
            models["QuantileLGB_p10"] = (lambda: lgb.LGBMRegressor(
                objective="quantile", alpha=0.10,
                n_estimators=300, learning_rate=0.05, num_leaves=63,
                random_state=42, n_jobs=-1, verbose=-1), False)
            models["QuantileLGB_p90"] = (lambda: lgb.LGBMRegressor(
                objective="quantile", alpha=0.90,
                n_estimators=300, learning_rate=0.05, num_leaves=63,
                random_state=42, n_jobs=-1, verbose=-1), False)
        except ImportError:
            logger.warning("lightgbm not installed — skipping quantile models")

    return models


def _build_ft_transformer_factory():
    try:
        import torch
        import torch.nn as nn
        from sklearn.base import BaseEstimator, RegressorMixin
    except ImportError:
        return None

    class FTTransformer(BaseEstimator, RegressorMixin):
        def __init__(self, d_token=64, n_heads=8, n_layers=3,
                     ffn_d=128, dropout=0.1, lr=1e-3, epochs=50,
                     batch=512, max_rows=None):
            self.d_token = d_token; self.n_heads = n_heads
            self.n_layers = n_layers; self.ffn_d = ffn_d
            self.dropout = dropout; self.lr = lr
            self.epochs = epochs; self.batch = batch
            self.max_rows = max_rows

        def fit(self, X, y):
            import torch, torch.nn as nn
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.max_rows and len(X) > self.max_rows:
                idx = np.random.choice(len(X), self.max_rows, replace=False)
                X, y = X[idx], y[idx]
            d = self.d_token
            self.embed_ = nn.Linear(1, d).to(dev)
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=self.n_heads, dim_feedforward=self.ffn_d,
                dropout=self.dropout, batch_first=True, norm_first=True)
            self.transformer_ = nn.TransformerEncoder(enc, num_layers=self.n_layers).to(dev)
            self.head_ = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(dev)
            self.cls_  = nn.Parameter(torch.zeros(1, 1, d, device=dev))
            self.dev_  = dev
            params = (list(self.embed_.parameters()) +
                      list(self.transformer_.parameters()) +
                      list(self.head_.parameters()) + [self.cls_])
            opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
            dl  = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)),
                batch_size=self.batch, shuffle=True, drop_last=True)
            loss_fn = nn.HuberLoss()
            best, wait = 1e9, 0
            for _ in range(self.epochs):
                self.embed_.train(); self.transformer_.train(); self.head_.train()
                for xb, yb in dl:
                    xb, yb = xb.to(dev), yb.to(dev)
                    tokens = self.embed_(xb.unsqueeze(-1))
                    cls = self.cls_.expand(xb.size(0), -1, -1)
                    out = self.transformer_(torch.cat([cls, tokens], dim=1))
                    loss = loss_fn(self.head_(out[:, 0]), yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                if loss.item() < best:
                    best = loss.item(); wait = 0
                else:
                    wait += 1
                    if wait >= 5: break
            return self

        def predict(self, X):
            import torch
            dev = self.dev_
            self.embed_.eval(); self.transformer_.eval(); self.head_.eval()
            with torch.no_grad():
                Xt = torch.FloatTensor(X).to(dev)
                tokens = self.embed_(Xt.unsqueeze(-1))
                cls = self.cls_.expand(Xt.size(0), -1, -1)
                out = self.transformer_(torch.cat([cls, tokens], dim=1))
                return self.head_(out[:, 0]).squeeze(1).cpu().numpy()

    return lambda max_rows=None: FTTransformer(max_rows=max_rows)


# ── Cross-validated evaluation ─────────────────────────────────────────────────

def _conformal_cv(model_spec, X, y, groups, n_splits,
                  levels=(0.80, 0.90, 0.95), cal_frac=0.2):
    """
    Split conformal prediction within each GroupKFold fold.

    Per fold: split training → fit set (1-cal_frac) + calibration set (cal_frac).
    Nonconformity score: |y - ŷ| in log-space.
    Conformal quantile: finite-sample corrected ceiling(( n_cal+1)(1-α)/n_cal).
    Provides marginal coverage ≥ 1-α (Vovk et al. 2005; Papadopoulos et al. 2002).

    Returns list of dicts: {nominal_coverage, picp, ece, mean_interval_width_log, n_folds}
    """
    mname, factory, needs_scale = model_spec
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    fold_agg = {lv: {"picp": [], "width": []} for lv in levels}

    for tr, va in cv.split(X, y, groups):
        n_cal   = max(10, int(len(tr) * cal_frac))
        fit_idx = tr[:-n_cal]
        cal_idx = tr[-n_cal:]

        Xfit, Xcal, Xva = X[fit_idx], X[cal_idx], X[va]
        yfit, ycal, yva = y[fit_idx], y[cal_idx], y[va]

        if needs_scale:
            sc   = StandardScaler()
            Xfit = sc.fit_transform(Xfit)
            Xcal = sc.transform(Xcal)
            Xva  = sc.transform(Xva)

        try:
            m = factory()
            m.fit(Xfit, yfit)
            scores   = np.abs(ycal   - m.predict(Xcal))
            val_pred = m.predict(Xva)
        except Exception:
            continue

        n = len(scores)
        for lv in levels:
            alpha = 1 - lv
            q = float(np.quantile(scores,
                                  min(1.0, np.ceil((n + 1) * (1 - alpha)) / n),
                                  method="higher"))
            covered = (yva >= val_pred - q) & (yva <= val_pred + q)
            fold_agg[lv]["picp"].append(float(covered.mean()))
            fold_agg[lv]["width"].append(2.0 * q)

    rows = []
    for lv in levels:
        picps = fold_agg[lv]["picp"]
        widths = fold_agg[lv]["width"]
        if not picps:
            continue
        picp = float(np.mean(picps))
        rows.append({
            "nominal_coverage":         lv,
            "picp":                     round(picp, 3),
            "ece":                      round(abs(picp - lv), 4),
            "mean_interval_width_log":  round(float(np.mean(widths)), 4),
            "n_folds":                  len(picps),
        })
    return rows


def _cv_predict(model_spec, X, y, groups, station_ids, n_splits, scaler=None):
    """Run GroupKFold CV and return prediction array (log-space)."""
    mname, factory, needs_scale = model_spec
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    preds = np.full(len(y), np.nan)

    for tr, va in cv.split(X, y, groups):
        if mname == "StationMean":
            m = StationMeanBaseline()
            m.fit(X[tr], y[tr], station_ids[tr])
            preds[va] = m.predict(X[va], station_ids[va])
        else:
            m = factory()
            Xtr, Xva = X[tr], X[va]
            if needs_scale:
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr)
                Xva = sc.transform(Xva)
            try:
                m.fit(Xtr, y[tr])
                preds[va] = m.predict(Xva)
            except Exception as exc:
                logger.warning(f"  {mname} fold failed: {exc}")

    return preds


# ── Rolling-origin temporal evaluation ─────────────────────────────────────────

def _rolling_origin_eval(ft, X_full, y_full, groups, station_ids,
                          feat_cols, model_specs, target, n_splits):
    """
    Rolling-origin temporal evaluation:
    train 2000-2010 → test 2011-2015
    train 2000-2015 → test 2016-2020
    train 2000-2020 → test 2021-2025
    """
    rows = []
    for train_start, train_end, test_start, test_end, label in ROLLING_ORIGINS:
        tr_mask = ((ft["year"] >= train_start) & (ft["year"] <= train_end)).values
        va_mask = ((ft["year"] >= test_start) & (ft["year"] <= test_end)).values
        if tr_mask.sum() < 50 or va_mask.sum() < 10:
            continue
        Xtr, Xva = X_full[tr_mask], X_full[va_mask]
        ytr, yva = y_full[tr_mask], y_full[va_mask]
        sids_va  = station_ids[va_mask]

        for mname, (factory, needs_scale) in model_specs.items():
            if mname in ("QuantileLGB_p10", "QuantileLGB_p90"):
                continue
            try:
                if mname == "StationMean":
                    m = StationMeanBaseline()
                    m.fit(Xtr, ytr, station_ids[tr_mask])
                    pred = m.predict(Xva, sids_va)
                else:
                    m = factory()
                    Xtr_, Xva_ = Xtr, Xva
                    if needs_scale:
                        sc = StandardScaler()
                        Xtr_ = sc.fit_transform(Xtr_)
                        Xva_ = sc.transform(Xva_)
                    m.fit(Xtr_, ytr)
                    pred = m.predict(Xva_)
                m_res = _metrics(np.expm1(yva), np.expm1(pred))
                rows.append({"target": target, "model": mname,
                             "test_period": label, **m_res})
                logger.info(f"  rolling {label} {mname}: RMSE={m_res['rmse']:.3f}")
            except Exception as exc:
                logger.warning(f"  rolling {label} {mname}: {exc}")
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def run(start_year=2000, end_year=2025,
        requested_models=None, max_rows=None,
        requested_targets=None, force=False):
    if requested_models is None:
        requested_models = ["mean", "linear", "rf", "xgb", "lgb", "mlp",
                            "fttrans", "quantile"]
    active_targets = requested_targets if requested_targets else TARGETS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading feature tables {start_year}–{end_year}")
    ft = _load_feature_tables(start_year, end_year)
    if ft.empty:
        return

    atlas = _load_atlas()
    if not atlas.empty:
        ha_cols = [c for c in HYDROATLAS_FEATURES if c in atlas.columns]
        # Only merge columns not already in ft — feature table may have joined atlas upstream
        ha_to_add = [c for c in ha_cols if c not in ft.columns]
        if ha_to_add:
            ft = ft.merge(
                atlas[["station_id"] + ha_to_add].drop_duplicates("station_id"),
                on="station_id", how="left")
    else:
        ha_cols = []
        logger.warning("HydroATLAS not found — static features absent")

    ft = ft.copy()  # defragment after multi-year concat + merges

    era5_cols   = [c for c in ft.columns if any(c.startswith(p) for p in ERA5_PATTERNS)]
    mesan_cols  = [c for c in ft.columns
                   if c.startswith("mesan_") and c not in ("mesan_product",)]
    sat_cols    = [c for c in ft.columns
                   if any(c.startswith(p) for p in SAT_PATTERNS)
                   and not any(c.endswith(s) for s in ("_offset", "_obs", "_product"))
                   and ft[c].dtype.kind in ("f", "i")]
    pub_alg_cols = [c for c in ft.columns
                    if any(p in c for p in PUBLISHED_ALG_PATTERNS)
                    and any(c.startswith(s + "_") for s in SENSORS)
                    and ft[c].dtype.kind in ("f", "i")]

    logger.info(f"  {len(ft):,} rows · HA:{len(ha_cols)} ERA5:{len(era5_cols)} "
                f"MESAN:{len(mesan_cols)} PubAlg:{len(pub_alg_cols)} Sat:{len(sat_cols)}")

    ft = _add_trophic_class(ft)
    ft["n_sensors"] = _sensor_count(ft)
    ft["sample_id"] = ft.index.astype(int)
    groups       = _get_cv_groups(ft)
    station_ids  = ft["station_id"].values
    n_splits     = min(5, len(np.unique(groups)))
    model_specs  = _get_models(requested_models)

    feature_sets = ["static", "met", "met_mesan", "published_alg", "full"]
    _CONFORMAL_MODELS = {"Ridge", "RF", "XGBoost", "LightGBM", "MLP", "FT-Transformer"}

    # ── Resume: pre-load existing results and skip completed targets ──
    def _load_csv(name, dedup_keys=None):
        p = OUT_DIR / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                if dedup_keys and not df.empty:
                    df = df.drop_duplicates(subset=dedup_keys, keep="first")
                return df.to_dict("records")
            except Exception:
                return []
        return []

    results        = _load_csv("baseline_model_results.csv",
                               dedup_keys=["target", "model", "feature_set"])
    trophic_rows   = _load_csv("baseline_trophic_results.csv")
    rolling_rows   = _load_csv("baseline_rolling_origin.csv")
    coverage_rows  = _load_csv("baseline_sensor_coverage.csv")
    quantile_rows  = _load_csv("baseline_quantile_coverage.csv")
    conformal_rows = _load_csv("baseline_conformal_results.csv")
    ac_failure_rows= _load_csv("baseline_ac_failure_results.csv")
    importance_rows= []  # parquet — always recompute (fast)

    completed_targets = {r["target"] for r in results} if results else set()
    # When --conformal-only: skip targets only if conformal already covers all
    # requested models for that target — otherwise rerun the CV loop.
    conformal_done = {}  # target → set of model names already in conformal CSV
    for r in conformal_rows:
        conformal_done.setdefault(r["target"], set()).add(r["model"])

    if completed_targets:
        logger.info(f"Resuming — skipping already-complete targets: {completed_targets}")

    for target in active_targets:
        if target not in ft.columns:
            logger.warning(f"Target {target} not in feature table — skipping")
            continue
        # Determine whether to rerun the main CV loop
        conf_models_needed = {m for m in _CONFORMAL_MODELS if m in model_specs}
        conf_models_done   = conformal_done.get(target, set())
        conformal_complete = conf_models_needed.issubset(conf_models_done)

        if force:
            # --force: always rerun everything
            rerun_main = True
        elif requested_targets:
            # --targets without --force: skip main CV if already done, only run conformal
            rerun_main = target not in completed_targets
            if not rerun_main and conformal_complete:
                logger.info(f"Skipping {target} — results and conformal already complete")
                continue
            if not rerun_main:
                logger.info(f"Conformal-only run for {target} (main CV already done)")
        else:
            # Normal resume logic
            if target in completed_targets and conformal_complete:
                logger.info(f"Skipping {target} — results and conformal already in CSV")
                continue
            if target in completed_targets:
                logger.info(f"Partial resume {target} — running only missing conformal models")
            rerun_main = target not in completed_targets
        valid = ft[ft[target].notna() & (ft[target] > 0)].copy()
        y_log = np.log1p(valid[target].values.astype(np.float32))
        sids  = valid["station_id"].values
        grp   = groups[valid.index]
        logger.info(f"\n── {target}  ({len(valid):,} samples) ──")

        # ── Main spatial CV across feature tiers ──
        for fs in (feature_sets if rerun_main else []):
            feat_cols = _select_features(
                valid, fs, ha_cols, era5_cols, mesan_cols, pub_alg_cols, sat_cols)
            if not feat_cols:
                logger.info(f"  {fs}: no features available — skipping")
                continue
            X = _fill(valid[feat_cols])

            for mname, (factory, needs_scale) in model_specs.items():
                if mname in ("QuantileLGB_p10", "QuantileLGB_p90"):
                    continue
                preds = _cv_predict(
                    (mname, factory, needs_scale), X, y_log, grp, sids, n_splits)
                ok = ~np.isnan(preds)
                rec = {"target": target, "model": mname, "feature_set": fs,
                       **_metrics(np.expm1(y_log), np.expm1(preds), ok)}
                results.append(rec)
                logger.info(f"  {mname:16s} {fs:14s}: "
                            f"RMSE={rec['rmse']:.3f}  R²={rec['r2']:.3f}  n={rec['n']:,}")

                # Feature importance (tree models, full tier only)
                if mname in ("RF", "XGBoost", "LightGBM") and fs == "full":
                    m_full = factory()
                    m_full.fit(X, y_log)
                    imp = pd.Series(
                        getattr(m_full, "feature_importances_", []),
                        index=feat_cols)
                    for feat, val in imp.nlargest(20).items():
                        importance_rows.append({"target": target, "model": mname,
                                                "feature": feat,
                                                "importance": round(float(val), 5)})

                # Trophic class breakdown (full tier only)
                if fs == "full" and mname == "LightGBM":
                    for tc, tc_idx in valid.groupby("trophic_class").groups.items():
                        mask = np.zeros(len(valid), dtype=bool)
                        mask[np.where(valid.index.isin(tc_idx))[0]] = True
                        ok_tc = ok & mask
                        if ok_tc.sum() < 10:
                            continue
                        m_tc = _metrics(np.expm1(y_log), np.expm1(preds), ok_tc)
                        trophic_rows.append({"target": target, "trophic_class": tc,
                                             "model": mname, "feature_set": fs, **m_tc})

                # AC failure breakdown (LGB on published_alg and full)
                # Compares: satellite-only degrades under AC failure;
                # full (catchment+met) maintains performance — the narrative arc.
                if mname == "LightGBM" and fs in ("published_alg", "full"):
                    flag_cols = [c for c in valid.columns if "rw_blue_negative_flag" in c]
                    if flag_cols:
                        any_fail = (valid[flag_cols].fillna(0).max(axis=1) > 0).values
                        for flag_val, flag_label in [(False, "ac_ok"), (True, "ac_failure")]:
                            mask_ac = (any_fail == flag_val) & ok
                            if mask_ac.sum() < 10:
                                continue
                            m_ac = _metrics(np.expm1(y_log), np.expm1(preds), mask_ac)
                            ac_failure_rows.append({
                                "target": target, "model": mname,
                                "feature_set": fs, "ac_flag": flag_label, **m_ac,
                            })

        # ── Sensor-coverage ablation (LightGBM full) ──
        # Two views:
        #   (A) by total sensor count (0-5) — marginal value of each additional sensor
        #   (B) per-sensor presence — does having sensor X improve predictions?
        if rerun_main and "lgb" in requested_models:
            feat_cols = _select_features(
                valid, "full", ha_cols, era5_cols, mesan_cols, pub_alg_cols, sat_cols)
            lgb_spec = model_specs.get("LightGBM")
            if feat_cols and lgb_spec:
                factory, _ = lgb_spec
                X = _fill(valid[feat_cols])
                preds = _cv_predict(
                    ("LightGBM", factory, False), X, y_log, grp, sids, n_splits)

                # (A) total count groups
                for n_s in range(6):
                    mask = (valid["n_sensors"].values == n_s) & ~np.isnan(preds)
                    if mask.sum() < 10:
                        continue
                    m_s = _metrics(np.expm1(y_log), np.expm1(preds), mask)
                    coverage_rows.append({
                        "target": target, "grouping": "total_count",
                        "sensor": "all", "present": n_s, **m_s,
                    })
                    logger.info(f"  sensor_cov total={n_s}: RMSE={m_s['rmse']:.3f} n={m_s['n']:,}")

                # (B) per-sensor presence: rows WITH vs WITHOUT each sensor
                for s in SENSORS:
                    col = f"n_sat_obs_{s}"
                    if col not in valid.columns:
                        continue
                    has_s  = (valid[col].fillna(0) > 0).values & ~np.isnan(preds)
                    lacks_s = ~(valid[col].fillna(0) > 0).values & ~np.isnan(preds)
                    for present, mask in [(1, has_s), (0, lacks_s)]:
                        if mask.sum() < 10:
                            continue
                        m_s = _metrics(np.expm1(y_log), np.expm1(preds), mask)
                        coverage_rows.append({
                            "target": target, "grouping": "per_sensor",
                            "sensor": s, "present": present, **m_s,
                        })
                logger.info(f"  per-sensor presence rows added for {target}")

        # ── Quantile UQ calibration ──
        # Metrics stored per (target, trophic_class, nominal_level):
        #   PICP    — Prediction Interval Coverage Probability (empirical vs nominal)
        #   PINAW   — Normalized Average Width = mean(hi-lo) / (y_max - y_min)
        #             normalises by target range → cross-target comparison valid
        #   Winkler — proper interval scoring rule: width + 2/alpha * max(lo-y, y-hi, 0)
        #             penalises miscoverage AND unnecessary width simultaneously
        #   CRPS    — Continuous Ranked Probability Score (quantile approximation)
        #             gold-standard proper scoring rule at ML venues
        #   ECE     — Expected Calibration Error = mean|empirical - nominal| over levels
        #             single scalar for tables; analogous to classification ECE
        #   Sharpness — mean interval width (raw); measures unconditional tightness
        if rerun_main and "quantile" in requested_models:
            feat_cols = _select_features(
                valid, "full", ha_cols, era5_cols, mesan_cols, pub_alg_cols, sat_cols)
            if feat_cols and "QuantileLGB_p10" in model_specs:
                X = _fill(valid[feat_cols])
                try:
                    import lightgbm as lgb

                    # Fit quantile predictions at 10 alpha levels (needed for CRPS + all metrics)
                    alphas = [0.025, 0.05, 0.10, 0.15, 0.25,
                              0.75,  0.85, 0.90, 0.95, 0.975]
                    q_preds = {}
                    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(grp))))
                    for alpha in alphas:
                        p = np.full(len(y_log), np.nan)
                        for tr, va in cv.split(X, y_log, grp):
                            m = lgb.LGBMRegressor(
                                objective="quantile", alpha=alpha,
                                n_estimators=300, learning_rate=0.05,
                                num_leaves=63, random_state=42,
                                n_jobs=-1, verbose=-1)
                            m.fit(X[tr], y_log[tr])
                            p[va] = m.predict(X[va])
                        q_preds[alpha] = np.expm1(p)   # back to original scale

                    y_orig       = np.expm1(y_log)
                    target_range = float(np.nanpercentile(y_orig, 97.5) -
                                         np.nanpercentile(y_orig, 2.5))
                    ok_q         = ~np.isnan(q_preds[0.10]) & ~np.isnan(q_preds[0.90])
                    trophic_vals = valid["trophic_class"].values

                    # Interval bounds: nominal → (lo_alpha, hi_alpha)
                    bounds = {
                        0.50: (0.25,  0.75),
                        0.70: (0.15,  0.85),
                        0.80: (0.10,  0.90),
                        0.90: (0.05,  0.95),
                        0.95: (0.025, 0.975),
                    }

                    def _crps_from_quantiles(y_true, q_preds_dict, alphas_sorted):
                        """Quantile-score approximation of CRPS (Gneiting & Raftery 2007)."""
                        total = 0.0
                        for a in alphas_sorted:
                            q = q_preds_dict[a]
                            total += (2 * a - 1) * (y_true - q) + \
                                     2 * np.where(y_true < q,
                                                  (1 - a) * (q - y_true),
                                                  a * (y_true - q))
                        return float(np.mean(total) / len(alphas_sorted))

                    def _winkler(y_true, lo, hi, alpha):
                        """Winkler interval score for nominal level (1-alpha)."""
                        width = hi - lo
                        penalty = (2 / alpha) * np.maximum(lo - y_true, 0) + \
                                  (2 / alpha) * np.maximum(y_true - hi, 0)
                        return float(np.mean(width + penalty))

                    for tc_label in (["all"] + list(valid["trophic_class"].unique())):
                        tc_mask = ok_q if tc_label == "all" else \
                                  ok_q & (trophic_vals == tc_label)
                        if tc_mask.sum() < 20:
                            continue
                        y_tc = y_orig[tc_mask]
                        tr_range = float(np.nanpercentile(y_tc, 97.5) -
                                         np.nanpercentile(y_tc, 2.5)) or target_range

                        # CRPS (all alpha levels, full mask)
                        q_tc = {a: q_preds[a][tc_mask] for a in alphas}
                        crps_val = _crps_from_quantiles(y_tc, q_tc,
                                                         sorted(alphas))

                        # Per nominal level: PICP, PINAW, Winkler, Sharpness
                        picp_vals = []
                        for nominal, (lo_a, hi_a) in bounds.items():
                            lo = q_preds[lo_a][tc_mask]
                            hi = q_preds[hi_a][tc_mask]
                            alpha_interval = 1.0 - nominal

                            picp      = float(np.mean((y_tc >= lo) & (y_tc <= hi)))
                            pinaw     = float(np.mean(hi - lo)) / tr_range
                            winkler   = _winkler(y_tc, lo, hi, alpha_interval)
                            sharpness = float(np.mean(hi - lo))
                            picp_vals.append((nominal, picp))

                            quantile_rows.append({
                                "target":              target,
                                "trophic_class":       tc_label,
                                "nominal_coverage":    nominal,
                                "picp":                round(picp,      3),
                                "pinaw":               round(pinaw,     4),
                                "winkler_score":       round(winkler,   3),
                                "sharpness":           round(sharpness, 3),
                                "crps":                round(crps_val,  3),
                                "n":                   int(tc_mask.sum()),
                            })

                        # ECE: mean absolute deviation of empirical from nominal
                        ece = float(np.mean([abs(emp - nom)
                                             for nom, emp in picp_vals]))
                        # Tag the last row for this (target, tc) with ECE
                        for row in reversed(quantile_rows):
                            if row["target"] == target and row["trophic_class"] == tc_label:
                                row["ece"] = round(ece, 4)
                                break

                        if tc_label == "all":
                            logger.info(
                                f"  UQ {target}: CRPS={crps_val:.3f}  "
                                f"ECE={ece:.4f}  "
                                f"80%PICP={dict(picp_vals).get(0.80, float('nan')):.3f}")
                except Exception as exc:
                    logger.warning(f"  UQ {target}: {exc}")

        # ── Split conformal prediction (Ridge/RF/XGB/LGB on full feature set) ──
        feat_cols_full = _select_features(
            valid, "full", ha_cols, era5_cols, mesan_cols, pub_alg_cols, sat_cols)
        if feat_cols_full:
            X_conf = _fill(valid[feat_cols_full])
            for mname, (factory, needs_scale) in model_specs.items():
                if mname not in _CONFORMAL_MODELS:
                    continue
                try:
                    conf = _conformal_cv(
                        (mname, factory, needs_scale),
                        X_conf, y_log, grp, n_splits)
                    for r in conf:
                        conformal_rows.append({
                            "target": target, "model": mname,
                            "feature_set": "full", **r})
                    logger.info(
                        f"  conformal {mname}: " +
                        "  ".join(f"{r['nominal_coverage']:.0%}→"
                                  f"PICP={r['picp']:.3f} ECE={r['ece']:.4f}"
                                  for r in conf))
                except Exception as exc:
                    logger.warning(f"  conformal {mname} failed: {exc}")

        # ── Rolling-origin temporal evaluation (LightGBM full) ──
        if rerun_main and "lgb" in requested_models:
            feat_cols = _select_features(
                ft, "full", ha_cols, era5_cols, mesan_cols, pub_alg_cols, sat_cols)
            if feat_cols:
                valid_all = ft[ft[target].notna() & (ft[target] > 0)].copy()
                y_all = np.log1p(valid_all[target].values.astype(np.float32))
                X_all = _fill(valid_all[feat_cols])
                sids_all = valid_all["station_id"].values
                row_specs = {k: v for k, v in model_specs.items()
                             if k in ("StationMean", "Ridge", "LightGBM")}
                rolling_rows += _rolling_origin_eval(
                    valid_all, X_all, y_all, groups[valid_all.index],
                    sids_all, feat_cols, row_specs, target, n_splits)

        # ── Incremental save after each target ──
        _MAIN_DEDUP = ["target", "model", "feature_set"]

        def _save_incremental(rows, name, dedup_keys=None):
            if rows:
                df_out = pd.DataFrame(rows)
                if dedup_keys:
                    df_out = df_out.drop_duplicates(subset=dedup_keys, keep="last")
                df_out.to_csv(OUT_DIR / name, index=False)

        _save_incremental(results,        "baseline_model_results.csv",
                          dedup_keys=_MAIN_DEDUP)
        _save_incremental(trophic_rows,   "baseline_trophic_results.csv")
        _save_incremental(rolling_rows,   "baseline_rolling_origin.csv")
        _save_incremental(coverage_rows,  "baseline_sensor_coverage.csv")
        _save_incremental(quantile_rows,  "baseline_quantile_coverage.csv")
        _save_incremental(conformal_rows, "baseline_conformal_results.csv")
        _save_incremental(ac_failure_rows,"baseline_ac_failure_results.csv")
        if importance_rows:
            pd.DataFrame(importance_rows).to_parquet(
                OUT_DIR / "baseline_feature_importance.parquet", index=False)
        logger.info(f"  [checkpoint] results saved after {target}")

    # ── Final save ──
    def _save(df, name):
        if df:
            path = OUT_DIR / name
            pd.DataFrame(df).to_csv(path, index=False)
            logger.info(f"Saved → {path}")

    results_df = (pd.DataFrame(results)
                  .drop_duplicates(subset=["target", "model", "feature_set"],
                                   keep="last"))
    results_df.to_csv(OUT_DIR / "baseline_model_results.csv", index=False)
    logger.info(f"\nSaved → {OUT_DIR / 'baseline_model_results.csv'}")

    try:
        pivot = results_df.pivot_table(
            index=["model", "feature_set"], columns="target",
            values=["rmse", "r2"], aggfunc="first")
        pivot.to_latex(str(OUT_DIR / "baseline_model_results.tex"),
                       float_format="%.3f", escape=False)
        logger.info(f"Saved → {OUT_DIR / 'baseline_model_results.tex'}")
    except Exception as e:
        logger.warning(f"LaTeX export: {e}")

    _save(trophic_rows,    "baseline_trophic_results.csv")
    _save(rolling_rows,    "baseline_rolling_origin.csv")
    _save(coverage_rows,   "baseline_sensor_coverage.csv")
    _save(quantile_rows,   "baseline_quantile_coverage.csv")
    _save(conformal_rows,  "baseline_conformal_results.csv")
    _save(ac_failure_rows, "baseline_ac_failure_results.csv")

    if importance_rows:
        pd.DataFrame(importance_rows).to_parquet(
            OUT_DIR / "baseline_feature_importance.parquet", index=False)
        logger.info(f"Saved → {OUT_DIR / 'baseline_feature_importance.parquet'}")

    logger.info("\n" + results_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Baseline models for NordWQ-26 (CIKM 2026)")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year",   type=int, default=2025)
    parser.add_argument("--models",  nargs="+",
                        default=["mean", "linear", "rf", "xgb", "lgb", "mlp",
                                 "fttrans", "quantile"],
                        choices=["mean", "linear", "rf", "xgb", "lgb",
                                 "mlp", "fttrans", "quantile"])
    parser.add_argument("--targets", nargs="+",
                        choices=["chla_ug_l", "tp_ug_l", "secchi_m"],
                        default=None,
                        help="Run only specific targets (bypasses resume skip logic)")
    parser.add_argument("--force", action="store_true",
                        help="Force rerun even for targets already in results CSV")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Subsample for FT-Transformer (e.g. 100000). "
                             "Applies only to FT-Trans; tree/linear models use full data.")
    args = parser.parse_args()
    run(args.start_year, args.end_year, args.models, args.max_rows,
        requested_targets=args.targets, force=args.force)


if __name__ == "__main__":
    main()
