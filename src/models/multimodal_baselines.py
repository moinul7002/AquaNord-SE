#!/usr/bin/env python3
"""
Multimodal baselines for NordWQ-26 (CIKM 2026).

Four models evaluating true multimodal fusion for Nordic lake WQ prediction:

  1. ERA5-LSTM       — LSTM on 30-day ERA5 time-series + HydroATLAS static features.
                       "Does antecedent meteorological forcing matter for WQ prediction?"
                       Temporal analogue of Kratzert et al. (HESS 2018/2019).

  2. TemporalTrans   — Temporal Transformer following Presto's encoder architecture
                       (Tseng et al., Remote Sensing 2023), trained from scratch.
                       Input: 12-month ERA5 aggregations (tp, t_min, t_max, t_mean).
                       CLS token + learnable positional encoding + 4-layer Transformer.
                       Pretrained Presto weights unavailable: presto 0.0.1 pins
                       numpy==1.23.5 / torch==2.0 (incompatible with Python 3.12 +
                       torch 2.13), and default_model.pt is absent from the git install.
                       Invoked via --models presto.
                       "Does a temporal Transformer on monthly ERA5 outperform LSTM?"

  3. CrossModalNet   — CNN patch spatial tokens + tabular token + cross-attention fusion.
                       Standard attention-based image+tabular fusion (stronger ablation
                       than the late-fusion FusionNet in patch_baseline.py).
                       "Is cross-modal attention better than late-fusion concatenation?"

  4. MOSAIKS         — Random Fourier features from ERA5 patch + tabular → Ridge.
                       Rolf et al. (PNAS 2021): reproducible, citable, cheap baseline.

Note on Prithvi (IBM/NASA) and SatMAE (Cong et al., NeurIPS 2022):
  Both require optical HLS / Sentinel-2 spatial patches not yet in the dataset.
  Include in journal version after GEE satellite patch extraction.

Targets (log1p-transformed): chla_ug_l, tp_ug_l, secchi_m
CV: GroupKFold on fold_key (SMHI basin / HydroLAKES lake ID), matching baseline_models.py

Outputs:
  data/outputs/multimodal_baseline_results.csv
  data/outputs/multimodal_baseline_results.tex
  data/outputs/multimodal_trophic_results.csv        — R²/RMSE by trophic class
  data/outputs/multimodal_rolling_origin.csv         — rolling-origin temporal CV
  data/outputs/multimodal_sensor_coverage.csv        — RMSE vs sensor availability

Usage:
    python -m src.evaluation.multimodal_baselines
    python -m src.evaluation.multimodal_baselines --models lstm presto crossmodal mosaiks
    python -m src.evaluation.multimodal_baselines --window 90   # 90-day ERA5 lookback
    python -m src.evaluation.multimodal_baselines --models lstm --window 30
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

MULTIMODAL_DIR = Path("data/processed/nordic_multimodal_dataset")
ERA5_RAW_DIR   = Path("data/raw/era5")
ATLAS_PATH     = Path("data/raw/hydroatlas/hydroatlas_station_covariates_SE.parquet")
OUT_DIR        = Path("data/outputs")

TARGETS = ["chla_ug_l", "tp_ug_l", "secchi_m"]

# Feature columns in raw ERA5 daily parquets
ERA5_SEQ_COLS = ["t2m_mean_c", "t2m_min_c", "t2m_max_c", "tp_mm", "sde_m", "lmlt_c", "tcc_frac"]
N_ERA5 = len(ERA5_SEQ_COLS)  # 7

# ERA5 channels relevant to Presto (monthly aggregations)
# Maps our column names → Presto semantic role
PRESTO_ERA5_MAP = {
    "tp_mm":       "precipitation",     # monthly sum
    "t2m_min_c":   "temp_min",          # monthly min
    "t2m_max_c":   "temp_max",          # monthly max
    "t2m_mean_c":  "temp_mean",         # monthly mean
}
PRESTO_ERA5_COLS = list(PRESTO_ERA5_MAP.keys())  # 4 channels

HYDROATLAS_FEATURES = [
    "for_pc_use", "crp_pc_use", "pst_pc_use", "urb_pc_use", "wet_pc_ug1",
    "pre_mm_uyr", "ele_mt_uav", "soc_th_uav", "dis_m3_pyr", "tmp_dc_uyr",
    "ari_ix_uav", "snw_pc_uyr", "slp_dg_uav", "cly_pc_uav",
    "Depth_avg", "Lake_area", "Res_time",
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _get_device() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        t = torch.zeros(1, device="cuda")
        _ = (t + 1).item()
        return "cuda"
    except Exception:
        return "cpu"


def _conformal_intervals(y_cal: np.ndarray, pred_cal: np.ndarray,
                         pred_val: np.ndarray,
                         levels=(0.80, 0.90, 0.95)) -> list:
    """
    Post-hoc split conformal prediction.
    Given nonconformity scores from a calibration set and point predictions on a
    validation set, return conformal intervals at each nominal level.
    Finite-sample coverage guarantee: PICP ≥ 1-α (Vovk et al. 2005).
    """
    scores = np.abs(y_cal - pred_cal)
    n = len(scores)
    rows = []
    for lv in levels:
        alpha = 1 - lv
        q = float(np.quantile(scores,
                              min(1.0, np.ceil((n + 1) * (1 - alpha)) / n),
                              method="higher"))
        rows.append({"nominal_coverage": lv, "conformal_halfwidth_log": round(q, 4)})
    return rows


def _conformal_coverage(y_true: np.ndarray, pred: np.ndarray,
                        halfwidths: list) -> list:
    """Compute PICP and ECE given predictions + per-level halfwidths."""
    rows = []
    for entry in halfwidths:
        lv = entry["nominal_coverage"]
        q  = entry["conformal_halfwidth_log"]
        picp = float(np.mean((y_true >= pred - q) & (y_true <= pred + q)))
        rows.append({
            "nominal_coverage": lv,
            "conformal_halfwidth_log": entry["conformal_halfwidth_log"],
            "picp": round(picp, 3),
            "ece":  round(abs(picp - lv), 4),
        })
    return rows


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ok = ~np.isnan(y_pred)
    if ok.sum() < 5:
        return {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan"), "n": 0}
    y0 = np.expm1(y_true[ok])
    p0 = np.expm1(y_pred[ok])
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y0, p0))), 3),
        "r2":   round(float(r2_score(y0, p0)), 3),
        "mae":  round(float(mean_absolute_error(y0, p0)), 3),
        "n":    int(ok.sum()),
    }


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_metadata() -> pd.DataFrame:
    p = MULTIMODAL_DIR / "metadata.parquet"
    if not p.exists():
        logger.error(f"metadata.parquet not found — run build_multimodal_dataset.py first")
        return pd.DataFrame()
    meta = pd.read_parquet(p)
    meta["station_id"] = meta["station_id"].astype(str)
    meta["date"]       = pd.to_datetime(meta["date"])
    return meta


def _load_patches() -> np.ndarray:
    import zarr
    p = MULTIMODAL_DIR / "patches.zarr"
    if not p.exists():
        logger.error(f"patches.zarr not found")
        return np.array([])
    z = zarr.open(str(p), mode="r")
    patches = z[:].astype(np.float32)  # (n_samples, 12, 12, 9) channels-last
    patches = np.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-channel check: sro (ch7) and ro (ch8) were not normalized when the zarr was
    # written (raw ERA5 values instead of (raw-offset)/scale).  Any channel with
    # max > 10 is flagged as un-normalized and is clipped to [p1, p99] then scaled
    # to [-1, 1].  Already-normalized channels (max ≤ 10) and zero channels are skipped.
    for ci in range(patches.shape[-1]):
        ch = patches[:, :, :, ci]
        ch_max = float(ch.max())
        if ch_max <= 10.0:
            continue
        p1  = float(np.percentile(ch, 1))
        p99 = float(np.percentile(ch, 99))
        rng = p99 - p1
        if rng < 1e-8:
            patches[:, :, :, ci] = 0.0
        else:
            patches[:, :, :, ci] = np.clip((ch - p1) / rng, 0.0, 1.0) * 2.0 - 1.0

    logger.info("Patches loaded and channel-normalized: shape=%s", patches.shape)
    return patches.transpose(0, 3, 1, 2).astype(np.float32)  # (n, 9, 12, 12)


def _load_era5_station_ts() -> dict:
    """
    Load all yearly ERA5 parquets and return a per-station dict of sorted daily DataFrames.
    Only loads stations present in at least one ERA5 file.
    Returns: {station_id_str: DataFrame(index=date, columns=ERA5_SEQ_COLS)}
    """
    files = sorted(ERA5_RAW_DIR.glob("era5_sweden_*.parquet"))
    if not files:
        logger.warning("No ERA5 parquet files found in data/raw/era5/ — LSTM/Presto disabled")
        return {}

    logger.info(f"Loading {len(files)} ERA5 yearly parquets...")
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["station_id", "date"] + ERA5_SEQ_COLS)
        df["station_id"] = df["station_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)

    era5 = pd.concat(frames, ignore_index=True)
    era5 = era5.sort_values(["station_id", "date"])

    station_ts = {}
    for sid, grp in era5.groupby("station_id", sort=False):
        ts = grp.set_index("date")[ERA5_SEQ_COLS].astype("float32")
        # Fill internal gaps with forward-fill (max 7 days); leave leading NaN as 0
        ts = ts.ffill(limit=7).fillna(0.0)
        station_ts[str(sid)] = ts

    logger.info(f"  ERA5 station time-series loaded: {len(station_ts):,} stations")
    return station_ts


def _build_era5_sequences(
    meta: pd.DataFrame,
    station_ts: dict,
    window: int = 30,
    force: bool = False,
) -> tuple:
    """
    Build fixed-length daily ERA5 sequences for every sample.

    Returns:
      sequences : (n_samples, window, n_features) float32
                  Right-aligned: sequences[:, -1, :] = day before observation.
                  Zero for samples whose station has no ERA5 data.
      era5_mask : (n_samples,) bool — True where station has ERA5 time-series data.
    """
    cache_path = MULTIMODAL_DIR / f"era5_sequences_w{window}.npy"
    mask_path  = MULTIMODAL_DIR / f"era5_seq_mask_w{window}.npy"

    if cache_path.exists() and mask_path.exists() and not force:
        logger.info(f"Loading ERA5 sequences from cache (window={window})")
        return np.load(str(cache_path)), np.load(str(mask_path))

    n = len(meta)
    n_feat = len(ERA5_SEQ_COLS)
    sequences = np.zeros((n, window, n_feat), dtype=np.float32)
    era5_mask  = np.zeros(n, dtype=bool)

    logger.info(f"Building ERA5 {window}-day sequences for {n:,} samples...")
    n_with_ts = 0

    for i, row in enumerate(meta.itertuples(index=False)):
        sid  = str(row.station_id)
        date = pd.Timestamp(row.date)

        if sid not in station_ts:
            continue

        ts         = station_ts[sid]
        end_excl   = date
        start_incl = end_excl - pd.Timedelta(days=window)

        window_df = ts.loc[start_incl : end_excl - pd.Timedelta(days=1)]

        if len(window_df) == 0:
            continue

        # Right-align into the fixed-length array
        n_days = min(len(window_df), window)
        sequences[i, -n_days:] = window_df.values[-n_days:].astype("float32")
        era5_mask[i] = True
        n_with_ts += 1

        if (i + 1) % 100_000 == 0:
            logger.info(f"  {i+1:,} / {n:,} sequences built")

    logger.info(f"  {n_with_ts:,} / {n:,} samples have ERA5 sequences")
    np.save(str(cache_path), sequences)
    np.save(str(mask_path), era5_mask)
    logger.info(f"  Cached → {cache_path}")
    return sequences, era5_mask


def _build_era5_monthly(
    meta: pd.DataFrame,
    station_ts: dict,
    n_months: int = 12,
    force: bool = False,
) -> tuple:
    """
    Monthly-aggregated ERA5 sequences for Presto.

    Returns:
      monthly : (n_samples, n_months, 4) float32
                Columns: tp_mm (sum), t2m_min_c (min), t2m_max_c (max), t2m_mean_c (mean).
                Right-aligned: column -1 = month before observation.
      era5_mask : (n_samples,) bool
    """
    cache_path = MULTIMODAL_DIR / f"era5_monthly_m{n_months}.npy"
    mask_path  = MULTIMODAL_DIR / f"era5_monthly_mask_m{n_months}.npy"

    if cache_path.exists() and mask_path.exists() and not force:
        logger.info(f"Loading ERA5 monthly sequences from cache (n_months={n_months})")
        return np.load(str(cache_path)), np.load(str(mask_path))

    n = len(meta)
    monthly  = np.zeros((n, n_months, len(PRESTO_ERA5_COLS)), dtype=np.float32)
    era5_mask = np.zeros(n, dtype=bool)

    logger.info(f"Building ERA5 {n_months}-month aggregations for {n:,} samples...")

    for i, row in enumerate(meta.itertuples(index=False)):
        sid  = str(row.station_id)
        date = pd.Timestamp(row.date)

        if sid not in station_ts:
            continue

        ts = station_ts[sid][PRESTO_ERA5_COLS]

        for m in range(n_months):
            month_end   = date - pd.offsets.MonthBegin(m)
            month_start = month_end - pd.offsets.MonthBegin(1)
            month_data  = ts.loc[month_start:month_end - pd.Timedelta(days=1)]

            if len(month_data) == 0:
                continue

            col_idx = n_months - 1 - m
            monthly[i, col_idx, 0] = float(month_data["tp_mm"].sum())
            monthly[i, col_idx, 1] = float(month_data["t2m_min_c"].min())
            monthly[i, col_idx, 2] = float(month_data["t2m_max_c"].max())
            monthly[i, col_idx, 3] = float(month_data["t2m_mean_c"].mean())
            era5_mask[i] = True

        if (i + 1) % 100_000 == 0:
            logger.info(f"  {i+1:,} / {n:,} monthly sequences built")

    np.save(str(cache_path), monthly)
    np.save(str(mask_path), era5_mask)
    logger.info(f"  Cached → {cache_path}")
    return monthly, era5_mask


def _get_tabular_features(meta: pd.DataFrame) -> tuple:
    """HydroATLAS + same-day ERA5 from metadata. Returns (X, feature_names)."""
    feat_cols = []
    df = meta.copy()

    if ATLAS_PATH.exists():
        atlas = pd.read_parquet(ATLAS_PATH)
        atlas["station_id"] = atlas["station_id"].astype(str)
        ha_cols = [c for c in HYDROATLAS_FEATURES if c in atlas.columns]
        ha_to_add = [c for c in ha_cols if c not in df.columns]
        if ha_to_add:
            df = df.merge(
                atlas[["station_id"] + ha_to_add].drop_duplicates("station_id"),
                on="station_id", how="left"
            )
        feat_cols += ha_cols

    era5_meta = [c for c in df.columns if c.startswith("era5_")]
    feat_cols += era5_meta

    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].apply(
        lambda c: c.fillna(c.median() if c.notna().any() else 0.0)
    ).values.astype(np.float32)
    return X, feat_cols


def _get_cv_groups(meta: pd.DataFrame) -> np.ndarray:
    if "fold_key" in meta.columns:
        return pd.Categorical(meta["fold_key"].fillna("unknown")).codes
    return np.zeros(len(meta), dtype=int)


# ── Model 1: MOSAIKS ──────────────────────────────────────────────────────────

class MOSAIKSBaseline:
    """
    Random Fourier Features from flattened ERA5 patch + tabular features → Ridge.
    Reference: Rolf et al., "A generalizable and accessible approach to machine
    learning with global satellite imagery", PNAS 2021.
    """

    def __init__(self, D: int = 2048, sigma: float = 1.0, alpha: float = 1.0,
                 seed: int = 42):
        self.D     = D
        self.sigma = sigma
        self.alpha = alpha
        self.seed  = seed

    def _rff(self, X_flat: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        d   = X_flat.shape[1]
        W   = rng.standard_normal((d, self.D // 2)) / self.sigma
        b   = rng.uniform(0, 2 * np.pi, self.D // 2)
        Z   = X_flat @ W + b
        return np.concatenate([np.cos(Z), np.sin(Z)], axis=1).astype(np.float32)

    def fit_predict_cv(self, patches: np.ndarray, X_tab: np.ndarray,
                       y: np.ndarray, groups: np.ndarray) -> np.ndarray:
        P_flat   = patches.reshape(len(patches), -1)
        P_rff    = self._rff(P_flat)
        X_full   = np.concatenate([P_rff, X_tab], axis=1)

        cv         = GroupKFold(n_splits=max(2, min(5, len(np.unique(groups)))))
        preds      = np.full(len(y), np.nan)
        cal_scores = []  # nonconformity scores for conformal calibration

        for tr, va in cv.split(X_full, y, groups):
            if len(tr) == 0 or len(va) == 0:
                continue
            # Split training → fit (80%) + conformal calibration (20%)
            n_cal   = max(5, int(len(tr) * 0.2))
            fit_idx = tr[:-n_cal]
            cal_idx = tr[-n_cal:]

            sc    = StandardScaler()
            Xfit  = sc.fit_transform(X_full[fit_idx])
            Xcal  = sc.transform(X_full[cal_idx])
            Xva   = sc.transform(X_full[va])

            model = Ridge(alpha=self.alpha)
            model.fit(Xfit, y[fit_idx])
            cal_pred  = model.predict(Xcal)
            preds[va] = model.predict(Xva)
            cal_scores.extend(np.abs(y[cal_idx] - cal_pred).tolist())

        self._cal_scores = np.array(cal_scores)
        return preds

    def conformal_halfwidths(self, levels=(0.80, 0.90, 0.95)) -> list:
        scores = self._cal_scores
        n = len(scores)
        out = []
        for lv in levels:
            alpha = 1 - lv
            q = float(np.quantile(scores,
                                  min(1.0, np.ceil((n + 1) * (1 - alpha)) / n),
                                  method="higher"))
            out.append({"nominal_coverage": lv, "conformal_halfwidth_log": round(q, 4)})
        return out


# ── Model 2a: ERA5-LSTM-Temporal (temporal sequences only, no tabular) ────────

class ERA5LSTMTemporalNet:
    """
    LSTM on ERA5 daily time-series only — no HydroATLAS tabular features.

    Pure temporal baseline: answers "what does a temporal sequence model achieve
    without any static catchment context?"  Pairs with TemporalTrans to give two
    temporal architectures (RNN vs Transformer) in the Temporal row of the table.
    Direct contrast with ERA5LSTMNet (which adds HydroATLAS → Multimodal row).

    Input : (n_samples, window, 7)  ERA5 daily sequences
    Output: (n_samples,)            log1p WQ prediction
    """

    def __init__(self, n_seq_feat: int = N_ERA5,
                 hidden: int = 128, n_layers: int = 2, dropout: float = 0.2,
                 epochs: int = 30, batch: int = 512, lr: float = 1e-3):
        self.n_seq_feat = n_seq_feat
        self.hidden     = hidden
        self.n_layers   = n_layers
        self.dropout    = dropout
        self.epochs     = epochs
        self.batch      = batch
        self.lr         = lr

    def _build(self, dev):
        import torch.nn as nn
        lstm = nn.LSTM(
            input_size=self.n_seq_feat, hidden_size=self.hidden,
            num_layers=self.n_layers, dropout=self.dropout,
            batch_first=True, bidirectional=False,
        )
        head = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        return lstm.to(dev), head.to(dev)

    def fit(self, seqs: np.ndarray, y: np.ndarray):
        import torch, torch.nn as nn
        dev  = torch.device(_get_device())
        lstm, head = self._build(dev)
        params = list(lstm.parameters()) + list(head.parameters())
        opt    = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss()

        Xs = torch.FloatTensor(seqs).to(dev)
        yt = torch.FloatTensor(y).unsqueeze(1).to(dev)
        ds = torch.utils.data.TensorDataset(Xs, yt)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch, shuffle=True, drop_last=True)

        best_loss, patience, wait = 1e9, 5, 0
        for _ in range(self.epochs):
            lstm.train(); head.train()
            epoch_loss = 0.0
            for sb, yb in dl:
                opt.zero_grad()
                _, (h_n, _) = lstm(sb)
                loss = loss_fn(head(h_n[-1]), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                epoch_loss += loss.item()
            sch.step()
            avg = epoch_loss / max(len(dl), 1)
            if avg < best_loss - 1e-4:
                best_loss = avg; wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self._lstm = lstm; self._head = head; self._dev = dev

    def predict(self, seqs: np.ndarray) -> np.ndarray:
        import torch
        self._lstm.eval(); self._head.eval()
        with torch.no_grad():
            Xs = torch.FloatTensor(seqs).to(self._dev)
            _, (h_n, _) = self._lstm(Xs)
            return self._head(h_n[-1]).squeeze(1).cpu().numpy()


# ── Model 2: ERA5-LSTM ────────────────────────────────────────────────────────

class ERA5LSTMNet:
    """
    Bidirectional LSTM on ERA5 daily time-series (window days × 7 channels) +
    HydroATLAS static features → WQ regression.

    Temporal modality baseline in the spirit of Kratzert et al. (HESS 2018/2019)
    who showed LSTM on precipitation/temperature sequences outperforms process models
    for streamflow. Here we ask the same question for lake WQ.
    """

    def __init__(self, n_seq_feat: int, n_tab: int,
                 hidden: int = 128, n_layers: int = 2, dropout: float = 0.2,
                 epochs: int = 30, batch: int = 512, lr: float = 1e-3):
        self.n_seq_feat = n_seq_feat
        self.n_tab      = n_tab
        self.hidden     = hidden
        self.n_layers   = n_layers
        self.dropout    = dropout
        self.epochs     = epochs
        self.batch      = batch
        self.lr         = lr

    def _build(self, dev):
        import torch.nn as nn
        lstm = nn.LSTM(
            input_size=self.n_seq_feat, hidden_size=self.hidden,
            num_layers=self.n_layers, dropout=self.dropout,
            batch_first=True, bidirectional=False,
        )
        head_in = self.hidden + self.n_tab
        head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        return lstm.to(dev), head.to(dev)

    def fit(self, seqs: np.ndarray, X_tab: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        dev = torch.device(_get_device())

        lstm, head = self._build(dev)
        params = list(lstm.parameters()) + list(head.parameters())
        opt    = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss()

        Xs  = torch.FloatTensor(seqs).to(dev)
        Xt  = torch.FloatTensor(X_tab).to(dev)
        yt  = torch.FloatTensor(y).unsqueeze(1).to(dev)
        ds  = torch.utils.data.TensorDataset(Xs, Xt, yt)
        dl  = torch.utils.data.DataLoader(
            ds, batch_size=self.batch, shuffle=True, drop_last=True)

        best_loss, patience, wait = 1e9, 5, 0
        for epoch in range(self.epochs):
            lstm.train(); head.train()
            epoch_loss = 0.0
            for sb, tb, yb in dl:
                opt.zero_grad()
                _, (h_n, _) = lstm(sb)        # h_n: (n_layers, batch, hidden)
                h_last  = h_n[-1]              # (batch, hidden)
                fused   = torch.cat([h_last, tb], dim=1)
                loss    = loss_fn(head(fused), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                epoch_loss += loss.item()
            sch.step()
            avg_loss = epoch_loss / max(len(dl), 1)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss; wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self._lstm = lstm; self._head = head; self._dev = dev

    def predict(self, seqs: np.ndarray, X_tab: np.ndarray) -> np.ndarray:
        import torch
        self._lstm.eval(); self._head.eval()
        with torch.no_grad():
            Xs = torch.FloatTensor(seqs).to(self._dev)
            Xt = torch.FloatTensor(X_tab).to(self._dev)
            _, (h_n, _) = self._lstm(Xs)
            h_last = h_n[-1]
            fused  = torch.cat([h_last, Xt], dim=1)
            return self._head(fused).squeeze(1).cpu().numpy()


# ── Model 3: CrossModalNet ────────────────────────────────────────────────────

def _build_patch_encoder(n_channels: int = 9, d_model: int = 128) -> "nn.Module":
    """CNN → (batch, 9_tokens, d_model). 9 spatial tokens from 3×3 pool."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(n_channels, 32, 3, padding=0), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=0),         nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, d_model, 3, padding=0),    nn.BatchNorm2d(d_model), nn.ReLU(),
        nn.AdaptiveAvgPool2d((3, 3)),
    )


class CrossModalNet:
    """
    Late cross-attention fusion: CNN patch spatial tokens attend to tabular token.

    Spatial tokens: CNN(12×12×9) → AdaptivePool(3×3) → reshape (batch, 9, 128)
    Tabular token:  Linear(n_tab, 128) → LayerNorm → (batch, 1, 128)
    Cross-attention: tabular token queries 9 patch spatial tokens (8 heads)
    Head: LayerNorm(128) → Linear(128, 1)

    Standard image+tabular cross-attention architecture. Stronger ablation than
    FusionNet (simple concatenation) in patch_baseline.py.
    """

    def __init__(self, n_tab: int, d_model: int = 128, n_heads: int = 8,
                 epochs: int = 40, batch: int = 256, lr: float = 1e-3):
        self.n_tab   = n_tab
        self.d_model = d_model
        self.n_heads = n_heads
        self.epochs  = epochs
        self.batch   = batch
        self.lr      = lr

    def fit(self, patches: np.ndarray, X_tab: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        dev = torch.device(_get_device())
        d   = self.d_model

        patch_cnn = _build_patch_encoder(patches.shape[1], d).to(dev)
        tab_enc   = nn.Sequential(
            nn.Linear(self.n_tab, d), nn.LayerNorm(d), nn.ReLU()
        ).to(dev)
        cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=self.n_heads,
            dropout=0.1, batch_first=True
        ).to(dev)
        head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(dev)

        params = (list(patch_cnn.parameters()) + list(tab_enc.parameters()) +
                  list(cross_attn.parameters()) + list(head.parameters()))
        opt     = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss()

        Xp = torch.FloatTensor(patches).to(dev)
        Xt = torch.FloatTensor(X_tab).to(dev)
        yt = torch.FloatTensor(y).unsqueeze(1).to(dev)
        ds = torch.utils.data.TensorDataset(Xp, Xt, yt)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch, shuffle=True, drop_last=True)

        best_loss, patience, wait = 1e9, 5, 0
        for _ in range(self.epochs):
            patch_cnn.train(); tab_enc.train(); cross_attn.train(); head.train()
            epoch_loss = 0.0
            for pb, tb, yb in dl:
                opt.zero_grad()
                # Patch tokens: (batch, d, 3, 3) → (batch, 9, d)
                feat_map    = patch_cnn(pb)
                B, C, H, W  = feat_map.shape
                patch_tokens = feat_map.view(B, C, H * W).permute(0, 2, 1)  # (B, 9, d)
                # Tabular token: (batch, 1, d)
                tab_token    = tab_enc(tb).unsqueeze(1)
                # Cross-attention: tab queries patch
                attn_out, _  = cross_attn(tab_token, patch_tokens, patch_tokens)
                out          = attn_out.squeeze(1)  # (batch, d)
                loss         = loss_fn(head(out), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                epoch_loss += loss.item()
            sch.step()
            avg_loss = epoch_loss / max(len(dl), 1)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss; wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self._patch_cnn  = patch_cnn
        self._tab_enc    = tab_enc
        self._cross_attn = cross_attn
        self._head       = head
        self._dev        = dev
        self._d          = d

    def predict(self, patches: np.ndarray, X_tab: np.ndarray) -> np.ndarray:
        import torch
        self._patch_cnn.eval(); self._tab_enc.eval()
        self._cross_attn.eval(); self._head.eval()
        with torch.no_grad():
            Xp = torch.FloatTensor(patches).to(self._dev)
            Xt = torch.FloatTensor(X_tab).to(self._dev)
            feat_map     = self._patch_cnn(Xp)
            B, C, H, W   = feat_map.shape
            patch_tokens = feat_map.view(B, C, H * W).permute(0, 2, 1)
            tab_token    = self._tab_enc(Xt).unsqueeze(1)
            attn_out, _  = self._cross_attn(tab_token, patch_tokens, patch_tokens)
            out          = attn_out.squeeze(1)
            return self._head(out).squeeze(1).cpu().numpy()


# ── Model 4: TemporalTransformerNet (Presto-architecture, trained from scratch) ─

class TemporalTransformerNet:
    """
    Temporal Transformer encoder on monthly ERA5 sequences → WQ regression.

    Architecture follows Presto (Tseng et al., Remote Sensing 2023):
      - Learnable [CLS] token prepended to the sequence
      - Learnable positional encoding
      - Transformer encoder (L=4 layers, H=8 heads, ffn=256, dropout=0.1)
      - CLS token output → LayerNorm → Linear(d_model, 1)

    Input: (n_samples, T=12, 4) — monthly ERA5 aggregations:
      tp_mm sum, t2m_min_c min, t2m_max_c max, t2m_mean_c mean.

    Trained from scratch on the NordWQ-26 dataset. Presto pretrained weights
    were inaccessible: the 0.0.1 distribution pins numpy==1.23.5 / torch==2.0
    (incompatible with Python 3.12 + torch 2.13), and the default_model.pt file
    is absent from the pip-from-git install. The architecture is faithful; the
    transfer-learning component cannot be evaluated without a compatible environment.

    ERA5 channel normalization applied inside the model matches Presto's published
    normalization for temperature_2m (shift=-272.15 applied as offset to °C values,
    div=35.0) and total_precipitation (div=0.03, our tp_mm / 1000 for unit parity).
    """

    # Presto normalization constants from dataops/pipelines/s1_s2_era5_srtm.py
    # ERA5_SHIFT_VALUES = [-272.15, 0.0]  (temp in K → div; we have °C so adjust)
    # ERA5_DIV_VALUES   = [35.0, 0.03]
    # Our channels:  tp_sum(mm), t_min(°C), t_max(°C), t_mean(°C)
    # Presto temp is in K; we have °C. Equivalent: (°C + 273.15 - 272.15) / 35 = (°C + 1) / 35
    _NORM_SHIFT = np.array([0.0,    1.0,    1.0,    1.0],  dtype=np.float32)   # tp, tmin, tmax, tmean
    _NORM_DIV   = np.array([30.0,  35.0,   35.0,   35.0],  dtype=np.float32)   # tp in mm → /30 ≈ /0.03*1000

    def __init__(self, n_channels: int = 4, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 4, ffn_dim: int = 256, dropout: float = 0.1,
                 epochs: int = 30, batch: int = 256, lr: float = 1e-3):
        self.n_channels = n_channels
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.ffn_dim    = ffn_dim
        self.dropout    = dropout
        self.epochs     = epochs
        self.batch      = batch
        self.lr         = lr

    def _build(self, T: int, dev):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self, n_ch, d, n_heads, n_layers, ffn, T, drop):
                super().__init__()
                self.input_proj  = nn.Linear(n_ch, d)
                self.cls_token   = nn.Parameter(torch.zeros(1, 1, d))
                self.pos_embed   = nn.Parameter(torch.zeros(1, T + 1, d))
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d, nhead=n_heads, dim_feedforward=ffn,
                    dropout=drop, batch_first=True, norm_first=True)
                self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
                self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
                nn.init.trunc_normal_(self.cls_token, std=0.02)
                nn.init.trunc_normal_(self.pos_embed, std=0.02)

            def forward(self, x):                          # x: (B, T, C)
                B = x.size(0)
                tokens = self.input_proj(x)                # (B, T, d)
                cls    = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
                tokens = torch.cat([cls, tokens], dim=1)   # (B, T+1, d)
                tokens = tokens + self.pos_embed
                out    = self.transformer(tokens)           # (B, T+1, d)
                return self.head(out[:, 0])                 # CLS → (B, 1)

        return _Model(self.n_channels, self.d_model, self.n_heads,
                      self.n_layers, self.ffn_dim, T, self.dropout).to(dev)

    def _normalize(self, monthly_era5: np.ndarray) -> np.ndarray:
        return ((monthly_era5 + self._NORM_SHIFT) / self._NORM_DIV).astype(np.float32)

    def fit(self, monthly_era5: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        dev = torch.device(_get_device())
        T   = monthly_era5.shape[1]

        model   = self._build(T, dev)
        opt     = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss()

        Xm = torch.FloatTensor(self._normalize(monthly_era5))
        yt = torch.FloatTensor(y).unsqueeze(1)
        ds = torch.utils.data.TensorDataset(Xm, yt)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch, shuffle=True, drop_last=True)

        best_loss, patience, wait = 1e9, 5, 0
        for _ in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in dl:
                xb = xb.to(dev); yb = yb.to(dev)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
            sch.step()
            avg_loss = epoch_loss / max(len(dl), 1)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss; wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self._model = model; self._dev = dev

    def predict(self, monthly_era5: np.ndarray) -> np.ndarray:
        import torch
        self._model.eval()
        with torch.no_grad():
            Xm = torch.FloatTensor(self._normalize(monthly_era5)).to(self._dev)
            return self._model(Xm).squeeze(1).cpu().numpy()



# ── Model 5: TriModalNet ──────────────────────────────────────────────────────

class TriModalNet:
    """
    Tri-modal fusion: CNN (spatial patch) + LSTM (temporal ERA5) + MLP (tabular).

    Three independent encoders trained jointly:
      CNN encoder   : 12×12×9 ERA5 patch → 128-dim (3-block conv + pool)
      LSTM encoder  : T×7 ERA5 daily sequence → 128-dim (last hidden state)
      Tabular encoder: n_tab HydroATLAS features → 64-dim (MLP)

    Fusion: concat(128, 128, 64) = 320 → LayerNorm → Linear(64) → ReLU → Linear(1)

    Answers the central multimodal question: does jointly encoding all three
    data modalities (spatial meteorological structure, temporal forcing history,
    and static catchment biogeochemistry) outperform any bi-modal baseline?
    Either result — improvement or parity — directly validates the benchmark
    design of AquaNord-SE.

    Note: requires both patches AND ERA5 daily sequences; only samples where the
    station has ERA5 time-series data are included (same restriction as ERA5-LSTM).
    """

    def __init__(self, n_tab: int, n_seq_feat: int = N_ERA5,
                 hidden: int = 128, n_lstm_layers: int = 2,
                 dropout: float = 0.2,
                 epochs: int = 40, batch: int = 256, lr: float = 1e-3):
        self.n_tab        = n_tab
        self.n_seq_feat   = n_seq_feat
        self.hidden       = hidden
        self.n_lstm_layers = n_lstm_layers
        self.dropout      = dropout
        self.epochs       = epochs
        self.batch        = batch
        self.lr           = lr

    def _build(self, n_channels: int, dev):
        import torch.nn as nn

        cnn_enc = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=0),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=0),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 4, 128), nn.ReLU(),
        ).to(dev)

        lstm_enc = nn.LSTM(
            input_size=self.n_seq_feat,
            hidden_size=self.hidden,
            num_layers=self.n_lstm_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=False,
        ).to(dev)

        tab_enc = nn.Sequential(
            nn.Linear(self.n_tab, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        ).to(dev)

        # 128 (CNN) + 128 (LSTM) + 64 (Tabular) = 320
        head = nn.Sequential(
            nn.LayerNorm(320),
            nn.Linear(320, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to(dev)

        return cnn_enc, lstm_enc, tab_enc, head

    def fit(self, patches: np.ndarray, seqs: np.ndarray,
            X_tab: np.ndarray, y: np.ndarray):
        import torch, torch.nn as nn
        dev = torch.device(_get_device())
        cnn_enc, lstm_enc, tab_enc, head = self._build(patches.shape[1], dev)

        params  = (list(cnn_enc.parameters()) + list(lstm_enc.parameters()) +
                   list(tab_enc.parameters()) + list(head.parameters()))
        opt     = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss()

        Xp = torch.FloatTensor(patches).to(dev)
        Xs = torch.FloatTensor(seqs).to(dev)
        Xt = torch.FloatTensor(X_tab).to(dev)
        yt = torch.FloatTensor(y).unsqueeze(1).to(dev)
        ds = torch.utils.data.TensorDataset(Xp, Xs, Xt, yt)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch, shuffle=True, drop_last=True)

        best_loss, patience, wait = 1e9, 5, 0
        for _ in range(self.epochs):
            cnn_enc.train(); lstm_enc.train()
            tab_enc.train(); head.train()
            epoch_loss = 0.0
            for pb, sb, tb, yb in dl:
                opt.zero_grad()
                cnn_feat         = cnn_enc(pb)             # (B, 128)
                _, (h_n, _)      = lstm_enc(sb)
                lstm_feat        = h_n[-1]                  # (B, 128)
                tab_feat         = tab_enc(tb)              # (B, 64)
                fused            = torch.cat(
                    [cnn_feat, lstm_feat, tab_feat], dim=1) # (B, 320)
                loss             = loss_fn(head(fused), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                epoch_loss += loss.item()
            sch.step()
            avg = epoch_loss / max(len(dl), 1)
            if avg < best_loss - 1e-4:
                best_loss = avg; wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self._cnn  = cnn_enc;  self._lstm = lstm_enc
        self._tab  = tab_enc;  self._head = head
        self._dev  = dev

    def predict(self, patches: np.ndarray, seqs: np.ndarray,
                X_tab: np.ndarray) -> np.ndarray:
        import torch
        self._cnn.eval(); self._lstm.eval()
        self._tab.eval(); self._head.eval()
        with torch.no_grad():
            Xp = torch.FloatTensor(patches).to(self._dev)
            Xs = torch.FloatTensor(seqs).to(self._dev)
            Xt = torch.FloatTensor(X_tab).to(self._dev)
            cnn_feat    = self._cnn(Xp)
            _, (h_n, _) = self._lstm(Xs)
            lstm_feat   = h_n[-1]
            tab_feat    = self._tab(Xt)
            fused       = torch.cat([cnn_feat, lstm_feat, tab_feat], dim=1)
            return self._head(fused).squeeze(1).cpu().numpy()


# ── Stratification helpers ────────────────────────────────────────────────────

def _trophic_rows_from_preds(target: str, model_name: str,
                              y_log: np.ndarray, preds: np.ndarray,
                              trophic: np.ndarray) -> list:
    """
    R²/RMSE/MAE by trophic class from existing fold predictions.
    Matches baseline_trophic_results.csv format.
    """
    rows = []
    ok = np.isfinite(preds)
    for tc in (["all"] + sorted({t for t in trophic[ok] if t not in ("nan", "unknown")})):
        mask = ok if tc == "all" else (ok & (trophic == tc))
        if mask.sum() < 5:
            continue
        m = _metrics(y_log[mask], preds[mask])
        rows.append({"target": target, "trophic_class": tc,
                     "model": model_name, "feature_set": "full", **m})
    return rows


def _rolling_origin_multimodal(target: str, model_name: str,
                                meta_sub: pd.DataFrame,
                                fit_fn,    # callable(tr_idx) → model
                                pred_fn,   # callable(model, va_idx) → np.ndarray
                                y_log: np.ndarray) -> list:
    """
    Rolling-origin temporal CV for any multimodal model.

    Expanding training windows (same periods as baseline_rolling_origin.csv):
      Train 2000–2010 → Test 2011–2015
      Train 2000–2015 → Test 2016–2020
      Train 2000–2020 → Test 2021–2025

    meta_sub : aligned DataFrame (rows correspond to y_log).
    fit_fn   : callable(tr_idx) returns a trained model object.
    pred_fn  : callable(model, va_idx) returns prediction array.
    """
    rows = []
    dates = pd.to_datetime(meta_sub["date"].values)
    windows = [
        ("2011-15", pd.Timestamp("2010-12-31"), pd.Timestamp("2011-01-01"), pd.Timestamp("2015-12-31")),
        ("2016-20", pd.Timestamp("2015-12-31"), pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
        ("2021-25", pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")),
    ]
    for period, train_end, test_start, test_end in windows:
        tr_mask = dates <= train_end
        va_mask = (dates >= test_start) & (dates <= test_end)
        tr_idx  = np.where(tr_mask)[0]
        va_idx  = np.where(va_mask)[0]
        if len(tr_idx) < 50 or len(va_idx) < 10:
            continue
        try:
            model = fit_fn(tr_idx)
            preds = pred_fn(model, va_idx)
            m = _metrics(y_log[va_idx], preds)
            rows.append({"target": target, "model": model_name,
                         "test_period": period, **m})
            logger.info("    rolling %s | %s | %s | R²=%.3f",
                        model_name, target, period, m["r2"])
        except Exception as e:
            logger.warning("    rolling %s | %s | %s failed: %s",
                           model_name, target, period, e)
    return rows


def _sensor_coverage_rows(target: str, model_name: str,
                           y_log: np.ndarray, preds: np.ndarray,
                           meta_sub: pd.DataFrame) -> list:
    """
    Stratify predictions by satellite sensor data availability.
    Matches baseline_sensor_coverage.csv format:
      target, grouping, sensor, present, rmse, r2, mae, n

    Uses n_sat_obs_* columns if available in meta_sub (joined from feature table).
    If not present, returns an empty list.
    """
    sensors = ["modis", "landsat", "sentinel1", "sentinel2", "sentinel3"]
    rows = []
    ok = np.isfinite(preds)
    for sensor in sensors:
        col = f"n_sat_obs_{sensor}"
        if col not in meta_sub.columns:
            continue
        sat_counts = meta_sub[col].fillna(0).values
        for present in [True, False]:
            mask = ok & ((sat_counts > 0) if present else (sat_counts == 0))
            if mask.sum() < 5:
                continue
            m = _metrics(y_log[mask], preds[mask])
            rows.append({"target": target, "model": model_name,
                         "grouping": "all", "sensor": sensor,
                         "present": present, **m})
    return rows


def _add_trophic_class(meta: pd.DataFrame) -> pd.DataFrame:
    """Assign trophic class from TP (µg/L) using same bins as baseline_models.py."""
    if "tp_ug_l" not in meta.columns:
        meta = meta.copy()
        meta["trophic_class"] = "unknown"
        return meta
    meta = meta.copy()
    meta["trophic_class"] = pd.cut(
        meta["tp_ug_l"],
        bins=[0, 10, 35, 100, np.inf],
        labels=["oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic"],
    ).astype(str)
    return meta


def _stratified_conformal_rows(target: str, model_name: str,
                                y_log: np.ndarray,
                                preds: np.ndarray,
                                cal_scores: np.ndarray,
                                trophic: np.ndarray,
                                levels=(0.80, 0.90, 0.95)) -> list:
    """
    Compute conformal PICP/ECE/width stratified by trophic class.
    Matches the column format of baseline_quantile_coverage.csv
    (picp, pinaw, winkler_score, ece, n) per (target, trophic_class, level).
    """
    rows = []
    ok   = np.isfinite(preds)
    n_cal = len(cal_scores)
    iqr_y = float(np.percentile(np.expm1(y_log[ok]), 75) -
                  np.percentile(np.expm1(y_log[ok]), 25)) or 1.0

    for lv in levels:
        alpha = 1 - lv
        q = float(np.quantile(cal_scores,
                              min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal),
                              method="higher"))
        for tc_label in (["all"] + sorted(set(trophic[ok]))):
            mask = ok if tc_label == "all" else (ok & (trophic == tc_label))
            n_tc = int(mask.sum())
            if n_tc < 5:
                continue
            lo  = preds[mask] - q
            hi  = preds[mask] + q
            y_m = y_log[mask]
            covered = float(np.mean((y_m >= lo) & (y_m <= hi)))
            width   = float(np.mean(hi - lo))
            # Winkler score (log-space)
            alpha_2 = 2 / (1 - lv)
            wink = width + alpha_2 * np.mean(
                np.maximum(lo - y_m, 0) + np.maximum(y_m - hi, 0))
            rows.append({
                "target":         target,
                "model":          model_name,
                "trophic_class":  tc_label,
                "nominal_coverage": lv,
                "picp":           round(covered, 3),
                "pinaw":          round(width / iqr_y, 4),
                "winkler_score":  round(float(wink), 3),
                "ece":            round(abs(covered - lv), 4),
                "n":              n_tc,
            })
    return rows


def _ac_failure_rows(target: str, model_name: str,
                     y_log: np.ndarray, preds: np.ndarray,
                     ac_flag: np.ndarray) -> list:
    """
    Compute R²/RMSE stratified by AC flag.
    Matches baseline_ac_failure_results.csv format.
    """
    from sklearn.metrics import r2_score, mean_squared_error
    rows = []
    for flag_val, label in [(0, "ac_ok"), (1, "ac_failure")]:
        mask = np.isfinite(preds) & (ac_flag == flag_val)
        if mask.sum() < 5:
            continue
        y0 = np.expm1(y_log[mask])
        p0 = np.expm1(preds[mask])
        rows.append({
            "target":      target,
            "model":       model_name,
            "feature_set": "full",
            "ac_flag":     label,
            "r2":   round(float(r2_score(y0, p0)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y0, p0))), 3),
            "mae":  round(float(np.mean(np.abs(y0 - p0))), 3),
            "n":    int(mask.sum()),
        })
    return rows


# ── Main evaluation loop ──────────────────────────────────────────────────────

def _save_incremental(rows: list, filename: str) -> None:
    """Write accumulated rows to CSV immediately — safe to call after each target."""
    if not rows:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / filename, index=False)
    logger.info("Checkpoint → %s (%d rows)", filename, len(rows))


def run(requested_models: list = None, window: int = 30,
        n_months: int = 12, force_cache: bool = False) -> None:
    try:
        import torch
        torch.backends.cudnn.benchmark = True
    except ImportError:
        if any(m in (requested_models or []) for m in ["lstm", "presto", "crossmodal"]):
            logger.error("PyTorch required for LSTM/Presto/CrossModalNet. pip install torch")
            return

    if requested_models is None:
        requested_models = ["mosaiks", "lstm", "crossmodal", "presto"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading multimodal metadata")
    meta = _load_metadata()
    if meta.empty:
        return

    # Trophic class from TP — same bins as baseline_models.py
    meta = _add_trophic_class(meta)

    # AC failure flag — use rw_blue_negative_flag if present, else None
    ac_flag_col = "rw_blue_negative_flag"
    has_ac_flag = ac_flag_col in meta.columns
    if not has_ac_flag:
        logger.info("rw_blue_negative_flag not in metadata — AC failure stratification skipped")

    logger.info("Loading ERA5 patches (zarr)")
    patches = _load_patches()
    if patches.size == 0:
        return
    assert len(meta) == len(patches), \
        f"metadata rows ({len(meta)}) ≠ patch count ({len(patches)})"

    logger.info("Building tabular features")
    X_tab, tab_cols = _get_tabular_features(meta)
    logger.info(f"  {len(tab_cols)} tabular features")

    logger.info("Building CV groups")
    groups = _get_cv_groups(meta)

    # Load ERA5 station time-series once if any temporal model is requested
    station_ts = None
    needs_era5_ts = any(m in requested_models
                        for m in ["lstm", "lstm_temporal", "presto", "trimodal"])
    if needs_era5_ts:
        station_ts = _load_era5_station_ts()
        if not station_ts:
            logger.warning(
                "No ERA5 daily parquets found — skipping LSTM, LSTM-Temporal, Presto, TriModal")
            requested_models = [m for m in requested_models
                                if m not in ("lstm", "lstm_temporal", "presto", "trimodal")]

    # ERA5 daily sequences (LSTM + LSTM-Temporal)
    sequences, seq_mask = None, None
    if any(m in requested_models for m in ["lstm", "lstm_temporal", "trimodal"]) \
            and station_ts:
        sequences, seq_mask = _build_era5_sequences(
            meta, station_ts, window, force=force_cache)

    # ERA5 monthly aggregations (TemporalTransformerNet / Presto-architecture)
    monthly_era5, monthly_mask = None, None
    if "presto" in requested_models and station_ts:
        monthly_era5, monthly_mask = _build_era5_monthly(
            meta, station_ts, n_months, force=force_cache)

    def _load_ckpt(fname):
        p = OUT_DIR / fname
        return pd.read_csv(p).to_dict("records") if p.exists() else []

    results        = _load_ckpt("multimodal_baseline_results.csv")
    conformal_rows = _load_ckpt("multimodal_conformal_results.csv")
    quantile_rows  = _load_ckpt("multimodal_quantile_coverage.csv")
    ac_rows        = _load_ckpt("multimodal_ac_failure_results.csv")
    trophic_rows   = _load_ckpt("multimodal_trophic_results.csv")
    rolling_rows   = _load_ckpt("multimodal_rolling_origin.csv")
    coverage_rows  = _load_ckpt("multimodal_sensor_coverage.csv")

    # completed_main: main R²/RMSE/MAE results already saved.
    # completed: conformal data present — governs re-runs so models that ran
    # before conformal was added (LSTM-Temporal, TriModal) get re-evaluated.
    completed_main = {(r["model"], r["target"]) for r in results}
    completed = {(r["model"], r["target"]) for r in conformal_rows}
    if completed:
        logger.info("Resuming — conformal done: %s", sorted(completed))
    missing_conf = completed_main - completed
    if missing_conf:
        logger.info("Re-running for missing conformal: %s", sorted(missing_conf))

    # Try to join n_sat_obs_* columns from feature table for sensor coverage
    _sat_cols = [f"n_sat_obs_{s}" for s in
                 ["modis", "landsat", "sentinel1", "sentinel2", "sentinel3"]]
    _ft_sample = Path("data/interim/SE/feature_table_SE_2020.parquet")
    if _ft_sample.exists():
        try:
            _ft_cols = pd.read_parquet(_ft_sample, columns=["station_id", "date"]
                                       + [c for c in _sat_cols
                                          if c in pd.read_parquet(_ft_sample,
                                                                   columns=[]).columns])
        except Exception:
            _ft_cols = None
    else:
        _ft_cols = None
    has_sensor_coverage = _ft_cols is not None and len(_ft_cols.columns) > 2

    for target in TARGETS:
        if target not in meta.columns:
            logger.warning(f"{target} not in metadata — skipping")
            continue

        valid_mask = meta[target].notna() & (meta[target] > 0)
        idx     = np.where(valid_mask)[0]
        y_log   = np.log1p(meta.loc[valid_mask, target].values.astype(np.float32))
        P       = patches[idx]
        T       = X_tab[idx]
        grp     = groups[idx]
        n_splits = max(2, min(5, len(np.unique(grp))))
        cv       = GroupKFold(n_splits=n_splits)
        logger.info(f"\n── {target}  ({len(idx):,} valid samples) ──")

        # ── MOSAIKS ──────────────────────────────────────────────────────────
        if "mosaiks" in requested_models and ("MOSAIKS", target) in completed:
            logger.info(f"  MOSAIKS {target} — skipping (checkpoint)")
        elif "mosaiks" in requested_models:
            try:
                sc_tab  = StandardScaler()
                T_sc    = sc_tab.fit_transform(T)
                mosaiks = MOSAIKSBaseline(D=2048)
                preds   = mosaiks.fit_predict_cv(P, T_sc, y_log, grp)
                rec     = {"target": target, "model": "MOSAIKS",
                           **_metrics(y_log, preds)}
                results.append(rec)
                ok = ~np.isnan(preds)
                # Trophic-stratified conformal (quantile UQ equivalent)
                trophic_vals = meta.loc[valid_mask, "trophic_class"].values
                if len(mosaiks._cal_scores) > 0:
                    quantile_rows += _stratified_conformal_rows(
                        target, "MOSAIKS", y_log, preds,
                        mosaiks._cal_scores, trophic_vals)
                # Trophic R²/RMSE
                trophic_rows += _trophic_rows_from_preds(
                    target, "MOSAIKS", y_log, preds, trophic_vals)
                # AC failure stratification
                if has_ac_flag:
                    ac_vals = meta.loc[valid_mask, ac_flag_col].fillna(0).values.astype(int)
                    ac_rows += _ac_failure_rows(target, "MOSAIKS",
                                                y_log, preds, ac_vals)
                # Sensor coverage
                if has_sensor_coverage:
                    coverage_rows += _sensor_coverage_rows(
                        target, "MOSAIKS", y_log, preds,
                        meta.loc[valid_mask].reset_index(drop=True))
                # Conformal intervals from calibration scores saved during CV
                for hw in mosaiks.conformal_halfwidths():
                    conformal_rows.append({
                        "target": target, "model": "MOSAIKS", **hw,
                        "picp": round(float(np.mean(
                            (y_log[ok] >= preds[ok] - hw["conformal_halfwidth_log"]) &
                            (y_log[ok] <= preds[ok] + hw["conformal_halfwidth_log"]))), 3),
                        "ece": round(abs(float(np.mean(
                            (y_log[ok] >= preds[ok] - hw["conformal_halfwidth_log"]) &
                            (y_log[ok] <= preds[ok] + hw["conformal_halfwidth_log"]))) -
                            hw["nominal_coverage"]), 4),
                    })
                logger.info(f"  MOSAIKS:       RMSE={rec['rmse']:.3f}  R²={rec['r2']:.3f}  n={rec['n']:,}")
                # Rolling-origin temporal evaluation
                _meta_mos = meta.loc[valid_mask].reset_index(drop=True)
                _P_rff_ro = mosaiks._rff(P.reshape(len(P), -1))
                _T_raw_ro = T
                _y_mos    = y_log

                def _fit_mos_ro(tr_idx,
                                _Prff=_P_rff_ro, _Tr=_T_raw_ro, _y=_y_mos):
                    X_tr = np.concatenate([_Prff[tr_idx], _Tr[tr_idx]], axis=1)
                    sc = StandardScaler()
                    m = Ridge(alpha=1.0)
                    m.fit(sc.fit_transform(X_tr), _y[tr_idx])
                    return (m, sc)

                def _pred_mos_ro(bundle, va_idx,
                                 _Prff=_P_rff_ro, _Tr=_T_raw_ro):
                    m, sc = bundle
                    X_va = np.concatenate([_Prff[va_idx], _Tr[va_idx]], axis=1)
                    return m.predict(sc.transform(X_va))

                rolling_rows += _rolling_origin_multimodal(
                    target, "MOSAIKS", _meta_mos,
                    _fit_mos_ro, _pred_mos_ro, y_log)
                completed.add(("MOSAIKS", target))
                _save_incremental(results,        "multimodal_baseline_results.csv")
                _save_incremental(conformal_rows, "multimodal_conformal_results.csv")
                _save_incremental(quantile_rows,  "multimodal_quantile_coverage.csv")
                _save_incremental(trophic_rows,   "multimodal_trophic_results.csv")
                _save_incremental(ac_rows,        "multimodal_ac_failure_results.csv")
                _save_incremental(rolling_rows,   "multimodal_rolling_origin.csv")
                _save_incremental(coverage_rows,  "multimodal_sensor_coverage.csv")
            except Exception as e:
                logger.warning(f"  MOSAIKS failed: {e}")

        # ── ERA5-LSTM ─────────────────────────────────────────────────────────
        if "lstm" in requested_models and sequences is not None:
            # Only use samples where the station has ERA5 daily time-series
            lstm_valid = valid_mask.values & seq_mask
            lstm_idx   = np.where(lstm_valid)[0]
            if len(lstm_idx) < 50:
                logger.warning(f"  ERA5-LSTM: only {len(lstm_idx)} samples with ERA5 sequences — skipping {target}")
            elif (f"ERA5-LSTM-w{window}", target) in completed:
                logger.info(f"  ERA5-LSTM-w{window} {target} — skipping (checkpoint)")
            else:
                try:
                    S     = sequences[lstm_idx]       # (n, window, 7)
                    T_l   = X_tab[lstm_idx]
                    y_l   = np.log1p(meta.loc[lstm_valid, target].values.astype(np.float32))
                    grp_l = groups[lstm_idx]
                    cv_l  = GroupKFold(n_splits=max(2, min(5, len(np.unique(grp_l)))))
                    preds_l      = np.full(len(y_l), np.nan)
                    cal_scores_l = []   # pooled nonconformity scores across folds

                    for tr, va in cv_l.split(S, y_l, grp_l):
                        if len(tr) == 0 or len(va) == 0:
                            continue
                        # Three-way split: fit (80% of tr) + cal (20% of tr) + val
                        n_cal   = max(5, int(len(tr) * 0.2))
                        fit_idx = tr[:-n_cal]
                        cal_idx = tr[-n_cal:]

                        sc_s = StandardScaler()
                        S_fit = sc_s.fit_transform(S[fit_idx].reshape(len(fit_idx), -1)).reshape(len(fit_idx), window, -1)
                        S_cal = sc_s.transform(S[cal_idx].reshape(len(cal_idx), -1)).reshape(len(cal_idx), window, -1)
                        S_va  = sc_s.transform(S[va].reshape(len(va), -1)).reshape(len(va), window, -1)
                        sc_t = StandardScaler()
                        T_fit = sc_t.fit_transform(T_l[fit_idx])
                        T_cal = sc_t.transform(T_l[cal_idx])
                        T_va  = sc_t.transform(T_l[va])

                        m = ERA5LSTMNet(n_seq_feat=S.shape[2], n_tab=T_l.shape[1])
                        m.fit(S_fit, T_fit, y_l[fit_idx])
                        cal_scores_l.extend(np.abs(y_l[cal_idx] - m.predict(S_cal, T_cal)).tolist())
                        preds_l[va] = m.predict(S_va, T_va)

                    rec = {"target": target, "model": f"ERA5-LSTM-w{window}",
                           **_metrics(y_l, preds_l)}
                    results.append(rec)
                    logger.info(f"  ERA5-LSTM:     RMSE={rec['rmse']:.3f}  R²={rec['r2']:.3f}  n={rec['n']:,}")
                    ok = ~np.isnan(preds_l)
                    # Trophic-stratified conformal
                    trophic_l = meta.loc[lstm_valid, "trophic_class"].values
                    if cal_scores_l:
                        quantile_rows += _stratified_conformal_rows(
                            target, f"ERA5-LSTM-w{window}", y_l, preds_l,
                            np.array(cal_scores_l), trophic_l)
                    # Trophic R²/RMSE
                    trophic_rows += _trophic_rows_from_preds(
                        target, f"ERA5-LSTM-w{window}", y_l, preds_l, trophic_l)
                    if has_ac_flag:
                        ac_l = meta.loc[lstm_valid, ac_flag_col].fillna(0).values.astype(int)
                        ac_rows += _ac_failure_rows(target, f"ERA5-LSTM-w{window}",
                                                    y_l, preds_l, ac_l)
                    if has_sensor_coverage:
                        coverage_rows += _sensor_coverage_rows(
                            target, f"ERA5-LSTM-w{window}", y_l, preds_l,
                            meta.loc[lstm_valid].reset_index(drop=True))
                    # Proper split conformal using pooled held-out calibration scores
                    if cal_scores_l and ok.sum() > 20:
                        scores = np.array(cal_scores_l)
                        n_cal_total = len(scores)
                        for lv in (0.80, 0.90, 0.95):
                            alpha = 1 - lv
                            q = float(np.quantile(
                                scores,
                                min(1.0, np.ceil((n_cal_total + 1) * (1 - alpha)) / n_cal_total),
                                method="higher"))
                            picp = float(np.mean(
                                (y_l[ok] >= preds_l[ok] - q) & (y_l[ok] <= preds_l[ok] + q)))
                            conformal_rows.append({
                                "target": target, "model": f"ERA5-LSTM-w{window}",
                                "nominal_coverage": lv,
                                "conformal_halfwidth_log": round(q, 4),
                                "picp": round(picp, 3),
                                "ece": round(abs(picp - lv), 4),
                            })
                    # Rolling-origin temporal evaluation
                    _meta_lstm = meta.loc[lstm_valid].reset_index(drop=True)
                    _S_ro = S; _Tl_ro = T_l; _y_ro = y_l; _w_ro = window

                    def _fit_lstm_ro(tr_idx,
                                     _S=_S_ro, _T=_Tl_ro, _y=_y_ro, _w=_w_ro):
                        sc_s = StandardScaler()
                        S_tr = sc_s.fit_transform(
                            _S[tr_idx].reshape(len(tr_idx), -1)
                        ).reshape(len(tr_idx), _w, -1)
                        sc_t = StandardScaler()
                        T_tr = sc_t.fit_transform(_T[tr_idx])
                        m = ERA5LSTMNet(n_seq_feat=_S.shape[2], n_tab=_T.shape[1])
                        m.fit(S_tr, T_tr, _y[tr_idx])
                        return (m, sc_s, sc_t)

                    def _pred_lstm_ro(bundle, va_idx,
                                      _S=_S_ro, _T=_Tl_ro, _w=_w_ro):
                        m, sc_s, sc_t = bundle
                        S_va = sc_s.transform(
                            _S[va_idx].reshape(len(va_idx), -1)
                        ).reshape(len(va_idx), _w, -1)
                        return m.predict(S_va, sc_t.transform(_T[va_idx]))

                    rolling_rows += _rolling_origin_multimodal(
                        target, f"ERA5-LSTM-w{window}", _meta_lstm,
                        _fit_lstm_ro, _pred_lstm_ro, y_l)
                    completed.add((f"ERA5-LSTM-w{window}", target))
                    _save_incremental(results,        "multimodal_baseline_results.csv")
                    _save_incremental(conformal_rows, "multimodal_conformal_results.csv")
                    _save_incremental(quantile_rows,  "multimodal_quantile_coverage.csv")
                    _save_incremental(trophic_rows,   "multimodal_trophic_results.csv")
                    _save_incremental(ac_rows,        "multimodal_ac_failure_results.csv")
                    _save_incremental(rolling_rows,   "multimodal_rolling_origin.csv")
                    _save_incremental(coverage_rows,  "multimodal_sensor_coverage.csv")
                except Exception as e:
                    logger.warning(f"  ERA5-LSTM failed: {e}")

        # ── ERA5-LSTM-Temporal (sequences only, no tabular) ───────────────────
        if "lstm_temporal" in requested_models and sequences is not None:
            lt_valid = valid_mask.values & seq_mask
            lt_idx   = np.where(lt_valid)[0]
            if len(lt_idx) < 50:
                logger.warning(
                    f"  LSTM-Temporal: only {len(lt_idx)} samples — skipping {target}")
            elif ("LSTM-Temporal", target) in completed:
                logger.info(f"  LSTM-Temporal {target} — skipping (checkpoint)")
            else:
                try:
                    S_lt  = sequences[lt_idx]
                    y_lt  = np.log1p(
                        meta.loc[lt_valid, target].values.astype(np.float32))
                    grp_lt = groups[lt_idx]
                    cv_lt  = GroupKFold(
                        n_splits=max(2, min(5, len(np.unique(grp_lt)))))
                    preds_lt     = np.full(len(y_lt), np.nan)
                    cal_scores_lt = []

                    for tr, va in cv_lt.split(S_lt, y_lt, grp_lt):
                        if len(tr) == 0 or len(va) == 0:
                            continue
                        n_cal   = max(5, int(len(tr) * 0.2))
                        fit_idx = tr[:-n_cal]
                        cal_idx = tr[-n_cal:]
                        m = ERA5LSTMTemporalNet()
                        m.fit(S_lt[fit_idx], y_lt[fit_idx])
                        cal_scores_lt.extend(
                            np.abs(y_lt[cal_idx] - m.predict(S_lt[cal_idx])).tolist())
                        preds_lt[va] = m.predict(S_lt[va])

                    rec = {"target": target, "model": "LSTM-Temporal",
                           **_metrics(y_lt, preds_lt)}
                    if ("LSTM-Temporal", target) not in completed_main:
                        results.append(rec)
                    logger.info(
                        f"  LSTM-Temporal: RMSE={rec['rmse']:.3f}  "
                        f"R²={rec['r2']:.3f}  n={rec['n']:,}")
                    ok_lt = ~np.isnan(preds_lt)
                    trophic_lt = meta.loc[lt_valid, "trophic_class"].values
                    trophic_rows += _trophic_rows_from_preds(
                        target, "LSTM-Temporal", y_lt, preds_lt, trophic_lt)
                    if cal_scores_lt and ok_lt.sum() > 20:
                        scores_lt  = np.array(cal_scores_lt)
                        n_cal_lt   = len(scores_lt)
                        quantile_rows += _stratified_conformal_rows(
                            target, "LSTM-Temporal", y_lt, preds_lt,
                            scores_lt, trophic_lt)
                        for lv in (0.80, 0.90, 0.95):
                            alpha = 1 - lv
                            q = float(np.quantile(
                                scores_lt,
                                min(1.0, np.ceil((n_cal_lt + 1) * (1 - alpha)) / n_cal_lt),
                                method="higher"))
                            picp = float(np.mean(
                                (y_lt[ok_lt] >= preds_lt[ok_lt] - q) &
                                (y_lt[ok_lt] <= preds_lt[ok_lt] + q)))
                            conformal_rows.append({
                                "target": target, "model": "LSTM-Temporal",
                                "nominal_coverage": lv,
                                "conformal_halfwidth_log": round(q, 4),
                                "picp": round(picp, 3),
                                "ece": round(abs(picp - lv), 4),
                            })
                    if has_ac_flag:
                        ac_lt = meta.loc[lt_valid, ac_flag_col].fillna(0).values.astype(int)
                        ac_rows += _ac_failure_rows(target, "LSTM-Temporal",
                                                    y_lt, preds_lt, ac_lt)
                    if has_sensor_coverage:
                        coverage_rows += _sensor_coverage_rows(
                            target, "LSTM-Temporal", y_lt, preds_lt,
                            meta.loc[lt_valid].reset_index(drop=True))
                    # Rolling-origin temporal evaluation
                    _meta_lt = meta.loc[lt_valid].reset_index(drop=True)
                    _S_lt_ro = S_lt; _y_lt_ro = y_lt

                    def _fit_lt_ro(tr_idx, _S=_S_lt_ro, _y=_y_lt_ro):
                        m = ERA5LSTMTemporalNet()
                        m.fit(_S[tr_idx], _y[tr_idx])
                        return m

                    def _pred_lt_ro(m, va_idx, _S=_S_lt_ro):
                        return m.predict(_S[va_idx])

                    rolling_rows += _rolling_origin_multimodal(
                        target, "LSTM-Temporal", _meta_lt,
                        _fit_lt_ro, _pred_lt_ro, y_lt)
                    completed.add(("LSTM-Temporal", target))
                    _save_incremental(results,        "multimodal_baseline_results.csv")
                    _save_incremental(conformal_rows, "multimodal_conformal_results.csv")
                    _save_incremental(quantile_rows,  "multimodal_quantile_coverage.csv")
                    _save_incremental(trophic_rows,   "multimodal_trophic_results.csv")
                    _save_incremental(ac_rows,        "multimodal_ac_failure_results.csv")
                    _save_incremental(rolling_rows,   "multimodal_rolling_origin.csv")
                    _save_incremental(coverage_rows,  "multimodal_sensor_coverage.csv")
                except Exception as e:
                    logger.warning(f"  LSTM-Temporal failed: {e}")

        # ── CrossModalNet ─────────────────────────────────────────────────────
        if "crossmodal" in requested_models and ("CrossModalNet", target) in completed:
            logger.info(f"  CrossModalNet {target} — skipping (checkpoint)")
        elif "crossmodal" in requested_models:
            try:
                preds_c      = np.full(len(y_log), np.nan)
                cal_scores_c = []

                for tr, va in cv.split(P, y_log, grp):
                    if len(tr) == 0 or len(va) == 0:
                        continue
                    n_cal   = max(5, int(len(tr) * 0.2))
                    fit_idx = tr[:-n_cal]
                    cal_idx = tr[-n_cal:]

                    sc_t  = StandardScaler()
                    T_fit = sc_t.fit_transform(T[fit_idx])
                    T_cal = sc_t.transform(T[cal_idx])
                    T_va  = sc_t.transform(T[va])

                    m = CrossModalNet(n_tab=T.shape[1])
                    m.fit(P[fit_idx], T_fit, y_log[fit_idx])
                    cal_scores_c.extend(np.abs(y_log[cal_idx] - m.predict(P[cal_idx], T_cal)).tolist())
                    preds_c[va] = m.predict(P[va], T_va)

                rec = {"target": target, "model": "CrossModalNet",
                       **_metrics(y_log, preds_c)}
                results.append(rec)
                logger.info(f"  CrossModalNet: RMSE={rec['rmse']:.3f}  R²={rec['r2']:.3f}  n={rec['n']:,}")
                ok = ~np.isnan(preds_c)
                trophic_c = meta.loc[valid_mask, "trophic_class"].values
                if cal_scores_c:
                    quantile_rows += _stratified_conformal_rows(
                        target, "CrossModalNet", y_log, preds_c,
                        np.array(cal_scores_c), trophic_c)
                trophic_rows += _trophic_rows_from_preds(
                    target, "CrossModalNet", y_log, preds_c, trophic_c)
                if has_ac_flag:
                    ac_c = meta.loc[valid_mask, ac_flag_col].fillna(0).values.astype(int)
                    ac_rows += _ac_failure_rows(target, "CrossModalNet",
                                                y_log, preds_c, ac_c)
                if has_sensor_coverage:
                    coverage_rows += _sensor_coverage_rows(
                        target, "CrossModalNet", y_log, preds_c,
                        meta.loc[valid_mask].reset_index(drop=True))
                if cal_scores_c and ok.sum() > 20:
                    scores = np.array(cal_scores_c)
                    n_c = len(scores)
                    for lv in (0.80, 0.90, 0.95):
                        alpha = 1 - lv
                        q = float(np.quantile(scores,
                                              min(1.0, np.ceil((n_c + 1) * (1 - alpha)) / n_c),
                                              method="higher"))
                        picp = float(np.mean(
                            (y_log[ok] >= preds_c[ok] - q) & (y_log[ok] <= preds_c[ok] + q)))
                        conformal_rows.append({
                            "target": target, "model": "CrossModalNet",
                            "nominal_coverage": lv,
                            "conformal_halfwidth_log": round(q, 4),
                            "picp": round(picp, 3),
                            "ece": round(abs(picp - lv), 4),
                        })
                # Rolling-origin temporal evaluation
                _meta_cm = meta.loc[valid_mask].reset_index(drop=True)
                _P_cm = P; _T_cm = T; _y_cm = y_log

                def _fit_cm_ro(tr_idx, _P=_P_cm, _T=_T_cm, _y=_y_cm):
                    sc_t = StandardScaler()
                    T_tr = sc_t.fit_transform(_T[tr_idx])
                    m = CrossModalNet(n_tab=_T.shape[1])
                    m.fit(_P[tr_idx], T_tr, _y[tr_idx])
                    return (m, sc_t)

                def _pred_cm_ro(bundle, va_idx, _P=_P_cm, _T=_T_cm):
                    m, sc_t = bundle
                    return m.predict(_P[va_idx], sc_t.transform(_T[va_idx]))

                rolling_rows += _rolling_origin_multimodal(
                    target, "CrossModalNet", _meta_cm,
                    _fit_cm_ro, _pred_cm_ro, y_log)
                completed.add(("CrossModalNet", target))
                _save_incremental(results,        "multimodal_baseline_results.csv")
                _save_incremental(conformal_rows, "multimodal_conformal_results.csv")
                _save_incremental(quantile_rows,  "multimodal_quantile_coverage.csv")
                _save_incremental(trophic_rows,   "multimodal_trophic_results.csv")
                _save_incremental(ac_rows,        "multimodal_ac_failure_results.csv")
                _save_incremental(rolling_rows,   "multimodal_rolling_origin.csv")
                _save_incremental(coverage_rows,  "multimodal_sensor_coverage.csv")
            except Exception as e:
                logger.warning(f"  CrossModalNet failed: {e}")

        # ── TemporalTransformerNet (Presto-architecture) ──────────────────────
        if "presto" in requested_models and monthly_era5 is not None:
            tt_valid = valid_mask.values & monthly_mask
            tt_idx   = np.where(tt_valid)[0]
            if len(tt_idx) < 50:
                logger.warning(f"  TemporalTrans: only {len(tt_idx)} samples with monthly ERA5 — skipping {target}")
            elif ("TemporalTrans-ERA5", target) in completed:
                logger.info(f"  TemporalTrans-ERA5 {target} — skipping (checkpoint)")
            else:
                try:
                    M_p     = monthly_era5[tt_idx]
                    y_p     = np.log1p(meta.loc[tt_valid, target].values.astype(np.float32))
                    grp_p   = groups[tt_idx]
                    cv_p    = GroupKFold(n_splits=max(2, min(5, len(np.unique(grp_p)))))
                    preds_p      = np.full(len(y_p), np.nan)
                    cal_scores_p = []

                    for tr, va in cv_p.split(M_p, y_p, grp_p):
                        if len(tr) == 0 or len(va) == 0:
                            continue
                        n_cal   = max(5, int(len(tr) * 0.2))
                        fit_idx = tr[:-n_cal]
                        cal_idx = tr[-n_cal:]

                        m = TemporalTransformerNet()
                        m.fit(M_p[fit_idx], y_p[fit_idx])
                        cal_scores_p.extend(np.abs(y_p[cal_idx] - m.predict(M_p[cal_idx])).tolist())
                        preds_p[va] = m.predict(M_p[va])

                    rec = {"target": target, "model": "TemporalTrans-ERA5",
                           **_metrics(y_p, preds_p)}
                    results.append(rec)
                    logger.info(f"  TemporalTrans: RMSE={rec['rmse']:.3f}  R²={rec['r2']:.3f}  n={rec['n']:,}")
                    ok = ~np.isnan(preds_p)
                    trophic_p = meta.loc[tt_valid, "trophic_class"].values
                    if cal_scores_p:
                        quantile_rows += _stratified_conformal_rows(
                            target, "TemporalTrans-ERA5", y_p, preds_p,
                            np.array(cal_scores_p), trophic_p)
                    trophic_rows += _trophic_rows_from_preds(
                        target, "TemporalTrans-ERA5", y_p, preds_p, trophic_p)
                    if has_ac_flag:
                        ac_p = meta.loc[tt_valid, ac_flag_col].fillna(0).values.astype(int)
                        ac_rows += _ac_failure_rows(target, "TemporalTrans-ERA5",
                                                    y_p, preds_p, ac_p)
                    if has_sensor_coverage:
                        coverage_rows += _sensor_coverage_rows(
                            target, "TemporalTrans-ERA5", y_p, preds_p,
                            meta.loc[tt_valid].reset_index(drop=True))
                    if cal_scores_p and ok.sum() > 20:
                        scores = np.array(cal_scores_p)
                        n_p = len(scores)
                        for lv in (0.80, 0.90, 0.95):
                            alpha = 1 - lv
                            q = float(np.quantile(scores,
                                                  min(1.0, np.ceil((n_p + 1) * (1 - alpha)) / n_p),
                                                  method="higher"))
                            picp = float(np.mean(
                                (y_p[ok] >= preds_p[ok] - q) & (y_p[ok] <= preds_p[ok] + q)))
                            conformal_rows.append({
                                "target": target, "model": "TemporalTrans-ERA5",
                                "nominal_coverage": lv,
                                "conformal_halfwidth_log": round(q, 4),
                                "picp": round(picp, 3),
                                "ece": round(abs(picp - lv), 4),
                            })
                    # Rolling-origin temporal evaluation
                    _meta_tt = meta.loc[tt_valid].reset_index(drop=True)
                    _M_tt = M_p; _y_tt = y_p

                    def _fit_tt_ro(tr_idx, _M=_M_tt, _y=_y_tt):
                        m = TemporalTransformerNet()
                        m.fit(_M[tr_idx], _y[tr_idx])
                        return m

                    def _pred_tt_ro(m, va_idx, _M=_M_tt):
                        return m.predict(_M[va_idx])

                    rolling_rows += _rolling_origin_multimodal(
                        target, "TemporalTrans-ERA5", _meta_tt,
                        _fit_tt_ro, _pred_tt_ro, y_p)
                    completed.add(("TemporalTrans-ERA5", target))
                    _save_incremental(results,        "multimodal_baseline_results.csv")
                    _save_incremental(conformal_rows, "multimodal_conformal_results.csv")
                    _save_incremental(quantile_rows,  "multimodal_quantile_coverage.csv")
                    _save_incremental(trophic_rows,   "multimodal_trophic_results.csv")
                    _save_incremental(ac_rows,        "multimodal_ac_failure_results.csv")
                    _save_incremental(rolling_rows,   "multimodal_rolling_origin.csv")
                    _save_incremental(coverage_rows,  "multimodal_sensor_coverage.csv")
                except Exception as e:
                    logger.warning(f"  TemporalTrans failed: {e}")

        # ── TriModalNet: CNN + LSTM + Tabular ─────────────────────────────────
        if "trimodal" in requested_models and sequences is not None:
            tri_valid = valid_mask.values & seq_mask
            tri_idx   = np.where(tri_valid)[0]
            if len(tri_idx) < 50:
                logger.warning(
                    f"  TriModal: only {len(tri_idx)} samples with ERA5 — skipping {target}")
            elif ("TriModal", target) in completed:
                logger.info(f"  TriModal {target} — skipping (checkpoint)")
            else:
                try:
                    P_tri   = patches[tri_idx]
                    S_tri   = sequences[tri_idx]
                    T_tri   = X_tab[tri_idx]
                    y_tri   = np.log1p(
                        meta.loc[tri_valid, target].values.astype(np.float32))
                    grp_tri = groups[tri_idx]
                    cv_tri  = GroupKFold(
                        n_splits=max(2, min(5, len(np.unique(grp_tri)))))
                    preds_tri      = np.full(len(y_tri), np.nan)
                    cal_scores_tri = []
                    sc_tri         = StandardScaler()

                    for tr, va in cv_tri.split(P_tri, y_tri, grp_tri):
                        if len(tr) == 0 or len(va) == 0:
                            continue
                        n_cal   = max(5, int(len(tr) * 0.2))
                        fit_idx = tr[:-n_cal]
                        cal_idx = tr[-n_cal:]
                        T_fit = sc_tri.fit_transform(T_tri[fit_idx])
                        T_cal = sc_tri.transform(T_tri[cal_idx])
                        T_va  = sc_tri.transform(T_tri[va])
                        m = TriModalNet(n_tab=T_fit.shape[1])
                        m.fit(P_tri[fit_idx], S_tri[fit_idx], T_fit, y_tri[fit_idx])
                        cal_scores_tri.extend(
                            np.abs(y_tri[cal_idx] -
                                   m.predict(P_tri[cal_idx], S_tri[cal_idx], T_cal)).tolist())
                        preds_tri[va] = m.predict(P_tri[va], S_tri[va], T_va)

                    rec = {"target": target, "model": "TriModal",
                           **_metrics(y_tri, preds_tri)}
                    if ("TriModal", target) not in completed_main:
                        results.append(rec)
                    logger.info(
                        f"  TriModal:      RMSE={rec['rmse']:.3f}  "
                        f"R²={rec['r2']:.3f}  n={rec['n']:,}")
                    ok_tri      = ~np.isnan(preds_tri)
                    trophic_tri = meta.loc[tri_valid, "trophic_class"].values
                    trophic_rows += _trophic_rows_from_preds(
                        target, "TriModal", y_tri, preds_tri, trophic_tri)
                    if cal_scores_tri and ok_tri.sum() > 20:
                        scores_tri  = np.array(cal_scores_tri)
                        n_cal_tri   = len(scores_tri)
                        quantile_rows += _stratified_conformal_rows(
                            target, "TriModal", y_tri, preds_tri,
                            scores_tri, trophic_tri)
                        for lv in (0.80, 0.90, 0.95):
                            alpha = 1 - lv
                            q = float(np.quantile(
                                scores_tri,
                                min(1.0, np.ceil((n_cal_tri + 1) * (1 - alpha)) / n_cal_tri),
                                method="higher"))
                            picp = float(np.mean(
                                (y_tri[ok_tri] >= preds_tri[ok_tri] - q) &
                                (y_tri[ok_tri] <= preds_tri[ok_tri] + q)))
                            conformal_rows.append({
                                "target": target, "model": "TriModal",
                                "nominal_coverage": lv,
                                "conformal_halfwidth_log": round(q, 4),
                                "picp": round(picp, 3),
                                "ece": round(abs(picp - lv), 4),
                            })
                    if has_ac_flag:
                        ac_tri = meta.loc[tri_valid, ac_flag_col].fillna(0).values.astype(int)
                        ac_rows += _ac_failure_rows(target, "TriModal",
                                                    y_tri, preds_tri, ac_tri)
                    if has_sensor_coverage:
                        coverage_rows += _sensor_coverage_rows(
                            target, "TriModal", y_tri, preds_tri,
                            meta.loc[tri_valid].reset_index(drop=True))
                    # Rolling-origin temporal evaluation
                    _meta_tri = meta.loc[tri_valid].reset_index(drop=True)
                    _P_tri_ro = P_tri; _S_tri_ro = S_tri
                    _T_tri_ro = T_tri; _y_tri_ro = y_tri

                    def _fit_tri_ro(tr_idx,
                                    _P=_P_tri_ro, _S=_S_tri_ro,
                                    _T=_T_tri_ro, _y=_y_tri_ro):
                        sc = StandardScaler()
                        T_tr = sc.fit_transform(_T[tr_idx])
                        m = TriModalNet(n_tab=T_tr.shape[1])
                        m.fit(_P[tr_idx], _S[tr_idx], T_tr, _y[tr_idx])
                        return (m, sc)

                    def _pred_tri_ro(bundle, va_idx,
                                     _P=_P_tri_ro, _S=_S_tri_ro, _T=_T_tri_ro):
                        m, sc = bundle
                        return m.predict(_P[va_idx], _S[va_idx], sc.transform(_T[va_idx]))

                    rolling_rows += _rolling_origin_multimodal(
                        target, "TriModal", _meta_tri,
                        _fit_tri_ro, _pred_tri_ro, y_tri)
                    completed.add(("TriModal", target))
                    _save_incremental(results,        "multimodal_baseline_results.csv")
                    _save_incremental(conformal_rows, "multimodal_conformal_results.csv")
                    _save_incremental(quantile_rows,  "multimodal_quantile_coverage.csv")
                    _save_incremental(trophic_rows,   "multimodal_trophic_results.csv")
                    _save_incremental(ac_rows,        "multimodal_ac_failure_results.csv")
                    _save_incremental(rolling_rows,   "multimodal_rolling_origin.csv")
                    _save_incremental(coverage_rows,  "multimodal_sensor_coverage.csv")
                except Exception as e:
                    logger.warning(f"  TriModal failed: {e}")

        # ── Incremental checkpoint after each target ──────────────────────────
        _save_incremental(results,       "multimodal_baseline_results.csv")
        _save_incremental(conformal_rows,"multimodal_conformal_results.csv")
        _save_incremental(quantile_rows, "multimodal_quantile_coverage.csv")
        _save_incremental(trophic_rows,  "multimodal_trophic_results.csv")
        _save_incremental(ac_rows,       "multimodal_ac_failure_results.csv")
        _save_incremental(rolling_rows,  "multimodal_rolling_origin.csv")
        _save_incremental(coverage_rows, "multimodal_sensor_coverage.csv")

    if not results:
        logger.error("No results produced")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "multimodal_baseline_results.csv", index=False)
    logger.info(f"\nSaved → {OUT_DIR / 'multimodal_baseline_results.csv'}")
    if conformal_rows:
        pd.DataFrame(conformal_rows).to_csv(
            OUT_DIR / "multimodal_conformal_results.csv", index=False)
        logger.info(f"Saved → {OUT_DIR / 'multimodal_conformal_results.csv'}")
    if quantile_rows:
        pd.DataFrame(quantile_rows).to_csv(
            OUT_DIR / "multimodal_quantile_coverage.csv", index=False)
        logger.info(f"Saved → {OUT_DIR / 'multimodal_quantile_coverage.csv'}")
    if ac_rows:
        pd.DataFrame(ac_rows).to_csv(
            OUT_DIR / "multimodal_ac_failure_results.csv", index=False)
        logger.info(f"Saved → {OUT_DIR / 'multimodal_ac_failure_results.csv'}")
    if trophic_rows:
        pd.DataFrame(trophic_rows).to_csv(
            OUT_DIR / "multimodal_trophic_results.csv", index=False)
        logger.info(f"Saved → {OUT_DIR / 'multimodal_trophic_results.csv'}")
    if rolling_rows:
        pd.DataFrame(rolling_rows).to_csv(
            OUT_DIR / "multimodal_rolling_origin.csv", index=False)
        logger.info(f"Saved → {OUT_DIR / 'multimodal_rolling_origin.csv'}")
    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(
            OUT_DIR / "multimodal_sensor_coverage.csv", index=False)
        logger.info(f"Saved → {OUT_DIR / 'multimodal_sensor_coverage.csv'}")

    try:
        pivot = df.pivot_table(
            index="model", columns="target",
            values=["rmse", "r2"], aggfunc="first")
        pivot.to_latex(str(OUT_DIR / "multimodal_baseline_results.tex"),
                       float_format="%.3f", escape=False)
        logger.info(f"Saved → {OUT_DIR / 'multimodal_baseline_results.tex'}")
    except Exception as e:
        logger.warning(f"LaTeX export: {e}")

    logger.info("\n" + df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal baselines for NordWQ-26 (CIKM 2026)")
    parser.add_argument(
        "--models", nargs="+",
        default=["mosaiks", "lstm", "lstm_temporal",
                 "crossmodal", "presto", "trimodal"],
        choices=["mosaiks", "lstm", "lstm_temporal",
                 "crossmodal", "presto", "trimodal"],
        help="Models to run (default: all)")
    parser.add_argument(
        "--window", type=int, default=30,
        help="ERA5 daily lookback window in days for LSTM (default: 30)")
    parser.add_argument(
        "--n-months", type=int, default=12,
        help="ERA5 monthly lookback for Presto (default: 12)")
    parser.add_argument(
        "--force-cache", action="store_true",
        help="Rebuild ERA5 sequence caches even if they exist")
    args = parser.parse_args()
    run(args.models, args.window, args.n_months, args.force_cache)


if __name__ == "__main__":
    main()
