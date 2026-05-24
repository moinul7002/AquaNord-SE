#!/usr/bin/env python3
"""
Spatial patch baselines for NordWQ-26 dataset paper (CIKM 2026).

Evaluates ERA5 12×12×9 spatial patches as a modality for WQ prediction,
alone and fused with tabular features.

Models:
  Patch-XGB   — flatten ERA5 patch (1296-dim) → XGBoost    [non-spatial baseline]
  CNN         — 3-block convolutional network on 12×12×9 patch
  CNN+Tabular — late fusion: CNN patch encoder + MLP tabular branch → combined head

Architecture details:
  CNN encoder: Conv2d(9→32, 3×3) → BN → ReLU → Conv2d(32→64, 3×3) → BN → ReLU
               → AdaptiveAvgPool(4×4) → flatten → Linear(1024→128)
  Tabular branch: Linear(n_tab→128) → BN → ReLU → Linear(128→64)
  Fusion head: concat(patch_emb, tab_emb) → Linear(192→64) → ReLU → Linear(64→1)

Targets (log1p-transformed): chla_ug_l, tp_ug_l, secchi_m

Spatial CV: same GroupKFold splits as baseline_models.py (SMHI basin keys).

Outputs:
  data/outputs/patch_baseline_results.csv   — RMSE / R² / MAE
  data/outputs/patch_baseline_results.tex   — LaTeX table

Usage:
    python -m src.evaluation.patch_baseline
    python -m src.evaluation.patch_baseline --models cnn fusion
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MULTIMODAL_DIR = Path("data/processed/nordic_multimodal_dataset")
INTERIM        = Path("data/interim/SE")
ATLAS_PATH     = Path("data/raw/hydroatlas/hydroatlas_station_covariates_SE.parquet")
SPLIT_PATH     = MULTIMODAL_DIR / "split_index.parquet"
OUT_DIR        = Path("data/outputs")

TARGETS = ["chla_ug_l", "tp_ug_l", "secchi_m"]

ROLLING_WINDOWS = [
    ("2011-15", pd.Timestamp("2010-12-31"),
     pd.Timestamp("2011-01-01"), pd.Timestamp("2015-12-31")),
    ("2016-20", pd.Timestamp("2015-12-31"),
     pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
    ("2021-25", pd.Timestamp("2020-12-31"),
     pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")),
]

HYDROATLAS_FEATURES = [
    "for_pc_use", "crp_pc_use", "pst_pc_use", "urb_pc_use", "wet_pc_ug1",
    "pre_mm_uyr", "ele_mt_uav", "soc_th_uav", "dis_m3_pyr", "tmp_dc_uyr",
    "ari_ix_uav", "snw_pc_uyr", "slp_dg_uav", "cly_pc_uav",
    "Depth_avg", "Lake_area", "Res_time",
]
ERA5_PATTERNS = ["era5_t2m", "era5_tp", "era5_ssrd", "era5_d2m",
                  "era5_sp", "era5_sro", "era5_u10", "era5_v10", "era5_ro"]


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_metadata() -> pd.DataFrame:
    p = MULTIMODAL_DIR / "metadata.parquet"
    if not p.exists():
        logger.error(f"metadata.parquet not found at {p}. Run build_multimodal_dataset.py first.")
        return pd.DataFrame()
    return pd.read_parquet(p)


def _load_patches() -> np.ndarray:
    """Load patches.zarr → numpy (n_samples, 9, 12, 12) float32 channels-first."""
    import zarr
    p = MULTIMODAL_DIR / "patches.zarr"
    if not p.exists():
        logger.error(f"patches.zarr not found at {p}")
        return np.array([])
    z = zarr.open(str(p), mode="r")
    patches = z[:].astype(np.float32)   # (n_samples, 12, 12, 9) channels-last
    patches = np.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)

    # sro (ch7) and ro (ch8) were stored un-normalized in the zarr (raw ERA5 values).
    # Any channel with max > 10 is clipped to [p1, p99] and scaled to [-1, 1].
    for ci in range(patches.shape[-1]):
        ch = patches[:, :, :, ci]
        if float(ch.max()) <= 10.0:
            continue
        p1  = float(np.percentile(ch, 1))
        p99 = float(np.percentile(ch, 99))
        rng = p99 - p1
        if rng < 1e-8:
            patches[:, :, :, ci] = 0.0
        else:
            patches[:, :, :, ci] = np.clip((ch - p1) / rng, 0.0, 1.0) * 2.0 - 1.0

    return patches.transpose(0, 3, 1, 2).astype(np.float32)   # → (n, 9, 12, 12)


def _get_cv_groups(meta: pd.DataFrame) -> np.ndarray:
    if SPLIT_PATH.exists() and "fold_key" in meta.columns:
        return pd.Categorical(meta["fold_key"].fillna("unknown")).codes
    # Fallback: lat bins from atlas
    if ATLAS_PATH.exists():
        atlas = pd.read_parquet(ATLAS_PATH, columns=["station_id", "latitude"])
        atlas["station_id"] = atlas["station_id"].astype(str)
        meta = meta.copy()
        meta["station_id"] = meta["station_id"].astype(str)
        merged = meta[["station_id"]].merge(atlas, on="station_id", how="left")
        return merged["latitude"].fillna(60.0).apply(int).values
    return np.zeros(len(meta), dtype=int)


def _get_tabular_features(meta: pd.DataFrame) -> tuple:
    """Return (X_tab, feature_names) from metadata: HydroATLAS + ERA5 columns."""
    feat_cols = []
    # HydroATLAS static
    if ATLAS_PATH.exists():
        atlas = pd.read_parquet(ATLAS_PATH)
        atlas["station_id"] = atlas["station_id"].astype(str)
        ha_cols = [c for c in HYDROATLAS_FEATURES if c in atlas.columns]
        ha_to_add = [c for c in ha_cols if c not in meta.columns]
        if ha_to_add:
            meta = meta.merge(
                atlas[["station_id"] + ha_to_add].drop_duplicates("station_id"),
                on="station_id", how="left")
        feat_cols += ha_cols
    # ERA5 columns already in metadata
    era5_in_meta = [c for c in meta.columns
                    if any(c.startswith(p) for p in ERA5_PATTERNS)]
    feat_cols += era5_in_meta

    feat_cols = [c for c in feat_cols if c in meta.columns]
    X = meta[feat_cols].apply(
        lambda c: c.fillna(c.median() if c.notna().any() else 0.0)
    ).values.astype(np.float32)
    return X, feat_cols


# ── PyTorch models ────────────────────────────────────────────────────────────

def _build_cnn_encoder(n_channels: int = 9) -> "nn.Module":
    import torch.nn as nn
    return nn.Sequential(
        # Block 1: 12×12 → 10×10
        nn.Conv2d(n_channels, 32, kernel_size=3, padding=0),
        nn.BatchNorm2d(32), nn.ReLU(),
        # Block 2: 10×10 → 8×8
        nn.Conv2d(32, 64, kernel_size=3, padding=0),
        nn.BatchNorm2d(64), nn.ReLU(),
        # Block 3: 8×8 → 6×6
        nn.Conv2d(64, 128, kernel_size=3, padding=0),
        nn.BatchNorm2d(128), nn.ReLU(),
        # Pool to fixed 2×2
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),                   # 128 * 2 * 2 = 512
        nn.Linear(512, 128), nn.ReLU(),
    )


class PatchOnlyNet:
    """CNN on ERA5 patch only → WQ target regression."""

    def __init__(self, epochs=30, batch=256, lr=1e-3):
        self.epochs = epochs; self.batch = batch
        self.lr = lr

    def fit(self, patches, y):
        import torch, torch.nn as nn
        dev = torch.device(_get_device())
        enc = _build_cnn_encoder(patches.shape[1]).to(dev)
        head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
        params = list(enc.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss()

        Xp = torch.FloatTensor(patches).to(dev)
        yt = torch.FloatTensor(y).unsqueeze(1).to(dev)
        ds = torch.utils.data.TensorDataset(Xp, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch,
                                          shuffle=True, drop_last=True)

        for _ in range(self.epochs):
            enc.train(); head.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(head(enc(xb)), yb)
                loss.backward(); opt.step()
            sch.step()

        self._enc = enc; self._head = head; self._dev = dev

    def predict(self, patches):
        import torch
        self._enc.eval(); self._head.eval()
        with torch.no_grad():
            Xp = torch.FloatTensor(patches).to(self._dev)
            return self._head(self._enc(Xp)).squeeze(1).cpu().numpy()


class FusionNet:
    """
    Late fusion: CNN patch encoder + MLP tabular branch → shared regression head.
    Designed to show the additive value of spatial patches over tabular features.
    """

    def __init__(self, n_tab, epochs=40, batch=256, lr=1e-3):
        self.n_tab = n_tab; self.epochs = epochs
        self.batch = batch; self.lr = lr

    def fit(self, patches, X_tab, y):
        import torch, torch.nn as nn
        dev = torch.device(_get_device())

        patch_enc = _build_cnn_encoder(patches.shape[1]).to(dev)  # → 128
        tab_enc   = nn.Sequential(
            nn.Linear(self.n_tab, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
        ).to(dev)
        head = nn.Sequential(
            nn.Linear(128 + 64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to(dev)
        params = (list(patch_enc.parameters()) + list(tab_enc.parameters()) +
                  list(head.parameters()))
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss()

        Xp = torch.FloatTensor(patches).to(dev)
        Xt = torch.FloatTensor(X_tab).to(dev)
        yt = torch.FloatTensor(y).unsqueeze(1).to(dev)
        ds = torch.utils.data.TensorDataset(Xp, Xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch,
                                          shuffle=True, drop_last=True)

        for _ in range(self.epochs):
            patch_enc.train(); tab_enc.train(); head.train()
            for pb, tb, yb in dl:
                opt.zero_grad()
                fused = torch.cat([patch_enc(pb), tab_enc(tb)], dim=1)
                loss  = loss_fn(head(fused), yb)
                loss.backward(); opt.step()
            sch.step()

        self._patch_enc = patch_enc; self._tab_enc = tab_enc
        self._head = head; self._dev = dev

    def predict(self, patches, X_tab):
        import torch
        self._patch_enc.eval(); self._tab_enc.eval(); self._head.eval()
        with torch.no_grad():
            Xp = torch.FloatTensor(patches).to(self._dev)
            Xt = torch.FloatTensor(X_tab).to(self._dev)
            fused = torch.cat([self._patch_enc(Xp), self._tab_enc(Xt)], dim=1)
            return self._head(fused).squeeze(1).cpu().numpy()


# ── ViT ───────────────────────────────────────────────────────────────────────

class ViTNet:
    """
    Vision Transformer (Dosovitskiy et al. 2021) on ERA5 12×12×9 patches.

    patch_size=3 → 4×4 = 16 non-overlapping spatial tokens + 1 CLS token.
    Architecture: dim=128, depth=4, heads=4, ffn=256, pre-norm.
    Designed for 12×12 native resolution — no downsampling pooling layers.

    Input:  (B, 9, 12, 12)
    Output: (B, 1)  via CLS token head
    """

    def __init__(self, patch_size: int = 3, dim: int = 128,
                 depth: int = 4, heads: int = 4,
                 epochs: int = 40, batch: int = 256, lr: float = 1e-3):
        self.patch_size = patch_size
        self.dim        = dim
        self.depth      = depth
        self.heads      = heads
        self.epochs     = epochs
        self.batch      = batch
        self.lr         = lr

    def _build(self, n_channels: int, dev):
        import torch, torch.nn as nn
        p         = self.patch_size
        dim       = self.dim
        heads     = self.heads
        depth     = self.depth
        n_patches = (12 // p) ** 2          # 16 for p=3
        patch_dim = n_channels * p * p       # 81 for C=9, p=3

        class _ViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Linear(patch_dim, dim)
                self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
                self.pos_embed   = nn.Parameter(
                    torch.zeros(1, n_patches + 1, dim))
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=dim, nhead=heads,
                    dim_feedforward=dim * 2, dropout=0.1,
                    batch_first=True, norm_first=True)
                self.transformer = nn.TransformerEncoder(
                    enc_layer, num_layers=depth)
                self.norm = nn.LayerNorm(dim)
                self.head = nn.Sequential(
                    nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

            def forward(self, x):
                B, C, H, W = x.shape
                # Extract non-overlapping p×p patches
                x = x.unfold(2, p, p).unfold(3, p, p)
                # (B, C, H//p, W//p, p, p)
                x = x.contiguous().view(B, C, -1, p * p)
                # (B, C, n_patches, p²)
                x = x.permute(0, 2, 1, 3).reshape(B, n_patches, -1)
                # (B, n_patches, patch_dim)
                x   = self.patch_embed(x)
                cls = self.cls_token.expand(B, -1, -1)
                x   = torch.cat([cls, x], dim=1) + self.pos_embed
                x   = self.norm(self.transformer(x))
                return self.head(x[:, 0])          # CLS token

        return _ViT().to(dev)

    def fit(self, patches: np.ndarray, y: np.ndarray):
        import torch, torch.nn as nn
        dev   = torch.device(_get_device())
        model = self._build(patches.shape[1], dev)
        opt   = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=1e-4)
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss()

        Xp = torch.FloatTensor(patches).to(dev)
        yt = torch.FloatTensor(y).unsqueeze(1).to(dev)
        ds = torch.utils.data.TensorDataset(Xp, yt)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch, shuffle=True, drop_last=True)

        best_loss, patience, wait = 1e9, 5, 0
        for _ in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        self._model = model
        self._dev   = dev

    def predict(self, patches: np.ndarray) -> np.ndarray:
        import torch
        self._model.eval()
        with torch.no_grad():
            Xp = torch.FloatTensor(patches).to(self._dev)
            return self._model(Xp).squeeze(1).cpu().numpy()


# ── Evaluation ────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred):
    ok = ~np.isnan(y_pred)
    if ok.sum() == 0:
        return {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan"), "n": 0}
    y0 = np.expm1(y_true[ok]); p0 = np.expm1(y_pred[ok])
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y0, p0))), 3),
        "r2":   round(float(r2_score(y0, p0)), 3),
        "mae":  round(float(mean_absolute_error(y0, p0)), 3),
        "n":    int(ok.sum()),
    }


TROPHIC_BINS   = [0, 10, 35, 100, np.inf]
TROPHIC_LABELS = ["oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic"]
CONF_LEVELS    = (0.80, 0.90, 0.95)


def _trophic_series(meta: pd.DataFrame, global_idx: np.ndarray) -> np.ndarray:
    if "tp_ug_l" not in meta.columns:
        return np.full(len(global_idx), "unknown", dtype=object)
    tp = meta["tp_ug_l"].iloc[global_idx].clip(lower=0)
    return pd.cut(tp, bins=TROPHIC_BINS,
                  labels=TROPHIC_LABELS).astype(str).values


def _rolling_origin(model_name: str, target: str,
                    meta_sub: pd.DataFrame,
                    fit_fn, pred_fn,
                    y_log: np.ndarray) -> list:
    """Expanding-window temporal CV: train up to cutoff, test in next period."""
    rows  = []
    dates = pd.to_datetime(meta_sub["date"].values)
    for period, train_end, test_start, test_end in ROLLING_WINDOWS:
        tr = np.where(dates <= train_end)[0]
        va = np.where((dates >= test_start) & (dates <= test_end))[0]
        if len(tr) < 50 or len(va) < 10:
            continue
        try:
            m = fit_fn(tr)
            p = pred_fn(m, va)
            rows.append({"model": model_name, "target": target,
                         "test_period": period, **_metrics(y_log[va], p)})
            logger.info("    rolling %s|%s|%s R²=%.3f",
                        model_name, target, period, rows[-1]["r2"])
        except Exception as e:
            logger.warning("rolling %s|%s|%s: %s", model_name, target, period, e)
    return rows


def _patch_conformal(model_name: str, target: str,
                     cal_data: list) -> list:
    rows = []
    for lv in CONF_LEVELS:
        alpha = 1 - lv
        picps, widths = [], []
        for cal_y, cal_p, val_y, val_p, _va in cal_data:
            scores = np.abs(cal_y - cal_p)
            n = len(scores)
            if n < 5:
                continue
            q = float(np.quantile(
                scores,
                min(1.0, np.ceil((n + 1) * (1 - alpha)) / n),
                method="higher"))
            picps.append(float(np.mean(
                (val_y >= val_p - q) & (val_y <= val_p + q))))
            widths.append(2.0 * q)
        if not picps:
            continue
        picp = float(np.mean(picps))
        rows.append({
            "model": model_name, "target": target,
            "nominal_coverage":        lv,
            "picp":                    round(picp, 3),
            "ece":                     round(abs(picp - lv), 4),
            "mean_interval_width_log": round(float(np.mean(widths)), 4),
            "n_folds":                 len(picps),
        })
    return rows


def _patch_trophic(model_name: str, target: str,
                   y_log: np.ndarray, preds: np.ndarray,
                   tc: np.ndarray) -> list:
    rows = []
    for cls in TROPHIC_LABELS:
        mask = (tc == cls) & np.isfinite(preds) & np.isfinite(y_log)
        if mask.sum() < 5:
            continue
        rows.append({"model": model_name, "target": target,
                     "trophic_class": cls,
                     **_metrics(y_log[mask], preds[mask])})
    return rows


def _patch_quantile(model_name: str, target: str,
                    cal_data: list, tc_val: np.ndarray) -> list:
    rows = []
    val_y_all  = np.concatenate([f[2] for f in cal_data])
    val_p_all  = np.concatenate([f[3] for f in cal_data])
    val_va_all = np.concatenate([f[4] for f in cal_data])
    val_tc     = tc_val[np.arange(len(val_va_all))]

    for lv in CONF_LEVELS:
        alpha = 1 - lv
        qs = []
        for cal_y, cal_p, _vy, _vp, _va in cal_data:
            scores = np.abs(cal_y - cal_p)
            n = len(scores)
            if n < 5:
                continue
            qs.append(float(np.quantile(
                scores,
                min(1.0, np.ceil((n + 1) * (1 - alpha)) / n),
                method="higher")))
        if not qs:
            continue
        q   = float(np.mean(qs))
        lo  = val_p_all - q;  hi = val_p_all + q
        for cls in (["all"] + list(TROPHIC_LABELS)):
            mask = np.ones(len(val_y_all), bool) if cls == "all" \
                   else (val_tc == cls)
            if mask.sum() < 5:
                continue
            y_tc = val_y_all[mask]
            picp = float(np.mean((y_tc >= lo[mask]) & (y_tc <= hi[mask])))
            pinaw = float(np.mean(hi[mask] - lo[mask]))
            width = hi[mask] - lo[mask]
            pen   = (2 / alpha) * (np.maximum(0, lo[mask] - y_tc) +
                                    np.maximum(0, y_tc - hi[mask]))
            rows.append({
                "model": model_name, "target": target,
                "trophic_class":    cls,
                "nominal_coverage": lv,
                "picp":             round(picp, 3),
                "ece":              round(abs(picp - lv), 4),
                "pinaw":            round(pinaw, 4),
                "winkler_score":    round(float(np.mean(width + pen)), 3),
                "n":                int(mask.sum()),
            })
    return rows


def _save_patch(rows: list, filename: str) -> None:
    if not rows:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / filename, index=False)
    logger.info("Checkpoint → %s (%d rows)", filename, len(rows))


def _get_device() -> str:
    """Return 'cuda' only if CUDA is available AND kernels actually execute on this GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        # is_available() can return True while compiled kernels miss the SM version
        t = torch.zeros(1, device="cuda")
        _ = (t + 1).item()
        return "cuda"
    except Exception:
        return "cpu"


def run(requested_models: list = None) -> None:
    try:
        import torch
        torch.backends.cudnn.benchmark = True
    except ImportError:
        logger.error("PyTorch required for patch baselines. Install: pip install torch")
        return

    if requested_models is None:
        requested_models = ["patch_xgb", "cnn", "fusion", "vit"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resume: load existing checkpoints ────────────────────────────────────
    def _load_ckpt(fname):
        p = OUT_DIR / fname
        return pd.read_csv(p).to_dict("records") if p.exists() else []

    results       = _load_ckpt("patch_baseline_results.csv")
    conf_rows     = _load_ckpt("patch_conformal_results.csv")
    trophic_rows  = _load_ckpt("patch_trophic_results.csv")
    quantile_rows = _load_ckpt("patch_quantile_coverage.csv")

    # completed_main: main R²/RMSE/MAE results already saved
    # completed: conformal data already saved — this governs re-runs,
    # so models that ran before conformal logic was added get re-evaluated.
    completed_main = {(r["model"], r["target"]) for r in results}
    completed = {(r["model"], r["target"]) for r in conf_rows}
    if completed:
        logger.info("Resuming — conformal already done: %s", sorted(completed))
    missing_conf = completed_main - completed
    if missing_conf:
        logger.info("Re-running for missing conformal data: %s", sorted(missing_conf))

    rolling_rows     = _load_ckpt("patch_rolling_origin.csv")
    completed_rolling = {(r["model"], r["target"]) for r in rolling_rows}
    missing_rolling   = completed_main - completed_rolling
    if missing_rolling:
        logger.info("Rolling origin pending: %s", sorted(missing_rolling))

    # ── Data loading ──────────────────────────────────────────────────────────
    logger.info("Loading multimodal dataset")
    meta = _load_metadata()
    if meta.empty:
        return

    logger.info("Loading ERA5 patches")
    patches = _load_patches()
    if patches.size == 0:
        return

    assert len(meta) == len(patches), \
        f"metadata rows ({len(meta)}) ≠ patch count ({len(patches)})"

    logger.info("Building tabular feature matrix")
    X_tab, tab_cols = _get_tabular_features(meta)
    scaler_tab = StandardScaler()

    logger.info("Building CV groups")
    groups = _get_cv_groups(meta)

    # MODEL_KEY maps CLI name → display name used in CSV
    MODEL_KEY = {
        "patch_xgb": "Patch-XGBoost",
        "cnn":       "CNN-patch",
        "fusion":    "CNN+Tabular",
        "vit":       "ViT",
    }

    for target in TARGETS:
        if target not in meta.columns:
            logger.warning(f"{target} not in metadata — skipping")
            continue

        valid_mask = meta[target].notna() & (meta[target] > 0)
        idx   = np.where(valid_mask)[0]
        y_log = np.log1p(meta.loc[valid_mask, target].values.astype(np.float32))
        P     = patches[idx]
        T     = X_tab[idx]
        grp   = groups[idx]
        cv    = GroupKFold(n_splits=max(2, min(5, len(np.unique(grp)))))
        logger.info(f"\n── {target}  ({len(idx):,} valid samples) ──")

        # ── Patch-XGB: flatten patch → XGBoost ───────────────────────────────
        mname = "Patch-XGBoost"
        if "patch_xgb" in requested_models:
            if (mname, target) in completed:
                logger.info(f"  {mname} {target} — skipping (checkpoint)")
            else:
                try:
                    from xgboost import XGBRegressor
                    P_flat   = P.reshape(len(P), -1)
                    preds    = np.full(len(y_log), np.nan)
                    cal_data = []
                    for tr, va in cv.split(P_flat, y_log, grp):
                        if len(tr) == 0 or len(va) == 0:
                            continue
                        n_cal   = max(1, int(0.2 * len(tr)))
                        fit_idx = tr[:-n_cal];  cal_idx = tr[-n_cal:]
                        m = XGBRegressor(n_estimators=300, max_depth=6,
                                         learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, tree_method="hist",
                                         device="cpu", verbosity=0, random_state=42)
                        m.fit(P_flat[fit_idx], y_log[fit_idx])
                        cal_data.append((y_log[cal_idx], m.predict(P_flat[cal_idx]),
                                         y_log[va],      m.predict(P_flat[va]), va))
                        preds[va] = cal_data[-1][3]
                    rec = {"target": target, "model": mname, **_metrics(y_log, preds)}
                    tc  = _trophic_series(meta, idx[np.concatenate([f[4] for f in cal_data])])
                    if (mname, target) not in completed_main:
                        results.append(rec)
                    conf_rows     += _patch_conformal(mname, target, cal_data)
                    trophic_rows  += _patch_trophic(mname, target, y_log, preds,
                                                    _trophic_series(meta, idx))
                    quantile_rows += _patch_quantile(mname, target, cal_data, tc)
                    completed.add((mname, target))
                    logger.info(f"  {mname}: RMSE={rec['rmse']:.3f}  R²={rec['r2']:.3f}")
                    _save_patch(results,       "patch_baseline_results.csv")
                    _save_patch(conf_rows,     "patch_conformal_results.csv")
                    _save_patch(trophic_rows,  "patch_trophic_results.csv")
                    _save_patch(quantile_rows, "patch_quantile_coverage.csv")
                except Exception as e:
                    logger.warning(f"  {mname} failed: {e}")

        # Rolling origin for Patch-XGB
        if "patch_xgb" in requested_models and (mname, target) not in completed_rolling:
            try:
                from xgboost import XGBRegressor
                _Pf = P.reshape(len(P), -1); _y = y_log
                meta_v = meta[valid_mask].reset_index(drop=True)
                def _fit_xro(tr, _Pf=_Pf, _y=_y):
                    m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     tree_method="hist", device="cpu",
                                     verbosity=0, random_state=42)
                    m.fit(_Pf[tr], _y[tr]); return m
                def _pred_xro(m, va, _Pf=_Pf): return m.predict(_Pf[va])
                rolling_rows += _rolling_origin(mname, target, meta_v,
                                                _fit_xro, _pred_xro, _y)
                completed_rolling.add((mname, target))
                _save_patch(rolling_rows, "patch_rolling_origin.csv")
            except Exception as e:
                logger.warning(f"  {mname} rolling failed: {e}")

        # ── CNN: patch only ───────────────────────────────────────────────────
        mname = "CNN-patch"
        if "cnn" in requested_models:
            if (mname, target) in completed:
                logger.info(f"  {mname} {target} — skipping (checkpoint)")
            else:
                try:
                    preds    = np.full(len(y_log), np.nan)
                    cal_data = []
                    for tr, va in cv.split(P, y_log, grp):
                        if len(tr) == 0 or len(va) == 0:
                            continue
                        n_cal   = max(1, int(0.2 * len(tr)))
                        fit_idx = tr[:-n_cal];  cal_idx = tr[-n_cal:]
                        m = PatchOnlyNet()
                        m.fit(P[fit_idx], y_log[fit_idx])
                        cal_data.append((y_log[cal_idx], m.predict(P[cal_idx]),
                                         y_log[va],      m.predict(P[va]), va))
                        preds[va] = cal_data[-1][3]
                    rec = {"target": target, "model": mname, **_metrics(y_log, preds)}
                    tc  = _trophic_series(meta, idx[np.concatenate([f[4] for f in cal_data])])
                    if (mname, target) not in completed_main:
                        results.append(rec)
                    conf_rows     += _patch_conformal(mname, target, cal_data)
                    trophic_rows  += _patch_trophic(mname, target, y_log, preds,
                                                    _trophic_series(meta, idx))
                    quantile_rows += _patch_quantile(mname, target, cal_data, tc)
                    completed.add((mname, target))
                    logger.info(f"  {mname}: RMSE={rec['rmse']:.3f}  R²={rec['r2']:.3f}")
                    _save_patch(results,       "patch_baseline_results.csv")
                    _save_patch(conf_rows,     "patch_conformal_results.csv")
                    _save_patch(trophic_rows,  "patch_trophic_results.csv")
                    _save_patch(quantile_rows, "patch_quantile_coverage.csv")
                except Exception as e:
                    logger.warning(f"  {mname} failed: {e}")

        # Rolling origin for CNN-patch
        if "cnn" in requested_models and (mname, target) not in completed_rolling:
            try:
                _P = P; _y = y_log
                meta_v = meta[valid_mask].reset_index(drop=True)
                def _fit_cro(tr, _P=_P, _y=_y):
                    m = PatchOnlyNet(); m.fit(_P[tr], _y[tr]); return m
                def _pred_cro(m, va, _P=_P): return m.predict(_P[va])
                rolling_rows += _rolling_origin(mname, target, meta_v,
                                                _fit_cro, _pred_cro, _y)
                completed_rolling.add((mname, target))
                _save_patch(rolling_rows, "patch_rolling_origin.csv")
            except Exception as e:
                logger.warning(f"  {mname} rolling failed: {e}")

        # ── Fusion: CNN patch + tabular ───────────────────────────────────────
        mname = "CNN+Tabular"
        if "fusion" in requested_models:
            if (mname, target) in completed:
                logger.info(f"  {mname} {target} — skipping (checkpoint)")
            else:
                try:
                    preds    = np.full(len(y_log), np.nan)
                    cal_data = []
                    n_tab    = T.shape[1]
                    for tr, va in cv.split(P, y_log, grp):
                        if len(tr) == 0 or len(va) == 0:
                            continue
                        n_cal   = max(1, int(0.2 * len(tr)))
                        fit_idx = tr[:-n_cal];  cal_idx = tr[-n_cal:]
                        T_fit = scaler_tab.fit_transform(T[fit_idx])
                        T_cal = scaler_tab.transform(T[cal_idx])
                        T_va  = scaler_tab.transform(T[va])
                        m = FusionNet(n_tab=n_tab)
                        m.fit(P[fit_idx], T_fit, y_log[fit_idx])
                        cal_data.append((y_log[cal_idx], m.predict(P[cal_idx], T_cal),
                                         y_log[va],      m.predict(P[va], T_va), va))
                        preds[va] = cal_data[-1][3]
                    rec = {"target": target, "model": mname, **_metrics(y_log, preds)}
                    tc  = _trophic_series(meta, idx[np.concatenate([f[4] for f in cal_data])])
                    if (mname, target) not in completed_main:
                        results.append(rec)
                    conf_rows     += _patch_conformal(mname, target, cal_data)
                    trophic_rows  += _patch_trophic(mname, target, y_log, preds,
                                                    _trophic_series(meta, idx))
                    quantile_rows += _patch_quantile(mname, target, cal_data, tc)
                    completed.add((mname, target))
                    logger.info(f"  {mname}: RMSE={rec['rmse']:.3f}  R²={rec['r2']:.3f}")
                    _save_patch(results,       "patch_baseline_results.csv")
                    _save_patch(conf_rows,     "patch_conformal_results.csv")
                    _save_patch(trophic_rows,  "patch_trophic_results.csv")
                    _save_patch(quantile_rows, "patch_quantile_coverage.csv")
                except Exception as e:
                    logger.warning(f"  {mname} failed: {e}")

        # Rolling origin for CNN+Tabular
        if "fusion" in requested_models and (mname, target) not in completed_rolling:
            try:
                _P = P; _T = T; _y = y_log; _n = T.shape[1]
                meta_v = meta[valid_mask].reset_index(drop=True)
                def _fit_fro(tr, _P=_P, _T=_T, _y=_y, _n=_n):
                    sc = StandardScaler()
                    m = FusionNet(n_tab=_n)
                    m.fit(_P[tr], sc.fit_transform(_T[tr]), _y[tr])
                    return m, sc
                def _pred_fro(bundle, va, _P=_P, _T=_T):
                    m, sc = bundle; return m.predict(_P[va], sc.transform(_T[va]))
                rolling_rows += _rolling_origin(mname, target, meta_v,
                                                _fit_fro, _pred_fro, _y)
                completed_rolling.add((mname, target))
                _save_patch(rolling_rows, "patch_rolling_origin.csv")
            except Exception as e:
                logger.warning(f"  {mname} rolling failed: {e}")

        # ── ViT: Vision Transformer on 12×12×9 patch ─────────────────────────
        mname = "ViT"
        if "vit" in requested_models:
            if (mname, target) in completed:
                logger.info(f"  {mname} {target} — skipping (checkpoint)")
            else:
                try:
                    preds    = np.full(len(y_log), np.nan)
                    cal_data = []
                    for tr, va in cv.split(P, y_log, grp):
                        if len(tr) == 0 or len(va) == 0:
                            continue
                        n_cal   = max(1, int(0.2 * len(tr)))
                        fit_idx = tr[:-n_cal];  cal_idx = tr[-n_cal:]
                        m = ViTNet()
                        m.fit(P[fit_idx], y_log[fit_idx])
                        cal_data.append((y_log[cal_idx], m.predict(P[cal_idx]),
                                         y_log[va],      m.predict(P[va]), va))
                        preds[va] = cal_data[-1][3]
                    rec = {"target": target, "model": mname, **_metrics(y_log, preds)}
                    tc  = _trophic_series(meta, idx[np.concatenate([f[4] for f in cal_data])])
                    if (mname, target) not in completed_main:
                        results.append(rec)
                    conf_rows     += _patch_conformal(mname, target, cal_data)
                    trophic_rows  += _patch_trophic(mname, target, y_log, preds,
                                                    _trophic_series(meta, idx))
                    quantile_rows += _patch_quantile(mname, target, cal_data, tc)
                    completed.add((mname, target))
                    logger.info(f"  {mname}: RMSE={rec['rmse']:.3f}  R²={rec['r2']:.3f}")
                    _save_patch(results,       "patch_baseline_results.csv")
                    _save_patch(conf_rows,     "patch_conformal_results.csv")
                    _save_patch(trophic_rows,  "patch_trophic_results.csv")
                    _save_patch(quantile_rows, "patch_quantile_coverage.csv")
                except Exception as e:
                    logger.warning(f"  {mname} failed: {e}")

        # Rolling origin for ViT
        if "vit" in requested_models and (mname, target) not in completed_rolling:
            try:
                _P = P; _y = y_log
                meta_v = meta[valid_mask].reset_index(drop=True)
                def _fit_vro(tr, _P=_P, _y=_y):
                    m = ViTNet(); m.fit(_P[tr], _y[tr]); return m
                def _pred_vro(m, va, _P=_P): return m.predict(_P[va])
                rolling_rows += _rolling_origin(mname, target, meta_v,
                                                _fit_vro, _pred_vro, _y)
                completed_rolling.add((mname, target))
                _save_patch(rolling_rows, "patch_rolling_origin.csv")
            except Exception as e:
                logger.warning(f"  {mname} rolling failed: {e}")

    if not results:
        logger.error("No results produced")
        return

    logger.info("\nFinal save complete. Results summary:")
    logger.info("\n" + pd.DataFrame(results).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Spatial patch baselines for NordWQ-26")
    parser.add_argument("--models", nargs="+",
                        default=["patch_xgb", "cnn", "fusion", "vit"],
                        choices=["patch_xgb", "cnn", "fusion", "vit"])
    args = parser.parse_args()
    run(args.models)


if __name__ == "__main__":
    main()
