#!/usr/bin/env python3
"""
Spatio-temporal baseline models for NordWQ-26 (CIKM 2026).

Five models spanning three architectural families:

  Patch-sequence models (input: T × 9 × 12 × 12 ERA5 spatial patches):
  -----------------------------------------------------------------------
  1. ConvLSTM        — Convolutional LSTM (Shi et al. 2015).
                       Spatial convolutions within LSTM gates; the 12×12
                       spatial structure is preserved throughout the full
                       temporal sequence.  Contrast with CNN-Transformer
                       where spatial dims are collapsed before temporal
                       encoding.

  2. CNN-Transformer — CNN encoder applied per timestep (collapses 12×12
                       to an embedding vector) followed by a Transformer
                       encoder over the T-step sequence.  Temporal
                       attention, not spatial.

  3. TimeSformer     — Divided space-time attention (Bertasius et al.
                       2021).  Temporal attention across patch tokens of
                       the SAME spatial position, then spatial attention
                       within each frame.  Factorised attention reduces
                       O(T·H·W) to O(T + H·W).

  Graph-based models (input: T × N × C station-level ERA5 + k-NN graph):
  -----------------------------------------------------------------------
  4. STGCN           — Spatio-Temporal Graph CNN (Yu et al. 2018).
                       Chebyshev graph convolution (spatial) interleaved
                       with 1-D temporal convolution over a k-NN station
                       neighbourhood graph.

  5. DCRNN           — Diffusion Convolutional RNN (Li et al. 2018).
                       Bidirectional random-walk diffusion convolution
                       inside GRU gates; tested on the same k-NN graph.

Data:
  Patch models : temporal sequences of ERA5 patches loaded from
                 data/processed/nordic_multimodal_dataset/patches.zarr
                 (single-timestep; each sample replicated T times as a
                 static-patch fallback when era5_wq.h5 is unavailable)
                 and era5_sequences_wW.npy for temporal variation per
                 channel.  Full temporal patch sequences require
                 era5_wq.h5 (256 GB; journal version).
  Graph models : station-level ERA5 sequences (T × 7) + geographic k-NN
                 adjacency built from HydroATLAS lat/lon.

CV: GroupKFold(k=5) on fold_key — identical to all evaluation scripts.
Targets: chla_ug_l, tp_ug_l, secchi_m (log1p-transformed).

Outputs:
  data/outputs/spatiotemporal_baseline_results.csv
  data/outputs/spatiotemporal_baseline_results.tex

Usage:
    python -m src.evaluation.spatiotemporal_baselines
    python -m src.evaluation.spatiotemporal_baselines --models convlstm cnn_transformer
    python -m src.evaluation.spatiotemporal_baselines --window 30 --k_neighbors 8
    python -m src.evaluation.spatiotemporal_baselines --window 90
"""

import argparse
import logging
import math
import os
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

MULTIMODAL_DIR = Path("data/processed/nordic_multimodal_dataset")
ERA5_RAW_DIR   = Path("data/raw/era5")
ATLAS_PATH     = Path("data/raw/hydroatlas/hydroatlas_station_covariates_SE.parquet")
FEATURE_TABLE_DIR = Path("data/interim/SE/feature_table")
OUT_DIR        = Path("data/outputs")

TARGETS       = ["chla_ug_l", "tp_ug_l", "secchi_m"]
ERA5_SEQ_COLS = ["t2m_mean_c", "t2m_min_c", "t2m_max_c",
                 "tp_mm", "sde_m", "lmlt_c", "tcc_frac"]
N_ERA5        = len(ERA5_SEQ_COLS)   # 7
PATCH_C       = 9                    # ERA5 patch channels
PATCH_H = PATCH_W = 12              # spatial grid size
MAX_TRAIN_SAMPLES = 80_000          # cap per fold to bound training time


# ── Device ────────────────────────────────────────────────────────────────────

def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ok = np.isfinite(y_pred) & np.isfinite(y_true)
    if ok.sum() < 5:
        return {"rmse": float("nan"), "r2": float("nan"),
                "mae": float("nan"), "n": 0}
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
        raise FileNotFoundError(
            f"{p} not found — run build_multimodal_dataset.py first")
    meta = pd.read_parquet(p)
    meta["station_id"] = meta["station_id"].astype(str)
    meta["date"]       = pd.to_datetime(meta["date"])
    return meta


def _load_patches() -> np.ndarray:
    """Load patches.zarr → (n_samples, 9, 12, 12) float32, channels-first."""
    import zarr
    p = MULTIMODAL_DIR / "patches.zarr"
    if not p.exists():
        logger.warning("patches.zarr not found — patch-sequence models disabled")
        return np.array([])
    z = zarr.open(str(p), mode="r")
    patches = z[:].astype(np.float32)          # (n, 12, 12, 9) channels-last
    patches = np.nan_to_num(patches, nan=0.0)
    for ci in range(patches.shape[-1]):
        ch = patches[:, :, :, ci]
        if float(ch.max()) <= 10.0:
            continue
        p1, p99 = float(np.percentile(ch, 1)), float(np.percentile(ch, 99))
        rng = p99 - p1
        patches[:, :, :, ci] = (
            np.clip((ch - p1) / rng, 0.0, 1.0) * 2.0 - 1.0
            if rng > 1e-8 else np.zeros_like(ch)
        )
    return patches.transpose(0, 3, 1, 2).astype(np.float32)  # (n,9,12,12)


def _load_era5_station_ts() -> dict:
    """Return {station_id: DataFrame(index=date, cols=ERA5_SEQ_COLS)}."""
    files = sorted(ERA5_RAW_DIR.glob("era5_sweden_*.parquet"))
    if not files:
        return {}
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["station_id", "date"] + ERA5_SEQ_COLS)
        df["station_id"] = df["station_id"].astype(str)
        df["date"]       = pd.to_datetime(df["date"])
        frames.append(df)
    era5 = pd.concat(frames, ignore_index=True).sort_values(["station_id", "date"])
    out = {}
    for sid, grp in era5.groupby("station_id", sort=False):
        ts = grp.set_index("date")[ERA5_SEQ_COLS].astype("float32")
        ts = ts.ffill(limit=7).fillna(0.0)
        out[str(sid)] = ts
    logger.info("ERA5 station time-series loaded: %d stations", len(out))
    return out


def _build_sequences(meta: pd.DataFrame, station_ts: dict,
                     window: int) -> tuple:
    """
    Build (n_samples, window, 7) ERA5 station-level sequences.
    Returns (sequences, mask) where mask[i] is True when station has ERA5 data.
    Uses cached .npy files if available.
    """
    cache   = MULTIMODAL_DIR / f"era5_sequences_w{window}.npy"
    mcache  = MULTIMODAL_DIR / f"era5_seq_mask_w{window}.npy"
    if cache.exists() and mcache.exists():
        logger.info("Loading ERA5 sequences from cache (window=%d)", window)
        return np.load(str(cache)), np.load(str(mcache))

    n   = len(meta)
    seq = np.zeros((n, window, N_ERA5), dtype=np.float32)
    msk = np.zeros(n, dtype=bool)
    logger.info("Building ERA5 %d-day sequences for %d samples…", window, n)
    for i, row in enumerate(meta.itertuples(index=False)):
        sid  = str(row.station_id)
        date = pd.Timestamp(row.date)
        if sid not in station_ts:
            continue
        ts  = station_ts[sid]
        end = date
        start = end - pd.Timedelta(days=window)
        win = ts.loc[start: end - pd.Timedelta(days=1)]
        if len(win) == 0:
            continue
        n_d = min(len(win), window)
        seq[i, -n_d:] = win.values[-n_d:].astype("float32")
        msk[i] = True
        if (i + 1) % 100_000 == 0:
            logger.info("  %d / %d", i + 1, n)
    np.save(str(cache), seq)
    np.save(str(mcache), msk)
    return seq, msk


ERA5_H5_PATH = Path("data/processed/patches_hdf5/era5_wq.h5")


def _build_patch_sequences(meta: pd.DataFrame, patches: np.ndarray,
                            station_ts: dict, window: int) -> np.ndarray:
    """
    Build (n_samples, window, 9, 12, 12) temporal patch sequences.

    Primary path  : era5_wq.h5 — direct T-day window lookup per sample.
    Fallback path : replicate the single observation-date static patch across
                    all T timesteps, modulated by ERA5 station scalars.
                    Produces near-identical frames (no real temporal signal).
    """
    cache = MULTIMODAL_DIR / f"patch_sequences_w{window}.npy"
    if cache.exists():
        logger.info("Loading patch sequences from cache (window=%d)", window)
        return np.load(str(cache), mmap_mode="r")

    n   = len(meta)
    seq = np.zeros((n, window, PATCH_C, PATCH_H, PATCH_W), dtype=np.float32)

    # ── Primary: era5_wq.h5 ──────────────────────────────────────────────────
    if ERA5_H5_PATH.exists():
        import h5py
        logger.info("Building patch sequences from era5_wq.h5 (window=%d)…",
                    window)
        with h5py.File(str(ERA5_H5_PATH), "r") as h5:
            # Keys: station_ids (18743,), dates (9497,) bytes, patch (18743,9497,12,12,9)
            station_index = {str(s): i for i, s in
                             enumerate(h5["station_ids"][:].astype(str))}
            dates_dt = pd.to_datetime(
                [d.decode() for d in h5["dates"][:]])  # DatetimeIndex (9497,)
            # MD-5: channel 4 = ssrd overflows float16 → inf; zero it out
            ZERO_CHANNELS = [4]   # ssrd

            for i, row in enumerate(meta.itertuples(index=False)):
                sid  = str(row.station_id)
                date = pd.Timestamp(row.date)
                if sid not in station_index:
                    continue
                si = station_index[sid]

                # Window: T days ending strictly before obs date
                end_idx   = int(np.searchsorted(dates_dt, date, side="left"))
                start_idx = max(0, end_idx - window)
                avail     = end_idx - start_idx
                if avail == 0:
                    continue

                # patch: (avail, H, W, C) float16 → float32, channels-first
                chunk = h5["patch"][si, start_idx:end_idx]   # (avail,12,12,9)
                chunk = np.nan_to_num(chunk.astype(np.float32),
                                      nan=0.0, posinf=0.0, neginf=0.0)
                for ci in ZERO_CHANNELS:
                    chunk[:, :, :, ci] = 0.0  # zero overflowed channel
                # Apply validity mask
                msk = h5["mask"][si, start_idx:end_idx]      # (avail,12,12)
                chunk[msk == 0] = 0.0
                chunk = chunk.transpose(0, 3, 1, 2)          # (avail,9,12,12)
                seq[i, -avail:] = chunk

                if (i + 1) % 100_000 == 0:
                    logger.info("  %d / %d", i + 1, n)

        np.save(str(cache), seq)
        logger.info("Patch sequences saved → %s", cache)
        return seq

    # ── Fallback: static-patch replication (no real temporal signal) ─────────
    logger.warning(
        "era5_wq.h5 not found — using static-patch replication fallback. "
        "Results will show near-identical R² across architectures.")
    scaler      = StandardScaler()
    all_scalars = np.zeros((n, window, N_ERA5), dtype=np.float32)
    for i, row in enumerate(meta.itertuples(index=False)):
        sid  = str(row.station_id)
        date = pd.Timestamp(row.date)
        if sid not in station_ts:
            continue
        ts    = station_ts[sid]
        end   = date
        start = end - pd.Timedelta(days=window)
        win   = ts.loc[start: end - pd.Timedelta(days=1)]
        if len(win) == 0:
            continue
        n_d = min(len(win), window)
        all_scalars[i, -n_d:] = win.values[-n_d:].astype("float32")

    flat        = all_scalars.reshape(-1, N_ERA5)
    flat        = scaler.fit_transform(flat).astype(np.float32)
    all_scalars = flat.reshape(n, window, N_ERA5)

    logger.info("Building static-patch sequences (window=%d, n=%d)…", window, n)
    for i in range(n):
        if len(patches) <= i or np.all(patches[i] == 0):
            continue
        base = patches[i]
        for t in range(window):
            frame = base.copy()
            for c in range(PATCH_C):
                sc_idx   = min(c, N_ERA5 - 1)
                frame[c] = base[c] * (1.0 + 0.2 * float(all_scalars[i, t, sc_idx]))
            seq[i, t] = frame
        if (i + 1) % 50_000 == 0:
            logger.info("  %d / %d patch sequences built", i + 1, n)

    np.save(str(cache), seq)
    logger.info("Patch sequences saved → %s", cache)
    return seq


def _build_station_graph(meta: pd.DataFrame,
                          k: int = 8) -> tuple:
    """
    Build a k-NN geographic adjacency for all stations in meta.

    Returns:
      station_ids : list of station_id strings (node index → station_id)
      A           : (N, N) float32 normalised adjacency matrix
      sid_to_idx  : dict station_id → node index
    """
    # Load coordinates from feature tables (atlas parquet has no lat/lon columns)
    ft_files = sorted(FEATURE_TABLE_DIR.glob("feature_table_SE_*.parquet"))
    if not ft_files:
        raise FileNotFoundError(
            f"No feature_table_SE_*.parquet found in {FEATURE_TABLE_DIR}")
    _frames = [pd.read_parquet(f, columns=["station_id", "latitude", "longitude"])
               for f in ft_files]
    all_coords = (pd.concat(_frames, ignore_index=True)
                  .drop_duplicates("station_id"))
    all_coords["station_id"] = all_coords["station_id"].astype(str)
    meta_sids = meta["station_id"].unique().tolist()
    coords = (all_coords[all_coords["station_id"].isin(meta_sids)]
              .set_index("station_id"))
    station_ids = [s for s in meta_sids if s in coords.index]
    N = len(station_ids)
    lat = np.array([coords.loc[s, "latitude"]  for s in station_ids])
    lon = np.array([coords.loc[s, "longitude"] for s in station_ids])

    # Haversine distance matrix (km)
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2)**2
    dist = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # k-NN adjacency (symmetric)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        nn = np.argsort(dist[i])[1: k + 1]
        A[i, nn] = 1.0
        A[nn, i] = 1.0

    # Row-normalise
    deg = A.sum(axis=1, keepdims=True).clip(min=1)
    A = (A / deg).astype(np.float32)
    sid_to_idx = {s: i for i, s in enumerate(station_ids)}
    logger.info("Station graph: N=%d nodes, k=%d neighbours", N, k)
    return station_ids, A, sid_to_idx


def _build_graph_sequences(meta: pd.DataFrame, station_ts: dict,
                            station_ids: list, A: np.ndarray,
                            window: int, k: int = 8) -> tuple:
    """
    Build per-sample local-neighbourhood sequences for STGCN / DCRNN.

    Each sample i stores only the centre station + k nearest neighbours:
      shape (n_samples, window, k+1, N_ERA5)  — node 0 = centre station.

    Written to a memmap so the array is never fully resident in RAM.

    Returns:
      graph_seq  : (n_samples, window, k+1, N_ERA5) float32  (mmap)
      node_mask  : (n_samples,) bool — True when centre station has ERA5 data
      k_loc      : int  k+1 (local neighbourhood size)
    """
    K_LOC  = k + 1
    cache  = MULTIMODAL_DIR / f"graph_sequences_local_w{window}_k{k}.npy"
    mcache = MULTIMODAL_DIR / f"graph_seq_mask_local_w{window}_k{k}.npy"
    if cache.exists() and mcache.exists():
        logger.info("Loading local graph sequences from cache (window=%d, k=%d)", window, k)
        return np.load(str(cache), mmap_mode="r"), np.load(str(mcache)), K_LOC

    n          = len(meta)
    sid_to_idx = {s: i for i, s in enumerate(station_ids)}

    # Precompute per-station local node list: [centre, nbr1, …, nbrK]
    sid_to_local: dict[str, list] = {}
    for sid, si in sid_to_idx.items():
        nbr_indices = np.where(A[si] > 0)[0][:k]
        sid_to_local[sid] = [sid] + [station_ids[ni] for ni in nbr_indices]

    # Allocate output as on-disk memmap — never holds the whole array in RAM
    gb = n * window * K_LOC * N_ERA5 * 4 / 1e9
    logger.info("Allocating graph sequence memmap: %.1f GB  (%d, %d, %d, %d)",
                gb, n, window, K_LOC, N_ERA5)
    graph_seq = np.lib.format.open_memmap(
        str(cache), mode="w+", dtype=np.float32,
        shape=(n, window, K_LOC, N_ERA5))
    node_mask = np.zeros(n, dtype=bool)

    for i, row in enumerate(meta.itertuples(index=False)):
        sid  = str(row.station_id)
        date = pd.Timestamp(row.date)
        if sid not in station_ts or sid not in sid_to_local:
            continue
        end   = date
        start = end - pd.Timedelta(days=window)

        for j, nsid in enumerate(sid_to_local[sid]):
            if nsid not in station_ts:
                continue
            ts  = station_ts[nsid]
            win = ts.loc[start: end - pd.Timedelta(days=1)]
            if len(win) == 0:
                continue
            n_d = min(len(win), window)
            graph_seq[i, -n_d:, j] = win.values[-n_d:].astype("float32")

        node_mask[i] = True
        if (i + 1) % 50_000 == 0:
            logger.info("  %d / %d graph sequences built", i + 1, n)

    np.save(str(mcache), node_mask)
    logger.info("Graph sequence memmap saved → %s", cache)
    return np.load(str(cache), mmap_mode="r"), node_mask, K_LOC


# ── Model architectures ───────────────────────────────────────────────────────

# 1. ConvLSTM ──────────────────────────────────────────────────────────────────

class _ConvLSTMCell(object):
    """Single ConvLSTM cell implemented in pure PyTorch."""
    def __init__(self, in_c, hidden_c, kernel=3):
        import torch.nn as nn
        self.hidden_c = hidden_c
        pad = kernel // 2
        self.conv = nn.Conv2d(in_c + hidden_c, 4 * hidden_c, kernel, padding=pad)

    def to(self, device):
        self.conv = self.conv.to(device)
        return self

    def forward(self, x, h, c):
        import torch
        inp = torch.cat([x, h], dim=1)
        gates = self.conv(inp)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        g = torch.tanh(g);    o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMModel:
    """
    ConvLSTM for spatio-temporal ERA5 patch sequences.
    Input : (B, T, 9, 12, 12)
    Output: (B, n_targets)
    """
    def __init__(self, n_targets=1, hidden_c=32, n_layers=2):
        import torch.nn as nn, torch
        self.hidden_c = hidden_c
        self.n_layers = n_layers

        cells_list = []
        for l in range(n_layers):
            in_c = PATCH_C if l == 0 else hidden_c
            cell = _ConvLSTMCell(in_c, hidden_c)
            cells_list.append(cell)
        self.cells = cells_list

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(hidden_c * 16, 64),
            nn.ReLU(),
            nn.Linear(64, n_targets),
        )

    def to(self, device):
        import torch.nn as nn
        self.head = self.head.to(device)
        for cell in self.cells:
            cell.to(device)
        return self

    def parameters(self):
        params = list(self.head.parameters())
        for cell in self.cells:
            params += list(cell.conv.parameters())
        return params

    def __call__(self, x):
        import torch
        B, T, C, H, W = x.shape
        device = x.device
        h = [torch.zeros(B, self.hidden_c, H, W, device=device)
             for _ in self.cells]
        c = [torch.zeros(B, self.hidden_c, H, W, device=device)
             for _ in self.cells]
        for t in range(T):
            inp = x[:, t]
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell.forward(inp, h[l], c[l])
                inp = h[l]
        return self.head(h[-1])

    def train(self):
        self.head.train()
        for cell in self.cells:
            cell.conv.train()

    def eval(self):
        self.head.eval()
        for cell in self.cells:
            cell.conv.eval()


# 2. CNN-Transformer ───────────────────────────────────────────────────────────

def _build_cnn_transformer(n_targets: int = 1):
    import torch.nn as nn

    class CNNTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(PATCH_C, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),      nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 3)),
                nn.Flatten(),
                nn.Linear(64 * 9, 128), nn.ReLU(),
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=128, nhead=4, dim_feedforward=256,
                dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.cls_token = nn.Parameter(
                __import__("torch").zeros(1, 1, 128))
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_targets))

        def forward(self, x):
            import torch
            B, T, C, H, W = x.shape
            frames = x.view(B * T, C, H, W)
            emb    = self.cnn(frames).view(B, T, 128)
            cls    = self.cls_token.expand(B, -1, -1)
            seq    = torch.cat([cls, emb], dim=1)
            out    = self.transformer(seq)
            return self.head(out[:, 0])

    return CNNTransformer()


# 3. TimeSformer ───────────────────────────────────────────────────────────────

def _build_timesformer(n_targets: int = 1,
                       patch_size: int = 3,
                       dim: int = 96,
                       depth: int = 4,
                       heads: int = 4):
    """
    Divided space-time attention (Bertasius et al. 2021).
    Input: (B, T, C, H, W).  Tokenise each frame into (H/p)*(W/p) patches.
    Temporal attention first (same spatial position across T),
    then spatial attention (all spatial positions within one frame).
    """
    import torch, torch.nn as nn

    n_patches = (PATCH_H // patch_size) * (PATCH_W // patch_size)  # 16 for p=3

    class DividedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.spatial_attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ff    = nn.Sequential(
                nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            self.norm3 = nn.LayerNorm(dim)

        def forward(self, x, T, P):
            # x: (B, T*P, dim)
            B = x.shape[0]
            # --- temporal attention: group by spatial position ---
            xr = x.view(B, T, P, -1)                    # (B,T,P,dim)
            xr = xr.permute(0, 2, 1, 3).reshape(B*P, T, -1)
            ta, _ = self.temporal_attn(xr, xr, xr)
            xr = (xr + ta).reshape(B, P, T, -1).permute(0, 2, 1, 3)
            x  = self.norm1(x + xr.reshape(B, T*P, -1))
            # --- spatial attention: group by timestep ---
            xr = x.view(B, T, P, -1).reshape(B*T, P, -1)
            sa, _ = self.spatial_attn(xr, xr, xr)
            xr = (xr + sa).reshape(B, T, P, -1).reshape(B, T*P, -1)
            x  = self.norm2(x + xr)
            x  = self.norm3(x + self.ff(x))
            return x

    class TimeSformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(
                PATCH_C, dim, kernel_size=patch_size, stride=patch_size)
            self.pos_emb     = nn.Parameter(
                torch.zeros(1, n_patches, dim))
            self.temp_emb    = nn.Parameter(
                torch.zeros(1, PATCH_H // patch_size * PATCH_W // patch_size, dim))
            self.layers = nn.ModuleList([DividedAttention() for _ in range(depth)])
            self.head   = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, n_targets))

        def forward(self, x):
            B, T, C, H, W = x.shape
            frames = x.view(B * T, C, H, W)
            tokens = self.patch_embed(frames)           # (B*T, dim, H/p, W/p)
            P      = tokens.shape[2] * tokens.shape[3]
            tokens = tokens.flatten(2).permute(0, 2, 1)  # (B*T, P, dim)
            tokens = tokens + self.pos_emb[:, :P]
            tokens = tokens.view(B, T, P, -1).reshape(B, T * P, -1)
            for layer in self.layers:
                tokens = layer(tokens, T, P)
            pooled = tokens.view(B, T, P, -1).mean(dim=(1, 2))
            return self.head(pooled)

    return TimeSformer()


# 4. STGCN ─────────────────────────────────────────────────────────────────────

def _build_stgcn(n_nodes: int, n_targets: int = 1):
    """
    Spatio-Temporal Graph CNN (Yu et al. 2018).
    Two ST-Conv blocks: (temporal_conv → graph_conv → temporal_conv).
    Input: (B, T, N, C).  Output: (B, n_targets) predicting the centre node.
    """
    import torch, torch.nn as nn

    class GraphConv(nn.Module):
        """Chebyshev K=1 graph convolution: H' = A_hat H W."""
        def __init__(self, in_c, out_c):
            super().__init__()
            self.W = nn.Linear(in_c, out_c, bias=False)

        def forward(self, x, A):
            # x: (B, T, N, C),  A: (N, N)
            return self.W(torch.matmul(A, x))

    class STConvBlock(nn.Module):
        def __init__(self, in_c, hid_c, out_c, kt=3):
            super().__init__()
            pad = kt // 2
            self.t1 = nn.Conv2d(in_c,  hid_c, (kt, 1), padding=(pad, 0))
            self.gc = GraphConv(hid_c, hid_c)
            self.t2 = nn.Conv2d(hid_c, out_c, (kt, 1), padding=(pad, 0))
            self.bn = nn.BatchNorm2d(out_c)

        def forward(self, x, A):
            # x: (B, T, N, C) → permute to (B, C, T, N) for Conv2d
            B, T, N, C = x.shape
            h = x.permute(0, 3, 1, 2)           # (B, C, T, N)
            h = torch.relu(self.t1(h))           # (B, hid, T, N)
            h = h.permute(0, 2, 3, 1)           # (B, T, N, hid)
            h = torch.relu(self.gc(h, A))        # (B, T, N, hid)
            h = h.permute(0, 3, 1, 2)           # (B, hid, T, N)
            h = self.bn(torch.relu(self.t2(h))) # (B, out, T, N)
            return h.permute(0, 2, 3, 1)        # (B, T, N, out)

    class STGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = STConvBlock(N_ERA5, 32, 64)
            self.block2 = STConvBlock(64,     32, 64)
            self.head   = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_targets))

        def forward(self, x, A):
            h = self.block1(x, A)
            h = self.block2(h, A)
            # Global pooling over time; predict centre node (index 0)
            h = h.mean(dim=1)   # (B, N, 64)
            return self.head(h[:, 0])  # predict centre node

    return STGCN()


# 5. DCRNN ─────────────────────────────────────────────────────────────────────

def _build_dcrnn(n_nodes: int, n_targets: int = 1, K: int = 2):
    """
    Diffusion Convolutional RNN (Li et al. 2018).
    Single-layer DCGRU with K-step diffusion convolution.
    Input: (B, T, N, C).  Output: (B, n_targets).
    """
    import torch, torch.nn as nn

    class DiffConv(nn.Module):
        """K-step Chebyshev diffusion convolution on both directions of graph."""
        def __init__(self, in_c, out_c, K):
            super().__init__()
            self.K    = K
            self.W    = nn.Linear((2 * K + 1) * in_c, out_c)

        def forward(self, x, A_fwd, A_bwd):
            # x: (B, N, C),  A_fwd/A_bwd: (N, N)
            supports = []
            h_fwd, h_bwd = x, x
            for _ in range(self.K):
                h_fwd = torch.matmul(A_fwd, h_fwd)
                h_bwd = torch.matmul(A_bwd, h_bwd)
                supports.extend([h_fwd, h_bwd])
            supports = [x] + supports           # include 0-hop
            return self.W(torch.cat(supports, dim=-1))

    class DCGRUCell(nn.Module):
        def __init__(self, in_c, hid_c, K):
            super().__init__()
            self.hid_c = hid_c
            self.r_dc  = DiffConv(in_c + hid_c, hid_c, K)
            self.u_dc  = DiffConv(in_c + hid_c, hid_c, K)
            self.c_dc  = DiffConv(in_c + hid_c, hid_c, K)

        def forward(self, x, h, A_fwd, A_bwd):
            xu = torch.cat([x, h], dim=-1)
            r  = torch.sigmoid(self.r_dc(xu, A_fwd, A_bwd))
            u  = torch.sigmoid(self.u_dc(xu, A_fwd, A_bwd))
            xrh = torch.cat([x, r * h], dim=-1)
            c  = torch.tanh(self.c_dc(xrh, A_fwd, A_bwd))
            return u * h + (1 - u) * c

    class DCRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = DCGRUCell(N_ERA5, 64, K)
            self.head = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_targets))

        def forward(self, x, A):
            import torch
            B, T, N, C = x.shape
            device = x.device
            A_bwd  = A.t()
            h = torch.zeros(B, N, 64, device=device)
            for t in range(T):
                h = self.cell(x[:, t], h, A, A_bwd)
            return self.head(h[:, 0])  # predict centre node

    return DCRNN()


# ── Lazy dataset (avoids materialising large mmap arrays) ─────────────────────

class _LazyNpyDataset:
    """
    torch Dataset backed by a (possibly memory-mapped) numpy array.

    Reads one sample at a time from X[global_indices[i]] so that large
    patch-sequence arrays (e.g. 883 k × 30 × 9 × 12 × 12, ~1.4 TB full)
    are never materialised in RAM.  Works transparently for both regular
    numpy arrays and numpy.memmap objects.
    """

    def __init__(self, X, global_indices: np.ndarray, y: np.ndarray):
        self.X   = X
        self.idx = global_indices
        self.y   = y.astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        import torch
        # np.array() forces a copy out of the mmap page — required so that
        # the DataLoader worker can serialise it without keeping the mmap open.
        x = np.array(self.X[self.idx[i]], dtype=np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[i])


# ── Training loop (shared) ────────────────────────────────────────────────────

FOLD_TIMEOUT_S = 1800  # 30 min per fold; kills CUDA hangs


def _train_nn(model, X, train_idx, y_train,
              val_idx, y_val,
              A=None, epochs=30, batch=256, lr=1e-3,
              patience=5, device="cpu"):
    """
    Generic mini-batch training loop using lazy per-sample loading.

    X        : numpy array or memmap — full dataset (never sliced whole)
    train_idx: 1-D integer array of training indices into X
    val_idx  : 1-D integer array of validation indices into X
    """
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader

    At = torch.tensor(A, dtype=torch.float32).to(device) if A is not None else None

    train_ds = _LazyNpyDataset(X, train_idx, y_train)
    val_ds   = _LazyNpyDataset(X, val_idx,   y_val)

    # num_workers=0: safest on Windows; mmap pages are shared across threads
    # anyway so parallel workers don't give a meaningful speedup here.
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=0, pin_memory=(device == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=0, pin_memory=(device == "cuda"))

    params = model.parameters()
    optim  = torch.optim.Adam(params, lr=lr)
    crit   = nn.MSELoss()

    best_val, best_state, no_imp = float("inf"), None, 0
    fold_start = time.time()

    for _ in range(epochs):
        if time.time() - fold_start > FOLD_TIMEOUT_S:
            logger.warning("Fold wall-clock timeout (%ds) — stopping early",
                           FOLD_TIMEOUT_S)
            break
        model.train()
        for xb, yb in train_dl:
            xb = torch.nan_to_num(xb.to(device), nan=0.0, posinf=1.0, neginf=-1.0)
            yb = yb.to(device).unsqueeze(1)
            optim.zero_grad()
            pred = model(xb, At) if At is not None else model(xb)
            loss = crit(pred, yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

        # Validation loss — batched to avoid loading all val samples at once
        model.eval()
        vl_sum, vl_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = torch.nan_to_num(xb.to(device), nan=0.0, posinf=1.0, neginf=-1.0)
                yb = yb.to(device).unsqueeze(1)
                pv  = model(xb, At) if At is not None else model(xb)
                vl = float(crit(pv, yb))
                if not math.isfinite(vl):
                    continue
                vl_sum += vl * len(yb)
                vl_n   += len(yb)
        vl = vl_sum / max(vl_n, 1)

        if vl < best_val - 1e-5:
            best_val   = vl
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()} \
                         if hasattr(model, "state_dict") else None
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state and hasattr(model, "load_state_dict"):
        model.load_state_dict(best_state)
    return model


def _predict_nn(model, X, idx, A=None, batch=512, device="cpu"):
    """Run inference in batches using lazy loading from X[idx]."""
    import torch
    from torch.utils.data import DataLoader

    At  = torch.tensor(A, dtype=torch.float32).to(device) if A is not None else None
    ds  = _LazyNpyDataset(X, idx, np.zeros(len(idx), dtype=np.float32))
    dl  = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            # Mirror _train_nn: clamp NaN/inf so inputs never propagate to outputs
            xb  = torch.nan_to_num(xb.to(device), nan=0.0, posinf=1.0, neginf=-1.0)
            out = model(xb, At) if At is not None else model(xb)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0).squeeze(-1)


# ── CV runner ─────────────────────────────────────────────────────────────────

def _run_cv(model_name: str, meta: pd.DataFrame,
            X: np.ndarray, targets_log: dict,
            A=None, n_splits: int = 5,
            epochs: int = 30, batch: int = 256,
            device: str = "cpu",
            model_factory=None) -> tuple:
    """
    GroupKFold cross-validation for any neural model.

    Passes indices into X rather than sliced sub-arrays so that large
    memory-mapped patch-sequence arrays are never fully materialised.
    model_factory() must return a freshly initialised model instance.
    """
    groups = (pd.Categorical(meta["fold_key"].fillna("unknown")).codes
              if "fold_key" in meta.columns
              else np.zeros(len(meta), dtype=int))
    gkf       = GroupKFold(n_splits=n_splits)
    rows      = []
    preds_dict = {}   # {target: full-n OOF predictions (log-space)}
    cal_dict   = {}   # {target: [(cal_y, cal_p, val_y, val_p, val_idx), ...]}

    for tgt in TARGETS:
        if tgt not in targets_log:
            continue
        y          = targets_log[tgt]
        fold_preds = np.full(len(meta), np.nan)
        cal_data   = []
        cur_device = device

        for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
            n_cal   = max(1, int(0.2 * len(tr)))
            fit_idx = tr[:-n_cal]
            cal_idx = tr[-n_cal:]

            if len(fit_idx) > MAX_TRAIN_SAMPLES:
                rng     = np.random.default_rng(42 + fold)
                fit_idx = rng.choice(fit_idx, MAX_TRAIN_SAMPLES, replace=False)

            for attempt, dev in enumerate([cur_device, "cpu"]):
                try:
                    model = model_factory()
                    if hasattr(model, "to"):
                        model = model.to(dev)
                    model = _train_nn(
                        model, X,
                        train_idx=fit_idx, y_train=y[fit_idx],
                        val_idx=va,        y_val=y[va],
                        A=A, epochs=epochs, batch=batch, device=dev)

                    cal_p = _predict_nn(model, X, idx=cal_idx,
                                        A=A, batch=batch, device=dev)
                    val_p = _predict_nn(model, X, idx=va,
                                        A=A, batch=batch, device=dev)

                    fold_preds[va] = val_p
                    cal_data.append((
                        y[cal_idx], cal_p,   # calibration true / pred
                        y[va],      val_p,   # validation true / pred
                        va,                  # global val indices (for trophic lookup)
                    ))
                    cur_device = dev
                    break
                except Exception as e:
                    err = str(e)
                    if attempt == 0 and ("cuda" in err.lower() or "CUDA" in err):
                        logger.warning("fold %d CUDA error — retrying on CPU: %s",
                                       fold, err)
                        cur_device = "cpu"
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    else:
                        logger.warning("fold %d failed on %s: %s", fold, dev, e)
                        break

        ok = np.isfinite(fold_preds) & np.isfinite(y)
        m  = _metrics(y[ok], fold_preds[ok])
        m.update({"model": model_name, "target": tgt})
        rows.append(m)
        preds_dict[tgt] = fold_preds
        cal_dict[tgt]   = cal_data
        logger.info("  %s | %s | R²=%.3f RMSE=%.3f n=%d",
                    model_name, tgt, m["r2"], m["rmse"], m["n"])

    return rows, preds_dict, cal_dict


# ── Incremental checkpoint ────────────────────────────────────────────────────

def _checkpoint(rows: list, out_dir: Path) -> None:
    """Write current results to CSV immediately after each model completes."""
    if not rows:
        return
    df = pd.DataFrame(rows)[["model", "target", "rmse", "r2", "mae", "n"]]
    path = out_dir / "spatiotemporal_baseline_results.csv"
    df.to_csv(path, index=False)
    logger.info("Checkpoint saved → %s (%d rows)", path, len(df))


def _checkpoint_extra(rows: list, out_dir: Path, filename: str,
                      col_order: list) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = [c for c in col_order if c in df.columns]
    df[cols].to_csv(out_dir / filename, index=False)
    logger.info("Checkpoint saved → %s (%d rows)", filename, len(df))


# ── Extra metric helpers ───────────────────────────────────────────────────────

LEVELS         = (0.80, 0.90, 0.95)
TROPHIC_BINS   = [0, 10, 35, 100, np.inf]
TROPHIC_LABELS = ["oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic"]

ROLLING_WINDOWS = [
    ("2011-15", pd.Timestamp("2010-12-31"),
     pd.Timestamp("2011-01-01"), pd.Timestamp("2015-12-31")),
    ("2016-20", pd.Timestamp("2015-12-31"),
     pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
    ("2021-25", pd.Timestamp("2020-12-31"),
     pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")),
]


def _compute_rolling_origin(model_name: str, meta: pd.DataFrame,
                             X: np.ndarray, targets_log: dict,
                             A=None, model_factory=None,
                             epochs: int = 30, batch: int = 128,
                             device: str = "cpu") -> list:
    """Expanding-window temporal evaluation (no GroupKFold — single train/test split)."""
    rows  = []
    dates = pd.to_datetime(meta["date"].values)
    for period, train_end, test_start, test_end in ROLLING_WINDOWS:
        tr_idx = np.where(dates <= train_end)[0]
        va_idx = np.where((dates >= test_start) & (dates <= test_end))[0]
        if len(tr_idx) < 100 or len(va_idx) < 10:
            logger.info("  rolling %s|%s: skip (tr=%d va=%d)",
                        model_name, period, len(tr_idx), len(va_idx))
            continue
        for tgt in TARGETS:
            if tgt not in targets_log:
                continue
            y = targets_log[tgt]
            try:
                model = model_factory()
                if hasattr(model, "to"):
                    model = model.to(device)
                model = _train_nn(model, X,
                                  train_idx=tr_idx, y_train=y[tr_idx],
                                  val_idx=va_idx,   y_val=y[va_idx],
                                  A=A, epochs=epochs, batch=batch, device=device)
                preds = _predict_nn(model, X, va_idx, A=A, batch=batch, device=device)
                m = _metrics(y[va_idx], preds)
                rows.append({"model": model_name, "target": tgt,
                             "test_period": period, **m})
                logger.info("  rolling %s|%s|%s R²=%.3f",
                            model_name, tgt, period, m["r2"])
            except Exception as e:
                logger.warning("rolling %s|%s|%s failed: %s",
                               model_name, tgt, period, e)
    return rows


def _trophic_series(meta: pd.DataFrame) -> np.ndarray:
    if "tp_ug_l" not in meta.columns:
        return np.full(len(meta), "unknown", dtype=object)
    tp = meta["tp_ug_l"].clip(lower=0)
    return pd.cut(tp, bins=TROPHIC_BINS,
                  labels=TROPHIC_LABELS).astype(str).values


def _compute_conformal(model_name: str, cal_dict: dict) -> list:
    """Split conformal PICP / ECE / interval width per target × level."""
    rows = []
    for tgt, folds in cal_dict.items():
        agg = {lv: {"picp": [], "width": []} for lv in LEVELS}
        for cal_y, cal_p, val_y, val_p, _va in folds:
            scores = np.abs(cal_y - cal_p)
            scores = scores[np.isfinite(scores)]   # drop NaN/inf from unstable folds
            n = len(scores)
            if n < 5:
                continue
            # Also drop non-finite val predictions before coverage check
            val_ok = np.isfinite(val_p) & np.isfinite(val_y)
            if val_ok.sum() < 5:
                continue
            val_y_ok = val_y[val_ok]; val_p_ok = val_p[val_ok]
            for lv in LEVELS:
                alpha = 1 - lv
                q = float(np.quantile(
                    scores,
                    min(1.0, np.ceil((n + 1) * (1 - alpha)) / n),
                    method="higher"))
                covered = (val_y_ok >= val_p_ok - q) & (val_y_ok <= val_p_ok + q)
                agg[lv]["picp"].append(float(covered.mean()))
                agg[lv]["width"].append(2.0 * q)
        for lv in LEVELS:
            ps = agg[lv]["picp"]
            if not ps:
                continue
            picp = float(np.mean(ps))
            rows.append({
                "model":                   model_name,
                "target":                  tgt,
                "nominal_coverage":        lv,
                "picp":                    round(picp, 3),
                "ece":                     round(abs(picp - lv), 4),
                "mean_interval_width_log": round(float(np.mean(agg[lv]["width"])), 4),
                "n_folds":                 len(ps),
            })
    return rows


def _compute_trophic(model_name: str, meta: pd.DataFrame,
                     preds_dict: dict, targets_log: dict) -> list:
    """R² / RMSE / MAE stratified by trophic class (TP-based thresholds)."""
    tc = _trophic_series(meta)
    rows = []
    for tgt in TARGETS:
        if tgt not in preds_dict or tgt not in targets_log:
            continue
        preds = preds_dict[tgt]
        y_log = targets_log[tgt]
        for cls in TROPHIC_LABELS:
            mask = (tc == cls) & np.isfinite(preds) & np.isfinite(y_log)
            if mask.sum() < 5:
                continue
            m = _metrics(y_log[mask], preds[mask])
            rows.append({"model": model_name, "target": tgt,
                         "trophic_class": cls, **m})
    return rows


def _winkler(y_true: np.ndarray, lo: np.ndarray,
             hi: np.ndarray, alpha: float) -> float:
    width   = hi - lo
    penalty = (2 / alpha) * (np.maximum(0, lo - y_true) +
                              np.maximum(0, y_true - hi))
    return float(np.mean(width + penalty))


def _compute_quantile(model_name: str, meta: pd.DataFrame,
                      cal_dict: dict) -> list:
    """Winkler score / PICP / PINAW per trophic class × nominal level."""
    tc_all = _trophic_series(meta)
    rows   = []
    for tgt, folds in cal_dict.items():
        # Accumulate val predictions and trophic class across all folds
        val_y_all  = np.concatenate([f[2] for f in folds])
        val_p_all  = np.concatenate([f[3] for f in folds])
        val_va_all = np.concatenate([f[4] for f in folds])
        val_tc     = tc_all[val_va_all]
        # Filter rows where either prediction or label is non-finite
        _fin = np.isfinite(val_p_all) & np.isfinite(val_y_all)
        val_y_all  = val_y_all[_fin]
        val_p_all  = val_p_all[_fin]
        val_tc     = val_tc[_fin]

        # Per-fold calibration quantiles (proper finite-sample correction)
        for lv in LEVELS:
            alpha   = 1 - lv
            qs_fold = []
            for cal_y, cal_p, _vy, _vp, _va in folds:
                scores = np.abs(cal_y - cal_p)
                scores = scores[np.isfinite(scores)]   # drop NaN/inf
                n = len(scores)
                if n < 5:
                    continue
                qs_fold.append(float(np.quantile(
                    scores,
                    min(1.0, np.ceil((n + 1) * (1 - alpha)) / n),
                    method="higher")))
            if not qs_fold:
                continue
            q = float(np.mean(qs_fold))

            lo = val_p_all - q
            hi = val_p_all + q

            for cls in (["all"] + list(TROPHIC_LABELS)):
                if cls == "all":
                    mask = np.ones(len(val_y_all), dtype=bool)
                else:
                    mask = (val_tc == cls)
                if mask.sum() < 5:
                    continue
                y_tc  = val_y_all[mask]
                lo_tc = lo[mask];  hi_tc = hi[mask]
                picp  = float(np.mean((y_tc >= lo_tc) & (y_tc <= hi_tc)))
                pinaw = float(np.mean(hi_tc - lo_tc))
                ws    = _winkler(y_tc, lo_tc, hi_tc, alpha)
                rows.append({
                    "model":           model_name,
                    "target":          tgt,
                    "trophic_class":   cls,
                    "nominal_coverage": lv,
                    "picp":            round(picp, 3),
                    "ece":             round(abs(picp - lv), 4),
                    "pinaw":           round(pinaw, 4),
                    "winkler_score":   round(ws, 3),
                    "n":               int(mask.sum()),
                })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["convlstm", "cnn_transformer",
                             "timesformer", "stgcn", "dcrnn"],
                    choices=["convlstm", "cnn_transformer",
                             "timesformer", "stgcn", "dcrnn"])
    ap.add_argument("--window",      type=int, default=30)
    ap.add_argument("--k_neighbors", type=int, default=8)
    ap.add_argument("--epochs",      type=int, default=30)
    ap.add_argument("--batch",       type=int, default=128)
    args = ap.parse_args()

    device = _device()
    logger.info("Device: %s", device)

    meta = _load_metadata()
    logger.info("Metadata: %d samples", len(meta))

    # Targets (log1p-transformed)
    targets_log = {}
    for tgt in TARGETS:
        if tgt in meta.columns:
            col = np.log1p(meta[tgt].clip(lower=0).values.astype("float32"))
            targets_log[tgt] = col

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    CONF_COLS = ["model", "target", "nominal_coverage", "picp", "ece",
                 "mean_interval_width_log", "n_folds"]
    TROPH_COLS = ["model", "target", "trophic_class", "r2", "rmse", "mae", "n"]
    QUANT_COLS = ["model", "target", "trophic_class", "nominal_coverage",
                  "picp", "ece", "pinaw", "winkler_score", "n"]

    # ── Resume: load existing checkpoints ─────────────────────────────────────
    def _load_checkpoint(filename, col_order):
        p = OUT_DIR / filename
        if p.exists():
            df = pd.read_csv(p)
            return df.to_dict("records")
        return []

    ROLL_COLS = ["model", "target", "test_period", "r2", "rmse", "mae", "n"]

    all_rows       = _load_checkpoint("spatiotemporal_baseline_results.csv",
                                      ["model", "target", "rmse", "r2", "mae", "n"])
    conformal_rows = _load_checkpoint("spatiotemporal_conformal_results.csv", CONF_COLS)
    trophic_rows   = _load_checkpoint("spatiotemporal_trophic_results.csv",   TROPH_COLS)
    quantile_rows  = _load_checkpoint("spatiotemporal_quantile_results.csv",  QUANT_COLS)
    rolling_rows   = _load_checkpoint("spatiotemporal_rolling_origin.csv",    ROLL_COLS)

    completed_main    = {r["model"] for r in all_rows}
    completed_conf    = {r["model"] for r in conformal_rows}
    completed_trophic = {r["model"] for r in trophic_rows}
    completed_quant   = {r["model"] for r in quantile_rows}
    completed_rolling = {r["model"] for r in rolling_rows}
    # Only skip a model entirely if ALL derived outputs are present
    completed = completed_main & completed_conf & completed_trophic & completed_quant
    if completed:
        logger.info("Resuming — fully completed: %s", sorted(completed))
    missing_conf = completed_main - completed_conf
    if missing_conf:
        logger.info("Conformal missing for: %s — will re-run CV", sorted(missing_conf))
    missing_rolling = completed_main - completed_rolling
    if missing_rolling:
        logger.info("Rolling origin pending: %s", sorted(missing_rolling))

    # ── Patch-sequence models ─────────────────────────────────────────────────
    patch_models = [m for m in args.models
                    if m in ("convlstm", "cnn_transformer", "timesformer")]
    pending_patch = [m for m in patch_models if m not in completed]
    if pending_patch:
        patches = _load_patches()
        station_ts = _load_era5_station_ts()
        if len(patches) == 0 or not station_ts:
            logger.warning("Patch or ERA5 data missing — skipping patch models")
            patch_models = []
        else:
            logger.info("Building patch sequences (window=%d)…", args.window)
            patch_seq = _build_patch_sequences(
                meta, patches, station_ts, args.window)
            # (n, T, 9, 12, 12)
    else:
        logger.info("All patch models already in checkpoint — skipping patch data load")

    for mname in patch_models:
        if mname in completed:
            logger.info("=== %s — skipping (already in checkpoint) ===", mname)
            continue
        logger.info("=== %s ===", mname)

        if mname == "convlstm":
            def factory():
                m = ConvLSTMModel(n_targets=1)
                return m
        elif mname == "cnn_transformer":
            def factory():
                return _build_cnn_transformer(n_targets=1)
        else:  # timesformer
            def factory():
                return _build_timesformer(n_targets=1)

        rows, preds_dict, cal_dict = _run_cv(
            mname, meta, patch_seq, targets_log,
            A=None, epochs=args.epochs, batch=args.batch,
            device=device, model_factory=factory)
        if mname not in completed_main:
            all_rows.extend(rows)
        if mname not in completed_conf:
            conformal_rows.extend(_compute_conformal(mname, cal_dict))
        if mname not in completed_trophic:
            trophic_rows.extend(_compute_trophic(mname, meta, preds_dict, targets_log))
        if mname not in completed_quant:
            quantile_rows.extend(_compute_quantile(mname, meta, cal_dict))
        _checkpoint(all_rows, OUT_DIR)
        _checkpoint_extra(conformal_rows, OUT_DIR,
                          "spatiotemporal_conformal_results.csv", CONF_COLS)
        _checkpoint_extra(trophic_rows, OUT_DIR,
                          "spatiotemporal_trophic_results.csv", TROPH_COLS)
        _checkpoint_extra(quantile_rows, OUT_DIR,
                          "spatiotemporal_quantile_results.csv", QUANT_COLS)

    # ── Graph-based models ────────────────────────────────────────────────────
    graph_models = [m for m in args.models if m in ("stgcn", "dcrnn")]
    if graph_models:
        station_ts = _load_era5_station_ts() if "station_ts" not in dir() else station_ts
        try:
            station_ids, A, sid_to_idx = _build_station_graph(
                meta, k=args.k_neighbors)
        except FileNotFoundError as e:
            logger.error("%s — skipping graph models", e)
            graph_models = []

    if graph_models:
        logger.info("Building local graph sequences (window=%d, k=%d)…",
                    args.window, args.k_neighbors)
        graph_seq, node_mask, K_LOC = _build_graph_sequences(
            meta, station_ts, station_ids, A, args.window, args.k_neighbors)
        # graph_seq: (n, T, k+1, 7)  — node 0 = centre station

        import torch
        # Fixed fully-connected local adjacency (k+1 × k+1), row-normalised
        A_local = np.ones((K_LOC, K_LOC), dtype=np.float32)
        np.fill_diagonal(A_local, 0.0)
        row_sum = A_local.sum(axis=1, keepdims=True)
        A_local /= np.where(row_sum == 0, 1.0, row_sum)
        A_tensor = torch.tensor(A_local, dtype=torch.float32)
        N_nodes  = K_LOC

        for mname in graph_models:
            if mname in completed:
                logger.info("=== %s — skipping (already in checkpoint) ===", mname)
                continue
            logger.info("=== %s ===", mname)

            if mname == "stgcn":
                def factory():
                    return _build_stgcn(n_nodes=N_nodes, n_targets=1)
            else:
                def factory():
                    return _build_dcrnn(n_nodes=N_nodes, n_targets=1)

            meta_sub = meta[node_mask].reset_index(drop=True)
            tlog_sub = {k: v[node_mask] for k, v in targets_log.items()}
            rows, preds_dict, cal_dict = _run_cv(
                mname, meta_sub, graph_seq[node_mask], tlog_sub,
                A=A_local, epochs=args.epochs, batch=64,
                device=device, model_factory=factory)
            if mname not in completed_main:
                all_rows.extend(rows)
            if mname not in completed_conf:
                conformal_rows.extend(_compute_conformal(mname, cal_dict))
            if mname not in completed_trophic:
                trophic_rows.extend(_compute_trophic(mname, meta_sub,
                                                      preds_dict, tlog_sub))
            if mname not in completed_quant:
                quantile_rows.extend(_compute_quantile(mname, meta_sub, cal_dict))
            _checkpoint(all_rows, OUT_DIR)
            _checkpoint_extra(conformal_rows, OUT_DIR,
                              "spatiotemporal_conformal_results.csv", CONF_COLS)
            _checkpoint_extra(trophic_rows, OUT_DIR,
                              "spatiotemporal_trophic_results.csv", TROPH_COLS)
            _checkpoint_extra(quantile_rows, OUT_DIR,
                              "spatiotemporal_quantile_results.csv", QUANT_COLS)

    # ── Rolling origin pass (separate from main CV) ───────────────────────────
    # Runs only for models that have baseline results but no rolling origin yet.
    patch_need_ro = [m for m in args.models
                     if m in ("convlstm", "cnn_transformer", "timesformer")
                     and m in completed_main and m not in completed_rolling]
    graph_need_ro = [m for m in args.models
                     if m in ("stgcn", "dcrnn")
                     and m in completed_main and m not in completed_rolling]

    if patch_need_ro:
        logger.info("Rolling origin — patch models: %s", patch_need_ro)
        # Load patch data if not already in scope from main loops
        if "patch_seq" not in dir() or patch_seq is None:
            _patches   = _load_patches()
            _st        = _load_era5_station_ts()
            patch_seq  = _build_patch_sequences(meta, _patches, _st, args.window)

        for mname in patch_need_ro:
            logger.info("=== %s rolling origin ===", mname)
            if mname == "convlstm":
                def factory(): return ConvLSTMModel(n_targets=1)
            elif mname == "cnn_transformer":
                def factory(): return _build_cnn_transformer(n_targets=1)
            else:
                def factory(): return _build_timesformer(n_targets=1)
            rolling_rows.extend(_compute_rolling_origin(
                mname, meta, patch_seq, targets_log,
                A=None, model_factory=factory,
                epochs=args.epochs, batch=args.batch, device=device))
            completed_rolling.add(mname)
            _checkpoint_extra(rolling_rows, OUT_DIR,
                              "spatiotemporal_rolling_origin.csv", ROLL_COLS)

    if graph_need_ro:
        logger.info("Rolling origin — graph models: %s", graph_need_ro)
        # Load graph data if not already in scope from main loops
        if "graph_seq" not in dir() or graph_seq is None:
            _st_gr = _load_era5_station_ts()
            try:
                station_ids, A, sid_to_idx = _build_station_graph(
                    meta, k=args.k_neighbors)
            except FileNotFoundError as e:
                logger.error("%s — skipping graph rolling origin", e)
                graph_need_ro = []
        if graph_need_ro:
            if "graph_seq" not in dir() or graph_seq is None:
                graph_seq, node_mask, K_LOC = _build_graph_sequences(
                    meta, _st_gr, station_ids, A, args.window, args.k_neighbors)
                import torch as _torch
                A_local  = np.ones((K_LOC, K_LOC), dtype=np.float32)
                np.fill_diagonal(A_local, 0.0)
                _rs = A_local.sum(axis=1, keepdims=True)
                A_local /= np.where(_rs == 0, 1.0, _rs)
                N_nodes = K_LOC
            meta_sub_ro   = meta[node_mask].reset_index(drop=True)
            tlog_sub_ro   = {k: v[node_mask] for k, v in targets_log.items()}
            graph_seq_sub = graph_seq[node_mask]

            for mname in graph_need_ro:
                logger.info("=== %s rolling origin ===", mname)
                if mname == "stgcn":
                    def factory(): return _build_stgcn(n_nodes=N_nodes, n_targets=1)
                else:
                    def factory(): return _build_dcrnn(n_nodes=N_nodes, n_targets=1)
                rolling_rows.extend(_compute_rolling_origin(
                    mname, meta_sub_ro, graph_seq_sub, tlog_sub_ro,
                    A=A_local, model_factory=factory,
                    epochs=args.epochs, batch=64, device=device))
                completed_rolling.add(mname)
                _checkpoint_extra(rolling_rows, OUT_DIR,
                                  "spatiotemporal_rolling_origin.csv", ROLL_COLS)

    # ── Save results ──────────────────────────────────────────────────────────
    if not all_rows:
        logger.warning("No results to save")
        return

    df = pd.DataFrame(all_rows)[["model", "target", "rmse", "r2", "mae", "n"]]
    csv_path = OUT_DIR / "spatiotemporal_baseline_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved → %s", csv_path)

    # LaTeX table
    tex = df.pivot_table(
        index="model",
        columns="target",
        values=["r2", "rmse"],
        aggfunc="first"
    ).round(3).to_latex()
    (OUT_DIR / "spatiotemporal_baseline_results.tex").write_text(tex)
    logger.info("Done.")


if __name__ == "__main__":
    main()
