# AquaNord-SE

AquaNord-SE is a large-scale, analysis-ready benchmark dataset covering **883,325 station–date instances** from **18,743 monitoring stations** across Sweden over 26 years. It co-registers quality-approved in-situ water quality chemistry with five satellite sensors, ERA5-Land and MESAN reanalysis, and 72 HydroATLAS catchment attributes, and ships a gridded Water Quality Index (WQI) NetCDF at 5 km resolution.

---

## Dataset

The dataset is available at:

> **https://ida.fairdata.fi/s/NOT-FOR-PUBLICATION-M3ZYWPq5Te8t**

The code repository is available for anonymous peer review at:

> **https://anonymous.4open.science/r/AquaNord-SE**

The dataset is released under the [AquaNord-SE Academic Research License](LICENSE_DATA.md). The code in this repository is released under the [MIT License](LICENSE).

### What is included

| Component | Path | Size |
|---|---|---|
| Analysis-ready feature tables (×26 years) | `data/interim/SE/feature_table/feature_table_SE_{YYYY}.parquet` | 133 MB |
| MESAN regional reanalysis (×26 years, zarr) | `data/raw/mesan/zarr/mesan_{YYYY}.zarr/` | 67.2 GB |
| Multimodal benchmark metadata | `data/processed/nordic_multimodal_dataset/metadata.parquet` | 52 MB |
| CV split index | `data/processed/nordic_multimodal_dataset/split_index.parquet` | 4 MB |
| ERA5 static catchment patches | `data/processed/nordic_multimodal_dataset/patches.zarr/` | 1.4 GB |
| Derived WQ parameters (station-level) | `data/interim/SE/derived_wq_params.parquet` | 2 MB |
| Derived WQ time series | `data/interim/SE/derived_wq_timeseries.parquet` | 5 MB |
| Gridded WQI NetCDF (5 km, 2000–2025) | `data/outputs/nordic_water_quality_v1.nc` | 97 MB |

---

## Repository Structure

```
AquaNord-SE/
├── src/
│   ├── common/             Shared utilities (config, I/O, geo)
│   ├── extraction/         API data fetchers (SLU MVM, ERA5, MESAN, satellite)
│   ├── processing/         Dataset construction pipeline
│   ├── patches/            ERA5 and satellite patch extractors
│   ├── models/             Baseline model implementations
│   │   ├── baseline_models.py          Tabular baselines (all tiers)
│   │   ├── patch_baseline.py           Patch-based baselines
│   │   ├── multimodal_baselines.py     Multimodal baselines
│   │   └── spatiotemporal_baselines.py Spatio-temporal baselines
│   ├── evaluation/         Derived parameters, splits, validation
│   ├── figures/            Figure generation scripts
│   └── tables/             LaTeX and CSV table generation
├── outputs/
│   ├── figures/            Generated paper figures (PNG)
│   ├── results/            Benchmark result CSVs
│   ├── tables/             Summary tables
│   └── nordic_water_quality_v1.nc
├── run_reproducibility.py  One-command reproducibility runner
├── Dockerfile              Containerised environment
├── requirements.txt        Python dependencies
├── LICENSE                 MIT License (code)
└── LICENSE_DATA.md         Academic Research License (dataset)
```

---

## Installation

### Option 1 — Local Python environment

```bash
git clone https://github.com/[anonymous]/AquaNord-SE.git
cd AquaNord-SE

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Option 2 — Docker (recommended for full reproducibility)

```bash
# Build image
docker build -t aquanord-se .

# Run full pipeline (mount your local data/ directory)
docker run --rm \
  -v $(pwd)/data:/app/data \
  aquanord-se

# GPU-accelerated (for spatiotemporal models)
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  aquanord-se
```

---

## Reproducibility Instructions

### 1. Download the dataset

Download and unzip the dataset from:

```
https://ida.fairdata.fi/s/NOT-FOR-PUBLICATION-M3ZYWPq5Te8t
```

Place the downloaded contents so that the following paths exist:

```
data/interim/SE/feature_table/feature_table_SE_2000.parquet   (and 2001–2025)
data/interim/SE/derived_wq_params.parquet
data/interim/SE/derived_wq_timeseries.parquet
data/raw/mesan/zarr/mesan_2000.zarr/                           (and 2001–2025)
data/processed/nordic_multimodal_dataset/metadata.parquet
data/processed/nordic_multimodal_dataset/split_index.parquet
data/processed/nordic_multimodal_dataset/patches.zarr/
data/outputs/nordic_water_quality_v1.nc
```

### 2. Run the full reproducibility pipeline

```bash
python run_reproducibility.py
```

This runs all steps in order and skips any step whose outputs already exist.

### 3. Run selected steps only

```bash
# Tabular and patch baselines only
python run_reproducibility.py --steps baselines patch

# Baselines + figures + tables (skip heavy ST models)
python run_reproducibility.py --steps baselines patch multimodal figures tables

# Force rerun of a specific step even if outputs exist
python run_reproducibility.py --steps baselines --force
```

### 4. Run individual model scripts directly

```bash
# Tabular baselines (all tiers, all targets)
python -m src.models.baseline_models

# Patch-based baselines
python -m src.models.patch_baseline

# Multimodal baselines
python -m src.models.multimodal_baselines

# Spatio-temporal baselines (GPU recommended)
python -m src.models.spatiotemporal_baselines

# Specific model families only
python -m src.models.spatiotemporal_baselines --models convlstm cnn_transformer
python -m src.models.multimodal_baselines --models mosaiks crossmodal
```

### 5. Regenerate figures and tables

```bash
# Individual figures
python -m src.figures.coverage_map
python -m src.figures.wq_trends
python -m src.figures.wqi_map
python -m src.figures.plot_model_results

# Tables
python -m src.tables.gen_tabular_best
python -m src.tables.gen_patch_table
python -m src.evaluation.validation_tables
```

All outputs are written to `data/outputs/figures/`, `data/outputs/tables/`, and `data/outputs/results/`.

---

## Evaluation Design

| Protocol          | Design                                       | Train     | Test    |
| ----------------- | -------------------------------------------- | --------- | ------- |
| Spatial (primary) | GroupKFold(k=5) by water-body identity       | 658,422   | 169,837 |
| Temporal          | Fixed holdout 2000–2020 → 2021–2025          | 682,822   | 200,503 |
| Latitudinal       | Fixed partition at 61°N                      | 729,118   | 154,207 |
| Rolling-origin    | Expanding windows: 2011–15, 2016–20, 2021–25 | expanding | —       |

---

## Citation

Please acknowledge the primary data sources: SLU Miljödata MVM, ERA5-Land (Copernicus/ECMWF), MESAN (SMHI), Google Earth Engine, Copernicus Data Space Ecosystem, and HydroATLAS. Citations for these sources are provided in the dataset documentation. Authors will provide a full citation for the associated paper upon publication.

---

## Licenses

| Component              | License                                                                     |
| ---------------------- | --------------------------------------------------------------------------- |
| Code (`src/`, scripts) | [MIT License](LICENSE)                                                      |
| Dataset                | [Academic Research License](LICENSE_DATA.md) — non-commercial research only |

---

## Contact

Author contact information will be released upon publication.
