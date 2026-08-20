# AquaNord-SE

AquaNord-SE is a large-scale, analysis-ready benchmark dataset covering **883,325 station–date instances** from **18,743 monitoring stations** across Sweden over 26 years. It co-registers quality-approved in-situ water quality chemistry with five satellite sensors, ERA5-Land and MESAN reanalysis, and 72 HydroATLAS catchment attributes, and ships a gridded Water Quality Index (WQI) NetCDF at 5 km resolution.

Associated paper: *AquaNord-SE: A Large-Scale Longitudinal Multimodal Benchmark for Swedish Inland Water Quality*, CIKM '26. See [Citation](#citation).

---

## Dataset

The dataset is available at:

> **https://doi.org/10.23729/fd-b932bcb9-1705-379e-9f47-0c0af9cbaba9**

| Component | License |
| --- | --- |
| Dataset | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Code (`src/`, scripts) | [MIT License](LICENSE) |

### What is included

| Component                                 | Path                                                        | Size     |
| ----------------------------------------- | ----------------------------------------------------------- | -------- |
| Analysis-ready feature tables (×26 years) | `AquaNord-SE/feature_table/feature_table_SE_{YYYY}.parquet` | 126.6 MB |
| MESAN station extractions (×26 years)     | `AquaNord-SE/mesan/station_mesan_{YYYY}.parquet`            | —        |
| Multimodal benchmark metadata             | `AquaNord-SE/multimodal/metadata.parquet`                   | 49.2 MB  |
| CV split index                            | `AquaNord-SE/multimodal/split_index.parquet`                | 3.7 MB   |
| Derived WQ parameters (station-level)     | `AquaNord-SE/derived_wq_params.parquet`                     | 1.5 MB   |
| Derived WQ time series                    | `AquaNord-SE/derived_wq_timeseries.parquet`                 | 4.9 MB   |
| Gridded WQI NetCDF (5 km, 2000–2025)      | `AquaNord-SE/se_wqi_v1.nc`                                  | 92.7 MB  |

### Upstream data sources and attribution

AquaNord-SE is derived from SLU Miljödata MVM, Copernicus ERA5-Land and the
Sentinel missions, SMHI MESAN, NASA/USGS MODIS and Landsat, and HydroATLAS.
Each source retains its own licence; see [`ATTRIBUTION.md`](ATTRIBUTION.md) for
the per-source terms. Neither the European Commission nor ECMWF is responsible
for any use of the modified Copernicus data contained in this dataset.

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
│   └── se_wqi_v1.nc
├── run_reproducibility.py  One-command reproducibility runner
├── Dockerfile              Containerised environment
├── requirements.txt        Python dependencies
├── ATTRIBUTION.md          Per-source upstream licences
└── LICENSE                 MIT License (code)
```

---

## Installation

### Option 1 — Local Python environment

```bash
git clone https://github.com/moinul7002/AquaNord-SE.git
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

Download the dataset from the Etsin landing page:

```
https://doi.org/10.23729/fd-b932bcb9-1705-379e-9f47-0c0af9cbaba9
```

Place the downloaded contents so that the following paths exist:

```
AquaNord-SE/feature_table/feature_table_SE_{YYYY}.parquet (for 2000–2025)
AquaNord-SE/mesan/station_mesan_{YYYY}.parquet (for 2000–2025)
AquaNord-SE/derived_wq_params.parquet
AquaNord-SE/derived_wq_timeseries.parquet
AquaNord-SE/multimodal/metadata.parquet
AquaNord-SE/multimodal/split_index.parquet
AquaNord-SE/se_wqi_v1.nc
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

If you use AquaNord-SE, please cite **both the paper and the dataset**.

> Md Moinul Islam, Getnet Demil, and Mourad Oussalah (2026).
> AquaNord-SE: A Large-Scale Longitudinal Multimodal Benchmark for Swedish
> Inland Water Quality. In *Proceedings of the 35th ACM International
> Conference on Information and Knowledge Management (CIKM '26)*, November
> 07–11, 2026, Rome, Italy. ACM, New York, NY, USA.
> https://doi.org/10.1145/3799682.3840989

> Md Moinul Islam, Getnet Demil, and Mourad Oussalah (2026).
> AquaNord-SE: A Large-Scale Longitudinal Multimodal Benchmark for Swedish
> Inland Water Quality [Dataset] (Version 1).
> https://doi.org/10.23729/fd-b932bcb9-1705-379e-9f47-0c0af9cbaba9

Please also acknowledge the primary upstream data sources: SLU Miljödata MVM,
ERA5-Land (Copernicus/ECMWF), MESAN (SMHI), Google Earth Engine, Copernicus
Data Space Ecosystem, and HydroATLAS. See [`ATTRIBUTION.md`](ATTRIBUTION.md).

### BibTeX

```bibtex
@inproceedings{islam2026aquanordse,
  author    = {Islam, Md Moinul and Demil, Getnet and
               Oussalah, Mourad},
  title     = {{AquaNord-SE}: A Large-Scale Longitudinal Multimodal
               Benchmark for {Swedish} Inland Water Quality},
  booktitle = {Proceedings of the 35th ACM International Conference on
               Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  location  = {Rome, Italy},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3799682.3840989}
}

@misc{islam2026aquanordsedata,
  author       = {Islam, Md Moinul and Demil, Getnet and
                  Oussalah, Mourad},
  title        = {{AquaNord-SE}: A Large-Scale Longitudinal Multimodal
                  Benchmark for {Swedish} Inland Water Quality [Dataset] (Version 1)},
  year         = {2026},
  doi          = {10.23729/fd-b932bcb9-1705-379e-9f47-0c0af9cbaba9}
}
```

---

## Contact

Md Moinul Islam — moinul.islam@oulu.fi
Centre for Machine Vision and Signal Analysis, Faculty of ITEE, University of Oulu, Finland
