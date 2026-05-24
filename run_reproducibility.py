#!/usr/bin/env python3
"""
AquaNord-SE — Reproducibility Runner
=====================================
Reproduces all benchmark results, figures, and tables reported in:

    "AquaNord-SE: A Multi-Source Water Quality Benchmark
    for Swedish Inland Waters (2000–2025)". CIKM 2026.
    [Authors anonymised for peer review.]

Usage
-----
    # Full pipeline (all steps)
    python run_reproducibility.py

    # Selected steps only
    python run_reproducibility.py --steps baselines figures tables

    # Skip heavy spatiotemporal models
    python run_reproducibility.py --steps baselines patch multimodal figures tables

Available steps
---------------
    baselines       Tabular baselines (Ridge, RF, XGBoost, LightGBM, MLP, FT-Transformer)
    patch           Patch-based baselines (Patch-XGBoost, CNN-patch, ViT, CNN-Tabular)
    multimodal      Multimodal baselines (MOSAIKS, ERA5-LSTM, CMNet, TT-ERA5, TriModal)
    spatiotemporal  Full spatio-temporal models (ConvLSTM, TimeSformer, STGCN, DCRNN, ...)
    figures         All paper figures
    tables          All paper tables

Requirements
------------
    - Dataset downloaded to data/ (see README.md)
    - Python dependencies installed: pip install -r requirements.txt
    - GPU recommended for spatiotemporal step; CPU fallback is automatic
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALL_STEPS = ["baselines", "patch", "multimodal", "spatiotemporal", "figures", "tables"]

STEP_COMMANDS = {
    "baselines": [
        sys.executable, "-m", "src.models.baseline_models",
    ],
    "patch": [
        sys.executable, "-m", "src.models.patch_baseline",
    ],
    "multimodal": [
        sys.executable, "-m", "src.models.multimodal_baselines",
    ],
    "spatiotemporal": [
        sys.executable, "-m", "src.models.spatiotemporal_baselines",
    ],
    "figures": [
        sys.executable, "-c",
        ";".join([
            "from src.figures.coverage_map import main as f1; f1()",
            "from src.figures.era5_consistency import main as f2; f2()",
            "from src.figures.wq_trends import main as f3; f3()",
            "from src.figures.wqi_map import main as f4; f4()",
            "from src.figures.wqi_grid_analysis import main as f5; f5()",
            "from src.figures.plot_model_results import main as f6; f6()",
            "from src.figures.tier1_fidelity import main as f7; f7()",
            "from src.figures.tier3_spatial import main as f8; f8()",
        ]),
    ],
    "tables": [
        sys.executable, "-c",
        ";".join([
            "from src.evaluation.validation_tables import main as t1; t1()",
            "from src.tables.gen_tabular_best import main as t2; t2()",
            "from src.tables.gen_patch_table import main as t3; t3()",
            "from src.tables.gen_rq2_rq3_tables import main as t4; t4()",
            "from src.tables.gen_tabular_latex import main as t5; t5()",
        ]),
    ],
}

EXPECTED_OUTPUTS = {
    "baselines": [
        "data/outputs/baseline_model_results.csv",
        "data/outputs/baseline_conformal_results.csv",
        "data/outputs/baseline_rolling_origin.csv",
        "data/outputs/baseline_trophic_results.csv",
    ],
    "patch": [
        "data/outputs/patch_baseline_results.csv",
        "data/outputs/patch_conformal_results.csv",
    ],
    "multimodal": [
        "data/outputs/multimodal_baseline_results.csv",
        "data/outputs/multimodal_conformal_results.csv",
    ],
    "spatiotemporal": [
        "data/outputs/spatiotemporal_baseline_results.csv",
        "data/outputs/spatiotemporal_conformal_results.csv",
    ],
    "figures": [
        "data/outputs/figures/fig1_coverage_map.png",
        "data/outputs/figures/fig3_wqi_map.png",
        "data/outputs/figures/fig4_wq_trends.png",
    ],
    "tables": [
        "data/outputs/tables/table1_derived_parameter_statistics.csv",
        "data/outputs/tables/table7_wqi_grid_summary.csv",
    ],
}

REQUIRED_DATA = [
    "data/processed/nordic_multimodal_dataset/metadata.parquet",
    "data/processed/nordic_multimodal_dataset/split_index.parquet",
    "data/interim/SE/feature_table/feature_table_SE_2000.parquet",
    "data/outputs/nordic_water_quality_v1.nc",
]


def check_data() -> bool:
    log.info("Checking required dataset files …")
    missing = [p for p in REQUIRED_DATA if not Path(p).exists()]
    if missing:
        log.error("Missing dataset files — download from:")
        log.error("  https://ida.fairdata.fi/s/NOT-FOR-PUBLICATION-M3ZYWPq5Te8t")
        for p in missing:
            log.error("  ✗  %s", p)
        return False
    log.info("  All required data files present ✓")
    return True


def check_already_done(step: str) -> bool:
    outputs = EXPECTED_OUTPUTS.get(step, [])
    return all(Path(p).exists() for p in outputs)


def run_step(step: str, force: bool = False) -> bool:
    if not force and check_already_done(step):
        log.info("  [%s] already complete — skipping (use --force to rerun)", step)
        return True

    cmd = STEP_COMMANDS[step]
    log.info("  [%s] running …", step)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.error("  [%s] FAILED after %.0fs", step, elapsed)
        return False

    log.info("  [%s] done in %.0fs ✓", step, elapsed)
    return True


def main():
    ap = argparse.ArgumentParser(description="AquaNord-SE reproducibility runner")
    ap.add_argument(
        "--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS,
        help="Steps to run (default: all)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-run steps even if outputs already exist",
    )
    ap.add_argument(
        "--skip-data-check", action="store_true",
        help="Skip dataset presence check",
    )
    args = ap.parse_args()

    print()
    print("=" * 60)
    print("  AquaNord-SE  —  Reproducibility Runner")
    print("=" * 60)
    print()

    if not args.skip_data_check:
        if not check_data():
            sys.exit(1)

    Path("data/outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("data/outputs/tables").mkdir(parents=True, exist_ok=True)
    Path("data/outputs/results").mkdir(parents=True, exist_ok=True)

    failed = []
    for step in args.steps:
        print()
        log.info("━━━  Step: %s  ━━━", step.upper())
        ok = run_step(step, force=args.force)
        if not ok:
            failed.append(step)

    print()
    print("=" * 60)
    if failed:
        log.error("FAILED steps: %s", ", ".join(failed))
        log.error("Check logs above for details.")
        sys.exit(1)
    else:
        log.info("All steps completed successfully ✓")
        log.info("Outputs written to data/outputs/")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
