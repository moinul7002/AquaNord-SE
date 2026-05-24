# AquaNord-SE — Docker image for reproducibility
# -----------------------------------------------
# Build:  docker build -t aquanord-se .
# Run:    docker run --rm -v $(pwd)/data:/app/data aquanord-se
# GPU:    docker run --rm --gpus all -v $(pwd)/data:/app/data aquanord-se

FROM python:3.11-slim

LABEL description="AquaNord-SE reproducibility environment (CIKM 2026)"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgdal-dev \
        gdal-bin \
        libgeos-dev \
        libproj-dev \
        libhdf5-dev \
        libnetcdf-dev \
        git \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ─────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        # Geospatial stack requires pre-built wheels on slim images
        GDAL==$(gdal-config --version) \
 && pip install --no-cache-dir -r requirements.txt

# ── Source code ────────────────────────────────────────────────────────────────
COPY src/           ./src/
COPY run_reproducibility.py .
COPY LICENSE        .
COPY LICENSE_DATA.md .

# ── Output directories ─────────────────────────────────────────────────────────
# data/ is mounted at runtime via -v; pre-create output tree inside the image
# so the runner does not need write access to the mount root.
RUN mkdir -p data/outputs/figures \
             data/outputs/tables  \
             data/outputs/results

# ── Entry point ────────────────────────────────────────────────────────────────
# Default: run full reproducibility pipeline.
# Override steps via CMD, e.g.:
#   docker run ... aquanord-se --steps baselines figures tables
ENTRYPOINT ["python", "run_reproducibility.py"]
CMD []
