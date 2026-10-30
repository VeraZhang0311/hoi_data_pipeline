#!/usr/bin/env bash
#SBATCH --job-name=hoi-contact-label
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
set -euo pipefail
: "${BASE_PYTHON:?Must name verified Python executable}"
: "${WORKER_DRIVER:?Must name frozen worker driver}"
: "${FROZEN_INPUTS:?Must name frozen input recipes}"
: "${LABEL_OUTPUT:?Must name fresh derived output directory}"
: "${SOURCE:?Must name source}"
: "${SHARDS:?Must name shard count}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "$BASE_PYTHON" -I -B "$WORKER_DRIVER"
