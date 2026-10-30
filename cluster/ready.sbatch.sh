#!/usr/bin/env bash
#SBATCH --job-name=hoi-contact-input-ready
#SBATCH --partition=liu-compute
#SBATCH --account=liu-account
#SBATCH --qos=ll-med
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:30:00
set -euo pipefail
: "${BASE_PYTHON:?}"
: "${WORKER_DRIVER:?}"
: "${FROZEN_INPUTS:?}"
: "${SOURCE:?}"
export WORKER_MODE=gate PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "$BASE_PYTHON" -I -B "$WORKER_DRIVER"
