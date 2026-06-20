#!/usr/bin/env bash
set -euo pipefail

RESERVATION="${AMD_SLURM_RESERVATION:-gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation}"

echo "[amd-slurm] host: $(hostname)"
echo "[amd-slurm] reservation: ${RESERVATION}"

if ! command -v salloc >/dev/null 2>&1; then
  echo "ERROR: salloc not found. Are you on the AMD cluster login node?" >&2
  exit 1
fi

echo "[amd-slurm] requesting exclusive full node with all memory"
echo "[amd-slurm] command: salloc --reservation=${RESERVATION} --exclusive --mem=0"
exec salloc --reservation="${RESERVATION}" --exclusive --mem=0
