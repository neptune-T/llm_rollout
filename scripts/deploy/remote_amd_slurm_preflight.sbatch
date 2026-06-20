#!/usr/bin/env bash
#SBATCH --job-name=lmm-amd-preflight
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=logs/deploy/amd_preflight_%j.out
#SBATCH --error=logs/deploy/amd_preflight_%j.err

set -euo pipefail

echo "[amd-slurm-preflight] SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "[amd-slurm-preflight] host: $(hostname)"
echo "[amd-slurm-preflight] cwd: $(pwd)"

bash scripts/deploy/remote_preflight_amd_train.sh
