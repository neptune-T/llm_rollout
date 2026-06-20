#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/deploy/lib.sh
source "${SCRIPT_DIR}/lib.sh"

CONFIG_PATH="${1:-}"
load_deploy_config "$CONFIG_PATH"
ROOT="$(repo_root)"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/tmp/deploy_packages"
mkdir -p "$OUT_DIR"
OUT="${OUT_DIR}/longhorizon_tunnel_${STAMP}.tar.gz"

EXCLUDES=(
  --exclude='./.git'
  --exclude='./.agents'
  --exclude='./.codex'
  --exclude='./.cache'
  --exclude='./**/.cache'
  --exclude='./**/__pycache__'
  --exclude='./**/*.pyc'
  --exclude='./**/.venv'
  --exclude='./**/venv'
  --exclude='./**/node_modules'
  --exclude='./**/wandb'
  --exclude='./**/outputs'
  --exclude='./**/output'
  --exclude='./lmm_rollout_project/results/videos'
  --exclude='./lmm_rollout_project/logs/errors'
  --exclude='./tmp'
)

if [ "${SYNC_CHECKPOINTS}" != "1" ]; then
  EXCLUDES+=(--exclude='./basecode/behavior-1k-solution/checkpoints')
fi
if [ "${SYNC_DATASETS}" != "1" ]; then
  EXCLUDES+=(--exclude='./**/data' --exclude='./**/datasets')
fi

info "Creating tunnel package: $OUT"
(
  cd "$ROOT"
  tar "${EXCLUDES[@]}" -czf "$OUT" .
)

cat <<EOF
Created: $OUT

AMD tunnel workflow:
1. Open AMD through VS Code Tunnel.
2. Upload this tarball to the AMD server.
3. In the AMD terminal:

   mkdir -p '${AMD_TRAIN_ROOT}'
   tar -xzf /path/to/$(basename "$OUT") -C '${AMD_TRAIN_ROOT}'
   cd '${AMD_TRAIN_ROOT}'
   bash lmm_rollout_project/scripts/deploy/remote_preflight_amd_train.sh

For an interactive full node allocation:

   ${AMD_SLURM_ALLOC_CMD}

EOF
