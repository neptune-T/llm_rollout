#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/deploy/lib.sh
source "${SCRIPT_DIR}/lib.sh"

CONFIG_PATH="${1:-}"
load_deploy_config "$CONFIG_PATH"
ROOT="$(repo_root)"

build_rsync_args() {
  local -n out_args=$1
  out_args=(-az --human-readable --info=progress2)
  if [ "${RSYNC_DELETE}" = "1" ]; then
    out_args+=(--delete)
  fi

  out_args+=(
    --exclude='.cache/'
    --exclude='.hf-cache/'
    --exclude='**/__pycache__/'
    --exclude='**/*.pyc'
    --exclude='.venv/'
    --exclude='venv/'
    --exclude='node_modules/'
    --exclude='wandb/'
    --exclude='outputs/'
    --exclude='output/'
    --exclude='lmm_rollout_project/logs/errors/'
    --exclude='lmm_rollout_project/results/videos/'
  )

  if [ "${SYNC_CHECKPOINTS}" != "1" ]; then
    out_args+=(--exclude='basecode/behavior-1k-solution/checkpoints/')
  fi
  if [ "${SYNC_DATASETS}" != "1" ]; then
    out_args+=(--exclude='**/data/' --exclude='**/datasets/')
  fi
  if [ "${SYNC_OUTPUTS}" != "1" ]; then
    out_args+=(--exclude='lmm_rollout_project/experiments/*/raw/' --exclude='lmm_rollout_project/results/videos/')
  fi
}

sync_one() {
  local host="$1"
  local remote_root="$2"
  local args=()
  build_rsync_args args

  info "Creating remote root on ${host}:${remote_root}"
  ssh_remote "$host" "mkdir -p '$remote_root'"

  info "Syncing code to ${host}:${remote_root}"
  # shellcheck disable=SC2086
  rsync "${args[@]}" -e "ssh ${SSH_OPTS}" "$ROOT/" "${host}:${remote_root}/"
}

sync_one "$NVIDIA_RENDER_HOST" "$NVIDIA_RENDER_ROOT"

if [ "$AMD_TRAIN_CONNECT_MODE" = "ssh" ]; then
  sync_one "$AMD_TRAIN_HOST" "$AMD_TRAIN_ROOT"
else
  info "AMD_TRAIN_CONNECT_MODE=tunnel: skipping rsync to AMD"
  info "Create a tunnel package with: bash lmm_rollout_project/scripts/deploy/package_for_tunnel.sh $CONFIG_PATH"
  info "Upload the tarball through VS Code Tunnel and extract it on AMD."
fi

info "Code sync complete"
