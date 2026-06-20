#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "[deploy] $*" >&2
}

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/../../.." && pwd
}

load_deploy_config() {
  local config_path="${1:-}"
  [ -n "$config_path" ] || die "Usage: $0 configs/deploy/lmm_deploy.env"
  [ -f "$config_path" ] || die "Missing deploy config: $config_path"
  # shellcheck disable=SC1090
  source "$config_path"

  : "${NVIDIA_RENDER_HOST:?Set NVIDIA_RENDER_HOST in $config_path}"
  : "${NVIDIA_RENDER_ROOT:?Set NVIDIA_RENDER_ROOT in $config_path}"
  : "${AMD_TRAIN_ROOT:?Set AMD_TRAIN_ROOT in $config_path}"

  SSH_OPTS="${SSH_OPTS:-}"
  AMD_TRAIN_CONNECT_MODE="${AMD_TRAIN_CONNECT_MODE:-ssh}"
  if [ "$AMD_TRAIN_CONNECT_MODE" = "ssh" ]; then
    : "${AMD_TRAIN_HOST:?Set AMD_TRAIN_HOST in $config_path or use AMD_TRAIN_CONNECT_MODE=tunnel}"
  elif [ "$AMD_TRAIN_CONNECT_MODE" = "tunnel" ]; then
    AMD_TRAIN_HOST="${AMD_TRAIN_HOST:-}"
  else
    die "AMD_TRAIN_CONNECT_MODE must be ssh or tunnel, got: $AMD_TRAIN_CONNECT_MODE"
  fi
  RSYNC_DELETE="${RSYNC_DELETE:-0}"
  SYNC_CHECKPOINTS="${SYNC_CHECKPOINTS:-0}"
  SYNC_DATASETS="${SYNC_DATASETS:-0}"
  SYNC_OUTPUTS="${SYNC_OUTPUTS:-0}"
  DEPLOY_BOOTSTRAP="${DEPLOY_BOOTSTRAP:-0}"
  AMD_SLURM_RESERVATION="${AMD_SLURM_RESERVATION:-gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation}"
  AMD_SLURM_ALLOC_CMD="${AMD_SLURM_ALLOC_CMD:-salloc --reservation=${AMD_SLURM_RESERVATION} --exclusive --mem=0}"
}

ssh_remote() {
  local host="$1"
  shift
  # Intentionally allow word splitting for SSH_OPTS.
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "$host" "$@"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}
