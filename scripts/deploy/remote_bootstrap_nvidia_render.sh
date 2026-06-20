#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
LOG_DIR="${ROOT}/logs/deploy"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/nvidia_render_bootstrap_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[nvidia-render-bootstrap] root: $ROOT"
echo "[nvidia-render-bootstrap] log: $LOG_FILE"

bash scripts/deploy/remote_preflight_nvidia_render.sh

INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
RUN_B1K_SETUP="${RUN_B1K_SETUP:-0}"
CHECK_DOCKER_FOR_ISAAC="${CHECK_DOCKER_FOR_ISAAC:-1}"

if [ "$INSTALL_SYSTEM_DEPS" = "1" ]; then
  echo "[nvidia-render-bootstrap] installing system deps"
  SUDO=""
  if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    SUDO="sudo"
  fi
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update
  $SUDO apt-get install -y git git-lfs curl rsync build-essential python3-venv python3-pip ffmpeg htop
  git lfs install || true
else
  echo "[nvidia-render-bootstrap] INSTALL_SYSTEM_DEPS=0; skipping apt installs"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[nvidia-render-bootstrap] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ "$RUN_B1K_SETUP" = "1" ]; then
  echo "[nvidia-render-bootstrap] running BEHAVIOR champion setup_remote.sh"
  cd "$ROOT/basecode/behavior-1k-solution"
  bash setup_remote.sh
  cd "$ROOT"
else
  echo "[nvidia-render-bootstrap] RUN_B1K_SETUP=0; skipping champion setup"
fi

if [ "$CHECK_DOCKER_FOR_ISAAC" = "1" ]; then
  if command -v docker >/dev/null 2>&1; then
    echo "[nvidia-render-bootstrap] checking docker NVIDIA runtime"
    docker run --rm --gpus all nvcr.io/nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi || \
      echo "WARNING: docker GPU smoke test failed. Fix NVIDIA container toolkit before Isaac Sim."
  else
    echo "WARNING: docker missing; install Docker + NVIDIA container toolkit for Isaac Sim containers."
  fi
fi

echo "[nvidia-render-bootstrap] done"
