#!/usr/bin/env bash
set -euo pipefail

echo "[nvidia-render] host: $(hostname)"
echo "[nvidia-render] cwd: $(pwd)"
echo "[nvidia-render] date: $(date -Iseconds)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install NVIDIA driver first." >&2
  exit 1
fi

nvidia-smi

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"
echo "[nvidia-render] GPUs: ${GPU_NAMES}"
if echo "$GPU_NAMES" | grep -Eiq 'A100|A800|H100|H800|MI[0-9]'; then
  echo "WARNING: This GPU name often indicates no RT cores or non-NVIDIA hardware. Isaac Sim rendering may fail." >&2
fi

for cmd in git git-lfs python3 bash; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "[nvidia-render] found $cmd: $(command -v "$cmd")"
  else
    echo "WARNING: missing $cmd" >&2
  fi
done

if command -v docker >/dev/null 2>&1; then
  echo "[nvidia-render] docker: $(docker --version)"
  if docker info >/dev/null 2>&1; then
    echo "[nvidia-render] docker daemon reachable"
  else
    echo "WARNING: docker command exists but daemon is not reachable for this user" >&2
  fi
else
  echo "WARNING: docker not found; Isaac Sim container workflow will not work" >&2
fi

if [ -d basecode/behavior-1k-solution ]; then
  echo "[nvidia-render] found champion repo"
else
  echo "ERROR: missing basecode/behavior-1k-solution under $(pwd)" >&2
  exit 1
fi

echo "[nvidia-render] preflight complete"
