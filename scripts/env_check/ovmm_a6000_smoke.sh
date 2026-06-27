#!/usr/bin/env bash
set -euo pipefail

CONTROL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$CONTROL_DIR/.." && pwd)"

HOME_ROBOT_ROOT="${HOME_ROBOT_ROOT:-$WORKSPACE_ROOT/benchmark/home-robot}"
MAMBA_BIN="${MAMBA_BIN:-/root/.local/bin/micromamba}"
CONDA_ENV="${CONDA_ENV:-home-robot}"
GPU_ID="${GPU_ID:-0}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-240}"
CREATE_DRI_NODES="${CREATE_DRI_NODES:-1}"

SHIM_DIR="${SHIM_DIR:-$CONTROL_DIR/shims}"
NVIDIA_OVERLAY_ROOT="${NVIDIA_OVERLAY_ROOT:-$WORKSPACE_ROOT/runtime_libs}"
NVIDIA_CUDA_OVERLAY="${NVIDIA_CUDA_OVERLAY:-$NVIDIA_OVERLAY_ROOT/nvidia57086_cuda}"
NVIDIA_GL_OVERLAY="${NVIDIA_GL_OVERLAY:-$NVIDIA_OVERLAY_ROOT/nvidia57086_gl/usr/lib/x86_64-linux-gnu}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-$NVIDIA_OVERLAY_ROOT/nvidia57086_gl/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

ensure_dev_dri_nodes() {
  if [[ "$CREATE_DRI_NODES" != "1" ]]; then
    return 0
  fi

  if [[ -e /dev/dri/renderD128 ]]; then
    return 0
  fi

  if [[ ! -d /sys/class/drm ]]; then
    echo "WARN: /sys/class/drm is unavailable; cannot create /dev/dri nodes." >&2
    return 0
  fi

  mkdir -p /dev/dri
  shopt -s nullglob
  for dev_file in /sys/class/drm/card*/dev /sys/class/drm/renderD*/dev; do
    local name
    local majmin
    local major
    local minor
    name="$(basename "$(dirname "$dev_file")")"
    majmin="$(cat "$dev_file")"
    major="${majmin%:*}"
    minor="${majmin#*:}"
    if [[ ! -e "/dev/dri/$name" ]]; then
      mknod -m 666 "/dev/dri/$name" c "$major" "$minor" || true
    else
      chmod 666 "/dev/dri/$name" || true
    fi
  done
  shopt -u nullglob
}

ensure_dev_dri_nodes

if [[ ! -x "$MAMBA_BIN" ]]; then
  echo "ERROR: micromamba not found or not executable: $MAMBA_BIN" >&2
  exit 2
fi

if [[ ! -d "$HOME_ROBOT_ROOT" ]]; then
  echo "ERROR: HOME_ROBOT_ROOT does not exist: $HOME_ROBOT_ROOT" >&2
  exit 2
fi

export HOME_ROBOT_ROOT
export PYTHONPATH="$SHIM_DIR:${PYTHONPATH:-}"

SYSTEM_GL_PRELOAD="/lib/x86_64-linux-gnu/libEGL.so.1:/lib/x86_64-linux-gnu/libGLdispatch.so.0:/lib/x86_64-linux-gnu/libOpenGL.so.0:/lib/x86_64-linux-gnu/libGLX.so.0"
export LD_PRELOAD="$SYSTEM_GL_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$NVIDIA_CUDA_OVERLAY:$NVIDIA_GL_OVERLAY:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

if [[ -f "$NVIDIA_EGL_VENDOR_JSON" ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR_JSON"
fi

cd "$HOME_ROBOT_ROOT"

timeout "$TIMEOUT_SECONDS" "$MAMBA_BIN" run -n "$CONDA_ENV" python projects/habitat_ovmm/eval_baselines_agent.py \
  --env_config_path projects/habitat_ovmm/configs/env/hssd_demo.yaml \
  --agent_type random \
  --num_episodes "$NUM_EPISODES" \
  habitat.environment.max_episode_steps="$MAX_EPISODE_STEPS" \
  habitat.simulator.habitat_sim_v0.gpu_device_id="$GPU_ID"
