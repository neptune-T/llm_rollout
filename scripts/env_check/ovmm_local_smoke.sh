#!/usr/bin/env bash
set -euo pipefail

CONTROL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$CONTROL_DIR/.." && pwd)"
HOME_ROBOT_ROOT="${HOME_ROBOT_ROOT:-$WORKSPACE_ROOT/benchmark/home-robot}"
CONDA_ENV="${CONDA_ENV:-home-robot}"
GPU_ID="${GPU_ID:-0}"
NUM_EPISODES="${NUM_EPISODES:-1}"
FORCE_STEP="${FORCE_STEP:-1}"

export HOME_ROBOT_ROOT
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CONTROL_DIR/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CONTROL_DIR/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cd "$HOME_ROBOT_ROOT"

conda run -n "$CONDA_ENV" python projects/habitat_ovmm/eval_baselines_agent.py \
  --env_config_path projects/habitat_ovmm/configs/env/hssd_demo.yaml \
  --agent_type random \
  --num_episodes "$NUM_EPISODES" \
  --force_step "$FORCE_STEP" \
  habitat.simulator.habitat_sim_v0.gpu_device_id="$GPU_ID"
