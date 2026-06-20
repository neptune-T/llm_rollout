#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/deploy/lib.sh
source "${SCRIPT_DIR}/lib.sh"

CONFIG_PATH="${1:-}"
load_deploy_config "$CONFIG_PATH"

info "Phase 1/4: local preflight"
bash "${SCRIPT_DIR}/preflight_local.sh" "$CONFIG_PATH"

info "Phase 2/4: sync code to both remotes"
bash "${SCRIPT_DIR}/sync_code.sh" "$CONFIG_PATH"
if [ "$AMD_TRAIN_CONNECT_MODE" = "tunnel" ]; then
  info "AMD tunnel mode: creating upload package"
  bash "${SCRIPT_DIR}/package_for_tunnel.sh" "$CONFIG_PATH"
fi

info "Phase 3/4: remote preflight"
ssh_remote "$NVIDIA_RENDER_HOST" "cd '$NVIDIA_RENDER_ROOT' && bash lmm_rollout_project/scripts/deploy/remote_preflight_nvidia_render.sh"
if [ "$AMD_TRAIN_CONNECT_MODE" = "ssh" ]; then
  ssh_remote "$AMD_TRAIN_HOST" "cd '$AMD_TRAIN_ROOT' && bash lmm_rollout_project/scripts/deploy/remote_preflight_amd_train.sh"
else
  info "AMD_TRAIN_CONNECT_MODE=tunnel: skipping AMD remote preflight from local"
  info "Run on AMD after uploading/extracting package:"
  info "  cd '$AMD_TRAIN_ROOT' && bash lmm_rollout_project/scripts/deploy/remote_preflight_amd_train.sh"
fi

if [ "${DEPLOY_BOOTSTRAP}" = "1" ]; then
  info "Phase 4/4: remote bootstrap"
  ssh_remote "$NVIDIA_RENDER_HOST" "cd '$NVIDIA_RENDER_ROOT' && INSTALL_SYSTEM_DEPS='${INSTALL_SYSTEM_DEPS:-0}' RUN_B1K_SETUP='${RUN_B1K_SETUP:-0}' CHECK_DOCKER_FOR_ISAAC='${CHECK_DOCKER_FOR_ISAAC:-1}' bash lmm_rollout_project/scripts/deploy/remote_bootstrap_nvidia_render.sh"
  if [ "$AMD_TRAIN_CONNECT_MODE" = "ssh" ]; then
    ssh_remote "$AMD_TRAIN_HOST" "cd '$AMD_TRAIN_ROOT' && RUN_ROCM_TORCH_INSTALL='${RUN_ROCM_TORCH_INSTALL:-0}' ROCM_TORCH_INDEX='${ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}' CONDA_ENV_LMM='${CONDA_ENV_LMM:-lmm}' bash lmm_rollout_project/scripts/deploy/remote_bootstrap_amd_train.sh"
  else
    info "AMD_TRAIN_CONNECT_MODE=tunnel: skipping AMD bootstrap from local"
    info "Run on AMD if desired:"
    info "  cd '$AMD_TRAIN_ROOT' && RUN_ROCM_TORCH_INSTALL='${RUN_ROCM_TORCH_INSTALL:-0}' ROCM_TORCH_INDEX='${ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}' CONDA_ENV_LMM='${CONDA_ENV_LMM:-lmm}' bash lmm_rollout_project/scripts/deploy/remote_bootstrap_amd_train.sh"
  fi
else
  info "Phase 4/4: DEPLOY_BOOTSTRAP=0; skipping remote bootstrap"
fi

info "Deployment flow complete"
