#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/deploy/lib.sh
source "${SCRIPT_DIR}/lib.sh"

CONFIG_PATH="${1:-}"
load_deploy_config "$CONFIG_PATH"
ROOT="$(repo_root)"

info "Local repo root: $ROOT"

require_cmd ssh
require_cmd rsync
require_cmd git
require_cmd bash

[ -d "$ROOT/basecode/behavior-1k-solution" ] || die "Missing BEHAVIOR champion repo"
[ -d "$ROOT/rltask/Sana" ] || die "Missing Sana/Sol-RL repo"
[ -f "$ROOT/lmm_rollout_project/memory/project_memory.md" ] || die "Missing project memory"
[ -f "$ROOT/lmm_rollout_project/docs/current_plan.md" ] || die "Missing current plan"

info "Checking subrepo status"
git -C "$ROOT/Planning" status --short || true
git -C "$ROOT/basecode/behavior-1k-solution" status --short || true
git -C "$ROOT/rltask/Sana" status --short || true

if [ "${SYNC_CHECKPOINTS}" = "1" ]; then
  [ -d "$ROOT/basecode/behavior-1k-solution/checkpoints" ] || die "SYNC_CHECKPOINTS=1 but checkpoints dir missing"
  info "SYNC_CHECKPOINTS=1: checkpoint sync may transfer tens of GB"
else
  info "SYNC_CHECKPOINTS=0: BEHAVIOR checkpoints will be excluded"
fi

info "Checking SSH reachability"
ssh_remote "$NVIDIA_RENDER_HOST" "printf 'nvidia host ok: '; hostname"
if [ "$AMD_TRAIN_CONNECT_MODE" = "ssh" ]; then
  ssh_remote "$AMD_TRAIN_HOST" "printf 'amd host ok: '; hostname"
else
  info "AMD_TRAIN_CONNECT_MODE=tunnel: skipping AMD SSH reachability check"
  info "Use VS Code Tunnel to open the AMD server, then run remote commands from its terminal."
fi

info "Local preflight complete"
