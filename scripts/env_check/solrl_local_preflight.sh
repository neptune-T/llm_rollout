#!/usr/bin/env bash
set -euo pipefail

CONTROL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$CONTROL_DIR/.." && pwd)"
SANA_ROOT="${SANA_ROOT:-$WORKSPACE_ROOT/rltask/Sana}"
CONDA_ENV="${CONDA_ENV:-sana}"

echo "SANA_ROOT=$SANA_ROOT"
echo "CONDA_ENV=$CONDA_ENV"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "missing_conda_env=$CONDA_ENV"
  echo "Sana/Sol-RL full install is heavy: Python 3.11, torch cu128, xformers, mmcv, flash-attn."
  echo "Do not run environment_setup.sh locally unless a CUDA training stack is explicitly needed."
  exit 2
fi

conda run -n "$CONDA_ENV" python -c '
import importlib.util
import torch

print("python_env_ok=True")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for name in ["sana", "diffusers", "transformers", "xformers", "flash_attn"]:
    print(name, "found" if importlib.util.find_spec(name) else "missing")
'
