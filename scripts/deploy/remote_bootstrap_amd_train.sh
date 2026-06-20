#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
LOG_DIR="${ROOT}/logs/deploy"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/amd_train_bootstrap_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[amd-train-bootstrap] root: $ROOT"
echo "[amd-train-bootstrap] log: $LOG_FILE"

bash scripts/deploy/remote_preflight_amd_train.sh

RUN_ROCM_TORCH_INSTALL="${RUN_ROCM_TORCH_INSTALL:-0}"
ROCM_TORCH_INDEX="${ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}"
CONDA_ENV_LMM="${CONDA_ENV_LMM:-lmm}"

if [ "$RUN_ROCM_TORCH_INSTALL" != "1" ]; then
  echo "[amd-train-bootstrap] RUN_ROCM_TORCH_INSTALL=0; skipping Python env install"
  echo "[amd-train-bootstrap] Note: Sana's bundled environment_setup.sh is CUDA-oriented and should not be used as-is on AMD."
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda/Mambaforge first or set up an existing Python 3.11 env." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_LMM"; then
  echo "[amd-train-bootstrap] reusing conda env: $CONDA_ENV_LMM"
else
  echo "[amd-train-bootstrap] creating conda env: $CONDA_ENV_LMM"
  conda create -n "$CONDA_ENV_LMM" python=3.11 -y
fi
conda activate "$CONDA_ENV_LMM"

python -m pip install -U pip wheel setuptools
python -m pip install torch torchvision torchaudio --index-url "$ROCM_TORCH_INDEX"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "[amd-train-bootstrap] ROCm torch env created. Install project-specific deps only after compatibility review."
