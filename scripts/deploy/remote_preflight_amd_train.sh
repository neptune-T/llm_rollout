#!/usr/bin/env bash
set -euo pipefail

echo "[amd-train] host: $(hostname)"
echo "[amd-train] cwd: $(pwd)"
echo "[amd-train] date: $(date -Iseconds)"
echo "[amd-train] SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"

if command -v sinfo >/dev/null 2>&1; then
  echo "[amd-train] slurm sinfo snapshot:"
  sinfo -N -o '%N %t %G %m %c' | head -20 || true
else
  echo "WARNING: sinfo not found. Slurm may not be available in this shell." >&2
fi

if command -v scontrol >/dev/null 2>&1; then
  RES="${AMD_SLURM_RESERVATION:-gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation}"
  echo "[amd-train] reservation snapshot for ${RES}:"
  scontrol show reservation "$RES" || true
fi

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi || true
else
  echo "WARNING: rocm-smi not found. ROCm may not be installed or not in PATH." >&2
fi

echo "[amd-train] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
echo "[amd-train] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"

if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | grep -E 'Name:|Marketing Name' | head -40 || true
else
  echo "WARNING: rocminfo not found." >&2
fi

python3 - <<'PY' || true
try:
    import torch
    print("[amd-train] torch:", torch.__version__)
    print("[amd-train] torch.version.hip:", getattr(torch.version, "hip", None))
    print("[amd-train] cuda/hip available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[amd-train] device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("WARNING: torch check failed:", repr(exc))
PY

if [ -d rltask/Sana ]; then
  echo "[amd-train] found Sana/Sol-RL repo"
else
  echo "ERROR: missing rltask/Sana under $(pwd)" >&2
  exit 1
fi

echo "[amd-train] preflight complete"
