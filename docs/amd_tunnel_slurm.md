# AMD MI325X Tunnel + Slurm Notes

Last updated: 2026-06-20

## Access Mode

The AMD server is reached through VS Code Tunnel / VS Code for the Web, not normal SSH from this workspace.

Do not store account passwords or verification codes in repository files. Login should be done manually through Microsoft auth in VS Code. If credentials were pasted into chat or logs, rotate them if there is any risk they were exposed outside the intended private session.

## Machine Role

This machine is for large-scale training, calibration, analysis, and ROCm-compatible experiments.

It is not for Isaac Sim rendering:

- AMD GPUs do not provide CUDA.
- BEHAVIOR / Isaac Sim rendering should use the NVIDIA RT-core render server.

## Hardware

Expected node configuration:

- AMD Instinct MI325X.
- 8 GPUs per full node.
- ROCm / amdgpu ecosystem.
- Large HBM memory per GPU.
- Large CPU count, around 256 cores per node.

## Slurm Allocation

Reserved full node command:

```bash
salloc --reservation=gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation --exclusive --mem=0
```

Use 4-8 GPUs for normal work. Do not request more than 8 GPUs unless explicitly approved; jobs using more than 8 GPUs may be killed by admins.

Convenience script after this repo is on AMD:

```bash
cd /path/to/longhorizon
bash scripts/deploy/remote_amd_slurm_fullnode.sh
```

Batch preflight after repo is on AMD:

```bash
cd /path/to/longhorizon
mkdir -p logs/deploy
sbatch --reservation=gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation \
  scripts/deploy/remote_amd_slurm_preflight.sbatch
```

## Code Transfer Without SSH

Because AMD is tunnel-only, local `rsync` cannot push directly to it.

Create a local package:

```bash
bash scripts/deploy/package_for_tunnel.sh configs/deploy/lmm_deploy.env
```

Then in VS Code Tunnel:

1. Upload the produced tarball from `tmp/deploy_packages/`.
2. Open an AMD terminal.
3. Extract:

```bash
mkdir -p /scratch/$USER/longhorizon
tar -xzf /path/to/longhorizon_tunnel_YYYYMMDD_HHMMSS.tar.gz -C /scratch/$USER/longhorizon
cd /scratch/$USER/longhorizon
bash scripts/deploy/remote_preflight_amd_train.sh
```

## Environment Policy

Do not run `rltask/Sana/environment_setup.sh` on AMD as-is. It installs CUDA 12.8 PyTorch wheels and CUDA-oriented packages.

First run:

```bash
bash scripts/deploy/remote_preflight_amd_train.sh
```

Then decide the ROCm environment path based on actual `rocm-smi`, `rocminfo`, and PyTorch HIP availability.

Optional ROCm torch bootstrap, only after confirmation:

```bash
RUN_ROCM_TORCH_INSTALL=1 bash scripts/deploy/remote_bootstrap_amd_train.sh
```

## First Validation Checklist

- `sinfo` works.
- Reservation is visible through `scontrol show reservation`.
- `rocm-smi` shows 8 MI325X GPUs on allocated node.
- `torch.version.hip` is non-empty in the intended Python env.
- `torch.cuda.is_available()` returns true under ROCm PyTorch.
- A small tensor matmul runs on one GPU.
- Multi-GPU visibility is correct under Slurm.
