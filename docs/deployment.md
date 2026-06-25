# Deployment Plan: Local -> NVIDIA Render Server + AMD Training Server

Last updated: 2026-06-25

## Goal

Make this workspace easy to deploy from the local machine to two remote roles:

- NVIDIA render server: BEHAVIOR / OmniGibson / Isaac Sim headless rendering and evaluator rollouts. This machine needs an NVIDIA GPU with RT cores and stable driver/container support.
- AMD training server: training, calibration, analysis, and Sol-RL-inspired online RL code. This machine is accessed through VS Code Tunnel rather than SSH. It is for large compute, but compatibility must be validated because the current champion baseline and Sana setup are CUDA-oriented.

The active project remains champion-solution-first. OpenVLA and J-EPA/V-JEPA are not part of the current mainline.

## Current NVIDIA Render Server

The active NVIDIA render server is:

```bash
ssh -p 20400 root@219.223.207.18
cd /root/workspace/tianshanzhang
```

VS Code Remote SSH target path:

```text
/root/workspace/tianshanzhang
```

Confirmed on 2026-06-25:

- GPU: 4x NVIDIA RTX A6000, 49140 MiB each.
- Driver: 570.86.10.
- Remote conda: `/root/anaconda3`.
- System Python: 3.8.2.
- `git-lfs`: installed and initialized.
- Workspace ownership: `root:root`.
- First rsync completed with exit code 0.
- NVIDIA preflight script passed.
- Docker: not installed.

Synced content:

- `lmm_rollout_project`: project docs, memory, configs, scripts.
- `rltask/Sana`: Sol-RL/Sana reference code.
- `benchmark/home-robot`: OVMM/HomeRobot source code, excluding `data/`.
- `benchmark/BEHAVIOR-1K`: BEHAVIOR code/resources.
- `basecode/behavior-1k-solution`: champion solution code/resources, excluding checkpoints and local heavy data.

Not yet present on A6000:

- `benchmark/home-robot/data`.
- `basecode/behavior-1k-solution/checkpoints`.
- `basecode/behavior-1k-solution/BEHAVIOR-1K-684a`.
- Project-specific conda envs such as `home-robot` and `sana`.

## Important Architecture Constraint

Do not assume the AMD server can run the BEHAVIOR champion policy unchanged.

- `basecode/behavior-1k-solution` depends on `jax[cuda12]` in `pyproject.toml`.
- `rltask/Sana/environment_setup.sh` installs CUDA 12.8 PyTorch wheels and CUDA-oriented packages.
- AMD/ROCm training needs a separate compatibility pass. The deploy scripts can sync code and run ROCm preflight, but they do not silently convert CUDA environments to ROCm.

Practical first topology:

```text
local laptop
  edits code and launches deploy

NVIDIA render server
  runs Isaac Sim / BEHAVIOR evaluator
  may also run champion policy server if GPU/RAM are sufficient

AMD training server
  stores code, logs, rollout datasets
  runs ROCm-compatible training/calibration after validation
  is reached through VS Code Tunnel, so local rsync/ssh cannot push directly unless SSH is later enabled
```

If the NVIDIA render server is too small for champion checkpoint loading, use a third CUDA model server or test whether the AMD server can run the policy after a ROCm/JAX port. Do not assume this will work.

## One-Time Setup

Create a local deploy config:

```bash
cp lmm_rollout_project/configs/deploy/lmm_deploy.example.env lmm_rollout_project/configs/deploy/lmm_deploy.env
```

Edit:

```bash
NVIDIA_RENDER_HOST="your-nvidia-ssh-alias"
AMD_TRAIN_HOST="your-amd-ssh-alias"
AMD_TRAIN_CONNECT_MODE="tunnel"
NVIDIA_RENDER_ROOT="/remote/path/longhorizon"
AMD_TRAIN_ROOT="/remote/path/longhorizon"
```

Run a safe dry deploy:

```bash
bash lmm_rollout_project/scripts/deploy/deploy_all.sh lmm_rollout_project/configs/deploy/lmm_deploy.env
```

This performs local preflight, rsync code sync to the NVIDIA host, creates an AMD tunnel upload package, and runs NVIDIA remote preflight. If `AMD_TRAIN_CONNECT_MODE=tunnel`, AMD SSH sync/preflight is skipped and the script prints the command to run inside the AMD VS Code Tunnel terminal. It does not run heavy setup unless `DEPLOY_BOOTSTRAP=1`.

## AMD Tunnel Workflow

For AMD, create a package locally:

```bash
bash lmm_rollout_project/scripts/deploy/package_for_tunnel.sh lmm_rollout_project/configs/deploy/lmm_deploy.env
```

Upload the generated tarball through VS Code Tunnel, then run on AMD:

```bash
mkdir -p "$AMD_TRAIN_ROOT"
tar -xzf /path/to/longhorizon_tunnel_YYYYMMDD_HHMMSS.tar.gz -C "$AMD_TRAIN_ROOT"
cd "$AMD_TRAIN_ROOT"
bash lmm_rollout_project/scripts/deploy/remote_preflight_amd_train.sh
```

See `lmm_rollout_project/docs/amd_tunnel_slurm.md` for the MI325X Slurm workflow.

## First Full Bootstrap

After the safe preflight looks correct, enable bootstrap explicitly in `lmm_rollout_project/configs/deploy/lmm_deploy.env`:

```bash
DEPLOY_BOOTSTRAP=1
```

For the NVIDIA render server, enable only what you want:

```bash
INSTALL_SYSTEM_DEPS=1
RUN_B1K_SETUP=1
CHECK_DOCKER_FOR_ISAAC=1
```

Then run:

```bash
bash lmm_rollout_project/scripts/deploy/deploy_all.sh lmm_rollout_project/configs/deploy/lmm_deploy.env
```

## Heavy Artifacts

By default, rsync excludes heavy artifacts:

- BEHAVIOR checkpoints
- datasets
- outputs
- wandb runs
- videos
- Python virtualenvs

To sync local champion checkpoints, set:

```bash
SYNC_CHECKPOINTS=1
```

Use this only when the network/storage budget is acceptable. The local champion checkpoints are about 48GB.

For the current A6000 server, prefer these next steps instead of blindly syncing all heavy artifacts from the laptop:

1. Create project conda envs on A6000.
2. Download OVMM/HSSD data on A6000 using remote bandwidth and HF mirror where possible.
3. Transfer BEHAVIOR checkpoints only after confirming the policy server and simulator plan.

## Expected First Smoke Tests

On NVIDIA render server:

```bash
cd "$NVIDIA_RENDER_ROOT"
bash lmm_rollout_project/scripts/deploy/remote_preflight_nvidia_render.sh
```

Then, after BEHAVIOR/Isaac is installed, run a minimal headless render/reset/step test. That test still needs to be written after confirming the exact BEHAVIOR install path.

On AMD training server:

```bash
cd "$AMD_TRAIN_ROOT"
bash lmm_rollout_project/scripts/deploy/remote_preflight_amd_train.sh
```

Do not run Sana CUDA setup on AMD unless the ROCm plan is explicit.

## Failure Policy

- If `nvidia-smi` fails on the render server, stop and fix driver/container runtime first.
- If the render GPU is A100/H100/A800 without RT cores, do not assume Isaac Sim rendering is valid.
- If AMD PyTorch/ROCm is not visible, do not start training.
- If champion checkpoint loading fails on a remote machine, record RAM, VRAM, JAX devices, and exact exit code in `lmm_rollout_project/logs/daily/YYYY-MM-DD.md`.
- If Git reports `dubious ownership` after rsync to a root-owned remote workspace, fix ownership of the target tree or add explicit safe directories. On the current A6000 server this was fixed with `chown -R root:root /root/workspace/tianshanzhang`.
- If HomeRobot shows many asset files as modified, check `git-lfs` first. On A6000 this was fixed by installing `git-lfs`.
- If using Isaac Sim containers on A6000, install and validate docker first. If starting with OVMM/HomeRobot, docker may not be needed for the first Habitat rendering smoke.

## Commands Added

```bash
bash lmm_rollout_project/scripts/deploy/preflight_local.sh lmm_rollout_project/configs/deploy/lmm_deploy.env
bash lmm_rollout_project/scripts/deploy/sync_code.sh lmm_rollout_project/configs/deploy/lmm_deploy.env
bash lmm_rollout_project/scripts/deploy/deploy_all.sh lmm_rollout_project/configs/deploy/lmm_deploy.env
```

Remote scripts synced with the repo:

```bash
bash lmm_rollout_project/scripts/deploy/remote_preflight_nvidia_render.sh
bash lmm_rollout_project/scripts/deploy/remote_bootstrap_nvidia_render.sh
bash lmm_rollout_project/scripts/deploy/remote_preflight_amd_train.sh
bash lmm_rollout_project/scripts/deploy/remote_bootstrap_amd_train.sh
```
