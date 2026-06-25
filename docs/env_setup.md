# Environment Setup Notes

Last updated: 2026-06-23

## Current Direction

The immediate Stage 1 target is OVMM first, because it is simpler to bring up than BEHAVIOR/Isaac and can provide an earlier mobile-manipulation rollout loop.

BEHAVIOR champion solution remains the stronger long-horizon target for the paper, but it is not the first local smoke target.

## Local Machine

- Machine role: local code editing, documentation, small imports, and short metadata checks.
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8GB VRAM.
- Driver: 580.126.09.
- CUDA reported by `nvidia-smi`: 13.0.
- Disk under workspace: about 93GB free when checked.
- RAM: 15GB total.
- Resource policy: do not run long `git lfs pull`, CUDA extension builds such as `flash-attn`, Habitat/OVMM rendering smoke tests, BEHAVIOR checkpoint loading, or large package installs locally unless explicitly approved. The laptop has frozen/rebooted during heavy setup.

## OVMM / HomeRobot

Repository:

```text
/home/plote/longhorizon/benchmark/home-robot
```

Local conda environment:

```text
home-robot
```

Confirmed imports:

- Python 3.9.23.
- `torch==1.13.1`, CUDA 11.7 build.
- Host execution sees CUDA and the RTX 3070 GPU.
- `habitat_sim==0.2.5` imports.
- `habitat` imports.
- `home_robot` imports from editable local path.
- `home_robot_sim` imports from editable local path.
- `habitat-lab==0.2.5` and `habitat-baselines==0.2.5` are editable installs from `src/third_party/habitat-lab`.

Data status:

- Stretch robot model exists under `data/robots/hab_stretch`.
- OVMM episodes were downloaded to `data/datasets/ovmm`.
- OVMM episodes were pinned to commit `9ad25fbd86a3fd352c7a0fc1f99132fbb5802378`.
- HSSD scenes under `data/hssd-hab` were partially downloaded but are not verified complete.
- OVMM object assets under `data/objects` were downloaded through Git LFS and appeared clean when checked.
- HuggingFace CLI reported not logged in; HSSD scenes may require accepting the dataset license and logging in.

Smoke command, only with explicit approval because local rendering has caused instability:

```bash
bash lmm_rollout_project/scripts/env_check/ovmm_local_smoke.sh
```

Equivalent command:

```bash
cd /home/plote/longhorizon/benchmark/home-robot
conda run -n home-robot python projects/habitat_ovmm/eval_baselines_agent.py \
  --env_config_path projects/habitat_ovmm/configs/env/hssd_demo.yaml \
  --agent_type random \
  --num_episodes 1 \
  --force_step 1 \
  habitat.simulator.habitat_sim_v0.gpu_device_id=0
```

Current result:

- Before downloading episodes, the smoke test failed at missing `data/datasets/ovmm/val/viewpoints.npy`.
- After downloading episodes, dataset initialization passed.
- Simulator initialization currently fails at OpenGL/EGL context creation:

```text
GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue
```

Also observed in sandboxed execution:

```text
Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device 0 among 1 EGL devices in total
WindowlessContext: Unable to create windowless context
```

Tried:

- Host execution outside sandbox.
- `env -u DISPLAY`.
- `EGL_PLATFORM=surfaceless`.

These did not fix local Habitat-Sim rendering.

Interpretation:

The local OVMM Python stack and episodes are mostly ready, but the local workstation is blocked at Habitat-Sim EGL/OpenGL context creation. This is a rendering backend issue, not an observed Python package import issue. Further OVMM rendering checks should run on a NVIDIA RTX render server.

## Sol-RL / Sana

Repository:

```text
/home/plote/longhorizon/rltask/Sana
```

Status:

- Local `sana` conda environment exists and partially passes preflight.
- `torch 2.9.1+cu128`, `sana`, `diffusers`, `transformers`, and `xformers` import.
- `mmcv 1.7.2` and `bitsandbytes 0.49.2` are installed.
- `flash_attn` is missing.
- `transformer-engine[pytorch]` is not installed; it is only required for NVFP4 / FP4 Sol-RL paths.
- Sana/Sol-RL install is CUDA-training heavy.
- `environment_setup.sh` creates or updates a Python 3.11 env, installs CUDA toolkit 12.8, torch 2.9.1 cu128, xformers, mmcv, editable Sana, Pi3, and flash-attn.
- Sol-RL NVFP4 paths may also need `transformer-engine[pytorch]`.

Local recommendation:

- Do not run full Sana install or `flash-attn` compilation locally. A low-priority single-thread install attempt still filled swap and was interrupted to avoid another freeze/reboot.
- Use local Sana source for code reading, config parsing, and algorithm extraction.
- Run Sol-RL training or heavy reward/model checks on the training machine after deciding CUDA vs ROCm strategy.

Preflight command:

```bash
bash lmm_rollout_project/scripts/env_check/solrl_local_preflight.sh
```

Expected current result:

```text
python_env_ok=True
torch 2.9.1+cu128 cuda 12.8 available <depends on execution context>
sana found
diffusers found
transformers found
xformers found
flash_attn missing
```

Server finish command for a NVIDIA CUDA machine with enough RAM:

```bash
cd /path/to/longhorizon/rltask/Sana
MAX_JOBS=2 NVCC_THREADS=1 CMAKE_BUILD_PARALLEL_LEVEL=2 \
  conda run -n sana pip install --no-build-isolation "flash-attn>=2.7.0"
```

Optional only for NVFP4 / FP4 Sol-RL configs:

```bash
conda run -n sana pip install --no-build-isolation "transformer-engine[pytorch]"
```

## Next Setup Tasks

1. Run OVMM rendering smoke on a NVIDIA RTX server instead of the local laptop.
2. Finish/verify HSSD scenes on a remote machine or only with explicit local approval.
3. Build `flash-attn` on a server if Sana needs it; do not compile it locally.
4. Keep local work limited to code edits, docs, imports, and short checks.
