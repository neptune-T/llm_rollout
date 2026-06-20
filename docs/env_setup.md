# Environment Setup Notes

Last updated: 2026-06-20

## Current Direction

The immediate Stage 1 target is OVMM first, because it is simpler to bring up than BEHAVIOR/Isaac and can provide an earlier mobile-manipulation rollout loop.

BEHAVIOR champion solution remains the stronger long-horizon target for the paper, but it is not the first local smoke target.

## Local Machine

- Machine role: local code editing and light smoke tests.
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8GB VRAM.
- Driver: 580.126.09.
- CUDA reported by `nvidia-smi`: 13.0.
- Disk under workspace: about 93GB free when checked.
- RAM: 15GB total.

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
- HSSD scenes under `data/hssd-hab` are not present.
- OVMM object assets under `data/objects` are not present.
- HuggingFace CLI reported not logged in; HSSD scenes may require accepting the dataset license and logging in.

Smoke command:

```bash
bash scripts/env_check/ovmm_local_smoke.sh
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

The local OVMM Python stack and episodes are mostly ready, but the local workstation is blocked at Habitat-Sim EGL/OpenGL context creation. This is a rendering backend issue, not an observed Python package import issue.

## Sol-RL / Sana

Repository:

```text
/home/plote/longhorizon/rltask/Sana
```

Status:

- No `sana` conda environment exists locally.
- Sana/Sol-RL install is CUDA-training heavy.
- `environment_setup.sh` creates or updates a Python 3.11 env, installs CUDA toolkit 12.8, torch 2.9.1 cu128, xformers, mmcv, editable Sana, Pi3, and flash-attn.
- Sol-RL NVFP4 paths may also need `transformer-engine[pytorch]`.

Local recommendation:

- Do not run the full Sana install locally unless we explicitly want a CUDA training stack on the laptop.
- Use local Sana source for code reading, config parsing, and algorithm extraction.
- Run Sol-RL training or heavy reward/model checks on the training machine after deciding CUDA vs ROCm strategy.

Preflight command:

```bash
bash scripts/env_check/solrl_local_preflight.sh
```

Expected current result:

```text
missing_conda_env=sana
```

## Next Setup Tasks

1. Fix or bypass local Habitat-Sim EGL/OpenGL rendering for OVMM.
2. If local rendering remains blocked, run OVMM smoke on a NVIDIA RTX server.
3. Download HSSD scenes and OVMM object assets only after HuggingFace license/login and storage location are confirmed.
4. Keep Sana/Sol-RL as algorithm reference locally; install full training environment on the actual training machine, not the laptop, unless explicitly needed.
