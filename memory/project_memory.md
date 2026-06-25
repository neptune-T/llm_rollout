# Project Memory: LMM Rollout Scaling

Last updated: 2026-06-26

## Core Research Question

Can low-cost rollout scaling improve online RL for long-horizon mobile manipulation, and when can cheap rollout be trusted as a proxy for high-precision rollout?

The project is about long-horizon mobile manipulation (LMM), not generic language models. The central scientific question is ranking reliability: whether cheap rollout preserves the ordering of candidate behaviors that high-precision rollout would produce, especially across long horizons and contact-rich stages.

## Current Hypotheses

- H1: Cheap rollout is not uniformly reliable. It may work for short horizon or navigation-like stages and collapse in contact-rich manipulation.
- H2: Contact-rich stages are the main failure bottleneck. Grasp, pull, open, place, and release are likely more sensitive to precision and timing errors than navigation.
- H3: Naive rollout scaling can mislead online RL if cheap rollout ranking disagrees with high-precision rollout ranking.
- H4: Stage-aware verification or calibration can recover utility by selectively rerunning risky or promising cheap candidates in high precision.

## Confirmed Facts

- BEHAVIOR champion solution path: `/home/plote/longhorizon/basecode/behavior-1k-solution`.
- Sol-RL reference code path: `/home/plote/longhorizon/rltask/Sana`.
- OVMM/HomeRobot code path: `/home/plote/longhorizon/benchmark/home-robot`.
- Prior exploration notes path: `/home/plote/longhorizon/Planning/Doc/Todo.md`. This is historical context from another exploration, not the active project plan.
- Active project control folder: `/home/plote/longhorizon/lmm_rollout_project`.
- Active project plan path: `/home/plote/longhorizon/lmm_rollout_project/docs/current_plan.md`.
- Current immediate Stage 1 route is OVMM-first for the simplest mobile-manipulation environment loop.
- BEHAVIOR champion solution remains the later/main long-horizon benchmark and policy baseline.
- The active project does not inherit the prior OpenVLA baseline route.
- Current mainline does not include J-EPA/V-JEPA/world-model reward as a planned component. J-EPA references in prior notes are treated as historical alternatives only.
- AMD training server access is through VS Code Tunnel / VS Code for the Web, not standard SSH from the local workspace.
- AMD training server hardware is expected to be 8x AMD Instinct MI325X per full node under Slurm/ROCm.
- AMD reserved full-node command: `salloc --reservation=gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation --exclusive --mem=0`.
- The BEHAVIOR champion solution uses Pi0.5/OpenPI-style policy code with task embeddings, stage tracking, rolling inpainting, action compression, correction rules, and 4 task-specialized checkpoints.
- Local BEHAVIOR checkpoints are present under `/home/plote/longhorizon/basecode/behavior-1k-solution/checkpoints`; each checkpoint is about 12GB, total about 48GB.
- Local workstation GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8GB VRAM, driver 580.126.09, CUDA 13.0 reported by `nvidia-smi`.
- Local checkpoint loading failed before websocket serving due to system OOM: minimal `checkpoint_1` load was killed with signal 9 / exit code 137 at about 9.86GB RSS. This is a resource limit, not an observed Python logic traceback.
- In sandboxed commands JAX saw only CPU; host execution was needed for JAX to see `CudaDevice(id=0)`.
- Top-level `/home/plote/longhorizon` is an umbrella workspace, not currently a usable git repo from the shell. Relevant subrepos have their own git histories.
- `Planning` repo current head when checked: `ae1c9c1 up1`, clean.
- `behavior-1k-solution` current head when checked: `ca556f7 Added video link`, dirty with previous local adaptations and downloaded/generated files.
- `rltask/Sana` current head when checked: `51baa3c Merge pull request #406 from NVlabs/release/sana-v2v-inference-local`, clean.
- `benchmark/home-robot` current head when checked: `ede6a67a Patch for issues (#491)`, dirty with an existing local edit to `src/home_robot/environment.yml`.
- Local `home-robot` conda env exists and imports `torch`, `habitat_sim`, `habitat`, `home_robot`, and `home_robot_sim`.
- Local host execution sees CUDA from the `home-robot` env; sandboxed execution may hide GPU or EGL devices.
- OVMM episodes were downloaded to `/home/plote/longhorizon/benchmark/home-robot/data/datasets/ovmm` and pinned to `9ad25fbd86a3fd352c7a0fc1f99132fbb5802378`.
- Local OVMM smoke currently reaches dataset and simulator initialization but fails at Habitat-Sim OpenGL/EGL context creation.
- Local `sana` conda env exists. It has `torch 2.9.1+cu128`, `torchvision 0.24.1+cu128`, `torchaudio 2.9.1+cu128`, `xformers 0.0.33.post2`, `mmcv 1.7.2`, `bitsandbytes 0.49.2`, `sana`, `diffusers`, and `transformers`.
- Local `sana` env is still missing `flash-attn`; `transformer-engine[pytorch]` is also not installed and is only required for NVFP4 paths.
- NVIDIA A6000 render server was provisioned at SSH target `root@219.223.207.18 -p 20400`.
- A6000 remote workspace path: `/root/workspace/tianshanzhang`.
- A6000 server GPU check on 2026-06-25: 4x NVIDIA RTX A6000, 49140 MiB each, driver 570.86.10.
- First A6000 code/resource sync completed on 2026-06-25 with rsync exit code 0. Remote tree size after sync is about 14GB.
- Synced A6000 key directories: `lmm_rollout_project` 1.7MB, `rltask/Sana` 121MB, `benchmark/home-robot` 994MB, `benchmark/BEHAVIOR-1K` 1.6GB, `basecode/behavior-1k-solution` 11GB.
- A6000 remote ownership was changed to `root:root` after rsync to avoid Git dubious ownership warnings in VS Code Remote SSH.
- A6000 remote `git-lfs` was installed with apt and initialized system-wide. This fixed the false modified status for HomeRobot LFS assets.
- A6000 remote conda is installed at `/root/anaconda3`; `conda init bash` was run for root. Project-specific `sana` and `home-robot` conda environments have not yet been created on A6000.
- A6000 sync intentionally excluded `benchmark/home-robot/data`, `basecode/behavior-1k-solution/checkpoints`, `basecode/behavior-1k-solution/BEHAVIOR-1K-684a`, caches, virtualenvs, outputs, and wandb.
- A6000 NVIDIA preflight script passed on 2026-06-26 local server time / 2026-06-25 project session: GPUs, git, git-lfs, python3, bash, and champion repo were found. Docker was not found.
- A6000 has usable micromamba at `/root/.local/bin/micromamba`, version 2.5.0, with root prefix `/root/micromamba`.
- A6000 `home-robot` environment was created on 2026-06-26 at `/root/micromamba/envs/home-robot` from `benchmark/home-robot/src/home_robot/environment.yml`.

## Open Questions

- Can the A6000 server run OVMM/Habitat-Sim rendering headlessly without EGL/OpenGL failure?
- Can the A6000 server run BEHAVIOR / Isaac Sim simulation headlessly, and which Isaac/OmniGibson version is already installed or should be installed?
- Can BEHAVIOR-1K/OmniGibson run headless/offscreen on the intended simulation GPU without snow screen, black frames, or segmentation faults?
- Can local Habitat-Sim EGL/OpenGL be fixed for OVMM, or should OVMM rollout move directly to a NVIDIA render server?
- Has the HSSD HuggingFace license been accepted, and which HF account/token should be used for full OVMM scenes/objects download?
- What is the exact Isaac Sim / OmniGibson version required by the local champion solution copy?
- Can the 4 champion checkpoints load on the intended large server with enough RAM and VRAM?
- Does Sol-RL code in `rltask/Sana` map cleanly to robot policy rollouts, or is it only an algorithmic reference from diffusion post-training?
- What cheap rollout modes are feasible for Pi0.5/OpenPI/B1K policy: low precision JAX, quantized PyTorch conversion, smaller model, lower resolution, fewer denoising steps, or surrogate policy?
- What stage metrics can be extracted reliably from BEHAVIOR logs: task stage, contacts, object joint progress, action saturation, resets, and timeout?

## Current Pipeline Status

- Stage: Stage 1.1 preparation, now OVMM-first.
- Documentation and project memory are now being initialized.
- The active plan has been reset around the champion solution, not the prior OpenVLA/J-EPA exploration.
- Deployment scaffolding now targets two roles: NVIDIA render server for BEHAVIOR/Isaac rollout and AMD training server for ROCm-compatible training/calibration work. AMD is tunnel-only, so deployment uses a local tarball package plus commands run inside the AMD VS Code Tunnel terminal.
- OVMM/HomeRobot local Python stack is usable, but rendering is blocked by Habitat-Sim OpenGL/EGL context creation.
- BEHAVIOR champion policy server command is identified, but not runnable on local workstation due to memory.
- BEHAVIOR simulation/eval has not yet been validated in this workspace.
- Sol-RL code has been located and partially read; it is a reference implementation for FP4/NVFP4 rollout + BF16 training in diffusion models, not yet integrated with BEHAVIOR.
- A6000 NVIDIA render server code migration is complete for the current non-checkpoint workspace snapshot.
- A6000 `home-robot` env creation is complete, but import/runtime validation is still pending. No OVMM data under `benchmark/home-robot/data` and no BEHAVIOR champion checkpoints are present on A6000.

## Environment Status

- Local workstation is suitable for code editing, documentation, small static checks, and light Python imports.
- Local workstation must be treated as resource-constrained. Avoid long `git lfs pull`, package compilation such as `flash-attn`, Habitat/OVMM rendering smoke tests, BEHAVIOR checkpoint loading, or any task likely to saturate CPU/RAM/disk IO unless the user explicitly approves.
- Local workstation is partially suitable for OVMM setup: Python package imports and dataset initialization work, but Habitat-Sim rendering currently fails.
- Local workstation is not suitable for loading BEHAVIOR champion checkpoints.
- BEHAVIOR / Isaac Sim rollout should be tested on a machine with a supported NVIDIA RTX GPU and stable headless/offscreen rendering.
- A6000 render server is suitable for the next remote rendering setup attempt: 4x RTX A6000 with driver 570.86.10, reachable by SSH, and code is located at `/root/workspace/tianshanzhang`.
- A6000 currently has system Python 3.8.2, `/root/anaconda3` conda 4.8.3, and micromamba 2.5.0 at `/root/.local/bin/micromamba`.
- A6000 `home-robot` env exists at `/root/micromamba/envs/home-robot`. Import-only preflight and rendering smoke are still pending.
- A6000 `sana` env has not yet been created.
- A6000 currently does not have docker, so Isaac Sim container workflows need docker installation or a non-container local install path.
- Large policy inference/training should run on machines with substantially more RAM and GPU memory than the local RTX 3070 Ti laptop.
- AMD server compatibility is not assumed for the champion policy because the current champion dependency stack is CUDA/JAX-oriented. Use ROCm preflight and separate compatibility validation before training.

## Important Commands

Do not run large experiments locally without explicit approval.

Deployment scaffold:

```bash
cp lmm_rollout_project/configs/deploy/lmm_deploy.example.env lmm_rollout_project/configs/deploy/lmm_deploy.env
bash lmm_rollout_project/scripts/deploy/deploy_all.sh lmm_rollout_project/configs/deploy/lmm_deploy.env
bash lmm_rollout_project/scripts/deploy/package_for_tunnel.sh lmm_rollout_project/configs/deploy/lmm_deploy.env
```

BEHAVIOR champion policy server, to run on a sufficiently large host:

```bash
cd /path/to/behavior-1k-solution
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ./.venv/bin/python scripts/serve_b1k.py \
  --task-checkpoint-mapping task_checkpoint_mapping.local.json \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_fast \
  --policy.dir checkpoints/checkpoint_1
```

BEHAVIOR eval client pattern:

```bash
python BEHAVIOR-1K/omnigibson/learning/eval.py \
  log_path=./eval_logs \
  policy=websocket \
  task.name=make_microwave_popcorn \
  model.host=<policy_server_host> \
  eval_instance_ids="[0,1,2,3]"
```

Sol-RL reference training entry points:

```bash
cd /home/plote/longhorizon/rltask/Sana
bash train_scripts/sol_rl/run_sana_single_node_8gpu.sh
bash train_scripts/sol_rl/run_sd3_single_node_8gpu.sh
bash train_scripts/sol_rl/run_flux1_single_node_8gpu.sh
```

OVMM local smoke:

```bash
bash lmm_rollout_project/scripts/env_check/ovmm_local_smoke.sh
```

Sol-RL local preflight:

```bash
bash lmm_rollout_project/scripts/env_check/solrl_local_preflight.sh
```

A6000 remote access:

```bash
ssh -p 20400 root@219.223.207.18
cd /root/workspace/tianshanzhang
```

A6000 remote checks:

```bash
nvidia-smi
/root/anaconda3/bin/conda info --envs
/root/.local/bin/micromamba env list
git -C rltask/Sana status --short
git -C benchmark/home-robot status --short
git -C basecode/behavior-1k-solution status --short
```

A6000 HomeRobot env:

```bash
cd /root/workspace/tianshanzhang/benchmark/home-robot
/root/.local/bin/micromamba run -n home-robot python --version
/root/.local/bin/micromamba run -n home-robot python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## Dataset / Benchmark Notes

- OVMM is the immediate first benchmark for environment bring-up because it should be simpler than BEHAVIOR/Isaac.
- BEHAVIOR-1K is still the preferred later long-horizon mobile manipulation benchmark, but Isaac Sim rendering stability is a major risk.
- LIBERO/CALVIN/RoboTwin can be auxiliary checks but should not pull the project away from mobile manipulation.

## Model Notes

- BEHAVIOR champion policy is Pi0.5/OpenPI-derived with task embeddings, no text prompt processing at inference.
- Policy inputs include three RGB cameras resized to 224x224 and R1Pro proprioception.
- Policy outputs 23-dimensional actions after truncating from model action dimension 32.
- Existing inference wrapper includes stage voting, rolling inpainting, cubic action interpolation, and correction rules.
- Sol-RL reference uses preview rollout with NVFP4 compiled model and full rollout with compiled BF16 model for diffusion post-training.
- Local Sana/Sol-RL env is partially usable for imports and code reading. It is not complete for full Sol-RL training because `flash-attn` is missing.
- `transformer-engine[pytorch]` is optional for the NVFP4 / FP4 rollout paths and is not installed locally.
- OpenVLA is not the active baseline unless the champion route becomes blocked and the project explicitly pivots.
- J-EPA/V-JEPA is not in the active method stack; do not add it without an explicit project decision.

## Experiment Index

- `exp_20260619_001`: Local BEHAVIOR checkpoint load smoke test. Result: failed due to OOM / SIGKILL before server start.
- `exp_20260620_001`: Local OVMM/HomeRobot import and CUDA visibility check. Result: imports pass; host sees CUDA.
- `exp_20260620_002`: Local OVMM one-episode random-agent smoke. Result: episodes added and dataset init passes; simulator fails at OpenGL/EGL context creation.
- `exp_20260620_003`: Local Sana/Sol-RL preflight. Historical result: no `sana` env at that time.
- `exp_20260623_001`: Local Sana/Sol-RL preflight. Result: main CUDA PyTorch/Sana packages import; `flash_attn` missing.
- `exp_20260623_002`: Local `flash-attn` low-priority single-thread install attempt. Result: interrupted because system swap became full and local crash risk was high.
- `exp_20260626_001`: A6000 OVMM/HomeRobot environment setup requirements preflight. Result: env file and install scripts identified; avoid running full `install_deps.sh` blindly.
- `exp_20260626_002`: A6000 mamba/micromamba search. Result: `mamba` not on PATH; `/root/.local/bin/micromamba` exists.
- `exp_20260626_003`: A6000 micromamba validation. Result: micromamba 2.5.0 usable with root prefix `/root/micromamba`.
- `exp_20260626_004`: A6000 `home-robot` env creation. Result: completed successfully at `/root/micromamba/envs/home-robot`.

## Key Results

- No scientific rollout results yet.
- Engineering result: local workstation cannot load BEHAVIOR champion checkpoint. Use remote large server for policy loading.
- Engineering result: local workstation has a mostly installed OVMM/HomeRobot stack and OVMM episodes, but cannot currently create the Habitat-Sim rendering context.
- Engineering result: A6000 now has a dedicated micromamba `home-robot` environment. Imports and rendering are not validated yet.

## Failure Modes

- Local OOM during checkpoint restore: `checkpoint_1` load killed by signal 9 at about 9.86GB RSS.
- Original BEHAVIOR mapping file pointed to `~/models/checkpoint_*`, which did not exist locally. A local mapping file was created in the champion repo during prior debugging.
- JAX device visibility differs between sandbox and host execution.
- Potential Isaac Sim snow screen / black render risk on unsupported or non-RTX server GPUs is documented in prior notes.
- Local OVMM Habitat-Sim fails with `GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue`.
- Sandboxed OVMM Habitat-Sim can also fail with `unable to find CUDA device 0 among 1 EGL devices`.
- Missing OVMM episodes caused `FileNotFoundError` for `data/datasets/ovmm/val/viewpoints.npy`; fixed by downloading `ai-habitat/OVMM_episodes`.
- Local heavy setup caused workstation instability/reboots. `flash-attn` build and HSSD `git lfs pull` are now considered remote-only or explicit-approval tasks.
- Local `flash-attn` install attempt on 2026-06-23 used `nice`, `ionice`, `MAX_JOBS=1`, and `NVCC_THREADS=1`, but still pushed swap to 100%; do not retry locally.
- A6000 initial rsync preserved local UID 1000, causing Git `dubious ownership` when accessed as root. Fixed by `chown -R root:root /root/workspace/tianshanzhang`.
- A6000 HomeRobot initially showed many modified LFS assets because `git-lfs` was missing. Fixed by installing `git-lfs`; remaining HomeRobot dirty status is the known `src/home_robot/environment.yml` edit.

## Paper Story

Working claim:

Low-cost rollout scaling can improve online RL for long-horizon mobile manipulation only when its ranking reliability is explicitly measured and corrected, especially under long horizons and contact-rich stages.

Required evidence:

- Cheap-vs-high-precision ranking correlation by stage and horizon.
- Failure cases showing contact-rich ranking collapse.
- Comparison of no RL, high-precision-only rollout, naive cheap rollout, and cheap rollout plus verification/calibration.
- Wall-clock and GPU-hour efficiency, not only success rate.

## Next Milestones

1. Run A6000 `home-robot` import-only preflight.
2. Download or sync OVMM data on A6000, preferably with remote download/HF mirror rather than local heavy `git lfs pull`.
3. Run OVMM headless rendering smoke on A6000.
4. Create A6000 `sana` environment and install `flash-attn` remotely, not locally.
5. Decide whether to transfer 48GB BEHAVIOR champion checkpoints to A6000 or download/copy them from a faster shared source.
6. Build the standard rollout record schema on OVMM first.
7. Run BEHAVIOR environment check later on the NVIDIA render machine: headless render, reset, step, observation extraction, and video/keyframe save.
