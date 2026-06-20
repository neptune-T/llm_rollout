# Project Memory: LMM Rollout Scaling

Last updated: 2026-06-20

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
- No local `sana` conda env exists. Full Sana/Sol-RL install was not run locally because it is a heavy CUDA 12.8 training stack.

## Open Questions

- Which remote machine will host BEHAVIOR / Isaac Sim simulation, and which machine will host large policy inference/training?
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

## Environment Status

- Local workstation is suitable for code editing, documentation, small static checks, and light Python imports.
- Local workstation is partially suitable for OVMM setup: Python package imports and dataset initialization work, but Habitat-Sim rendering currently fails.
- Local workstation is not suitable for loading BEHAVIOR champion checkpoints.
- BEHAVIOR / Isaac Sim rollout should be tested on a machine with a supported NVIDIA RTX GPU and stable headless/offscreen rendering.
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
- Local Sana/Sol-RL has no installed `sana` env. Treat the repo as algorithm reference locally unless explicitly installing a heavy CUDA training stack.
- OpenVLA is not the active baseline unless the champion route becomes blocked and the project explicitly pivots.
- J-EPA/V-JEPA is not in the active method stack; do not add it without an explicit project decision.

## Experiment Index

- `exp_20260619_001`: Local BEHAVIOR checkpoint load smoke test. Result: failed due to OOM / SIGKILL before server start.
- `exp_20260620_001`: Local OVMM/HomeRobot import and CUDA visibility check. Result: imports pass; host sees CUDA.
- `exp_20260620_002`: Local OVMM one-episode random-agent smoke. Result: episodes added and dataset init passes; simulator fails at OpenGL/EGL context creation.
- `exp_20260620_003`: Local Sana/Sol-RL preflight. Result: no `sana` env; full local install not run.

## Key Results

- No scientific rollout results yet.
- Engineering result: local workstation cannot load BEHAVIOR champion checkpoint. Use remote large server for policy loading.
- Engineering result: local workstation has a mostly installed OVMM/HomeRobot stack and OVMM episodes, but cannot currently create the Habitat-Sim rendering context.

## Failure Modes

- Local OOM during checkpoint restore: `checkpoint_1` load killed by signal 9 at about 9.86GB RSS.
- Original BEHAVIOR mapping file pointed to `~/models/checkpoint_*`, which did not exist locally. A local mapping file was created in the champion repo during prior debugging.
- JAX device visibility differs between sandbox and host execution.
- Potential Isaac Sim snow screen / black render risk on unsupported or non-RTX server GPUs is documented in prior notes.
- Local OVMM Habitat-Sim fails with `GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue`.
- Sandboxed OVMM Habitat-Sim can also fail with `unable to find CUDA device 0 among 1 EGL devices`.
- Missing OVMM episodes caused `FileNotFoundError` for `data/datasets/ovmm/val/viewpoints.npy`; fixed by downloading `ai-habitat/OVMM_episodes`.

## Paper Story

Working claim:

Low-cost rollout scaling can improve online RL for long-horizon mobile manipulation only when its ranking reliability is explicitly measured and corrected, especially under long horizons and contact-rich stages.

Required evidence:

- Cheap-vs-high-precision ranking correlation by stage and horizon.
- Failure cases showing contact-rich ranking collapse.
- Comparison of no RL, high-precision-only rollout, naive cheap rollout, and cheap rollout plus verification/calibration.
- Wall-clock and GPU-hour efficiency, not only success rate.

## Next Milestones

1. Fix local OVMM Habitat-Sim EGL/OpenGL or run OVMM smoke on a NVIDIA RTX server.
2. Confirm HuggingFace license/login and download HSSD scenes plus OVMM object assets if local or remote storage is sufficient.
3. Build the standard rollout record schema on OVMM first.
4. Fill `lmm_rollout_project/configs/deploy/lmm_deploy.env` with the NVIDIA SSH host, AMD tunnel mode, and both remote root paths.
5. Run BEHAVIOR environment check later on the NVIDIA render machine: headless render, reset, step, observation extraction, and video/keyframe save.
