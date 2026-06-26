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
- A6000 remote conda is installed at `/root/anaconda3`; `conda init bash` was run for root. Project-specific `home-robot` micromamba env has been created under micromamba, and A6000 `sana` env has been created under `/root/anaconda3/envs/sana`.
- A6000 sync intentionally excluded `benchmark/home-robot/data`, `basecode/behavior-1k-solution/checkpoints`, `basecode/behavior-1k-solution/BEHAVIOR-1K-684a`, caches, virtualenvs, outputs, and wandb.
- A6000 NVIDIA preflight script passed on 2026-06-26 local server time / 2026-06-25 project session: GPUs, git, git-lfs, python3, bash, and champion repo were found. Docker was not found.
- A6000 has usable micromamba at `/root/.local/bin/micromamba`, version 2.5.0, with root prefix `/root/micromamba`.
- A6000 `home-robot` environment was created on 2026-06-26 at `/root/micromamba/envs/home-robot` from `benchmark/home-robot/src/home_robot/environment.yml`.
- A6000 HomeRobot/OVMM import-only preflight passed on 2026-06-26 after installing Habitat-Sim, Habitat-Lab, and Habitat-Baselines.
- A6000 `home-robot` confirmed versions: Python 3.9.23, PyTorch 1.13.1 CUDA 11.7 build, torchvision 0.14.1, PyTorch3D 0.7.5, PyG 2.5.2, numpy 1.23.5, habitat-sim 0.2.5, habitat-lab 0.2.5, habitat-baselines 0.2.5.
- A6000 `home-robot` PyTorch CUDA sees 4x NVIDIA RTX A6000.
- A6000 OVMM data roots are now present under `/root/workspace/tianshanzhang/benchmark/home-robot/data`.
- A6000 `OVMM_episodes` was downloaded through `HF_ENDPOINT=https://hf-mirror.com` and pinned to `9ad25fbd86a3fd352c7a0fc1f99132fbb5802378`.
- A6000 `OVMM_objects` mirror snapshot failed because mirror pagination redirected to `huggingface.co`; mirror Git metadata worked, but Git LFS payload download failed because `us.aws.cdn.hf-mirror.org` did not resolve. The runtime working tree was filled by rsyncing verified local payload files while excluding `.git`.
- A6000 `hssd-hab` scene data was synced by clean local-to-remote rsync while excluding `.git`; final runtime size is 3.9G and sample hashes match local files.
- A6000 `hab_stretch` robot assets were synced by rsync; final runtime size is 47M and sample hashes match local files.
- A6000 OVMM data-root sanity check passed for `data/hssd-hab`, `data/objects`, `data/datasets/ovmm`, and `data/robots/hab_stretch`.
- First bounded A6000 OVMM random-agent smoke reached the eval entrypoint but failed before simulator construction because `home_robot_sim` was not installed into the `home-robot` env.
- A6000 `home_robot` and `home_robot_sim` are now installed editable with `pip install --no-deps -e ...`; direct import and expanded import preflight pass after this repair.
- A6000 uses `lmm_rollout_project/shims/sophus.py` to expose installed `sophuspy` as the legacy module name `sophus` required by HomeRobot geometry utilities.
- A6000 `libegl1` is installed; this fixed the initial `no EGL devices found` state and made `libEGL.so.1` discoverable.
- A6000 has an NVIDIA driver/user-space mismatch: loaded kernel module is 570.86.10, while system `libnvidia-gl-570` and `libnvidia-compute-570` user-space libraries are 570.133.07.
- A non-invasive NVIDIA 570.86.10 runtime overlay exists under `/root/workspace/tianshanzhang/runtime_libs/`; with Magnum validation it exposes 4 EGL devices and maps CUDA device 0 to EGL device 0, but Habitat-Sim still fails at OpenGL version retrieval.
- A6000 OVMM Python environment and data roots are configured, but OVMM headless rendering is currently blocked by host/container NVIDIA GL/EGL stack consistency.
- A6000 `sana` env exists at `/root/anaconda3/envs/sana` with Python 3.11.6, CUDA toolkit 12.8, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, torchaudio 2.9.1+cu128, xformers 0.0.33.post2, mmcv 1.7.2, editable Sana 0.2.0, and Pi3 0.1.
- A6000 `sana` core import preflight passes for `torch`, `torchvision`, `torchaudio`, `xformers`, `mmcv`, `sana`, `diffusers`, `transformers`, `accelerate`, `bitsandbytes`, `clip`, `peft`, `timm`, `hpsv2`, `open_clip`, `wandb`, `gradio`, and `pi3`.
- A6000 `sana` still lacks `flash-attn`; `transformer-engine[pytorch]` is not installed and is only required for NVFP4/FP4 Sol-RL paths.

## Open Questions

- Can the A6000 server run OVMM/Habitat-Sim rendering headlessly after host-level NVIDIA kernel/user-space library alignment or reboot?
- Can the A6000 server run BEHAVIOR / Isaac Sim simulation headlessly, and which Isaac/OmniGibson version is already installed or should be installed?
- Can BEHAVIOR-1K/OmniGibson run headless/offscreen on the intended simulation GPU without snow screen, black frames, or segmentation faults?
- Can local Habitat-Sim EGL/OpenGL be fixed for OVMM, or should OVMM rollout move directly to a NVIDIA render server?
- Has the HSSD HuggingFace license been accepted for future direct downloads, and which HF account/token should be used if we need to reconstruct OVMM data without the local working tree?
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
- A6000 `home-robot` env creation, import-only validation, OVMM data-root validation, editable local package registration, and `sophus` compatibility repair are complete.
- A6000 OVMM rendering smoke reaches dataset and simulator initialization but is blocked at Habitat-Sim OpenGL context creation due to the current NVIDIA GL/EGL stack.
- A6000 `sana` non-render setup is mostly complete: CUDA toolkit, PyTorch cu128, xformers, mmcv, editable Sana, and Pi3 import validation pass. Remaining Sol-RL dependency work is `flash-attn`, plus optional `transformer-engine[pytorch]` for NVFP4/FP4.
- BEHAVIOR champion checkpoints are still not present on A6000.

## Environment Status

- Local workstation is suitable for code editing, documentation, small static checks, and light Python imports.
- Local workstation must be treated as resource-constrained. Avoid long `git lfs pull`, package compilation such as `flash-attn`, Habitat/OVMM rendering smoke tests, BEHAVIOR checkpoint loading, or any task likely to saturate CPU/RAM/disk IO unless the user explicitly approves.
- Local workstation is partially suitable for OVMM setup: Python package imports and dataset initialization work, but Habitat-Sim rendering currently fails.
- Local workstation is not suitable for loading BEHAVIOR champion checkpoints.
- BEHAVIOR / Isaac Sim rollout should be tested on a machine with a supported NVIDIA RTX GPU and stable headless/offscreen rendering.
- A6000 render server is suitable for the next remote rendering setup attempt: 4x RTX A6000 with driver 570.86.10, reachable by SSH, and code is located at `/root/workspace/tianshanzhang`.
- A6000 currently has system Python 3.8.2, `/root/anaconda3` conda 4.8.3, and micromamba 2.5.0 at `/root/.local/bin/micromamba`.
- A6000 `home-robot` env exists at `/root/micromamba/envs/home-robot`. Import-only preflight passes; rendering smoke is still pending.
- A6000 OVMM data roots present: `data/datasets/ovmm`, `data/objects`, `data/hssd-hab`, and `data/robots/hab_stretch`.
- A6000 `home_robot==0.1.0` and `home_robot_sim==0.1.0` are installed editable in the `home-robot` env with `--no-deps`.
- A6000 requires `PYTHONPATH=/root/workspace/tianshanzhang/lmm_rollout_project/shims:$PYTHONPATH` for OVMM eval commands unless the shim is packaged later.
- A6000 currently should not be treated as a working OVMM render host until the NVIDIA kernel/user-space library mismatch is fixed.
- A6000 `sana` env exists at `/root/anaconda3/envs/sana`. Core Sana/Sol-RL imports and Pi3 import pass; `flash-attn` remains missing.
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

A6000 Sana/Sol-RL env:

```bash
/root/anaconda3/envs/sana/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
/root/anaconda3/envs/sana/bin/python -c "import sana, diffusers, transformers, xformers, bitsandbytes, pi3; print('SANA_CORE_IMPORT_OK')"
```

A6000 Pi3 source-only install workflow, used because full Git checkout/archive downloads were unreliable:

```bash
git --git-dir=/root/workspace/tianshanzhang/externals/Pi3_shallow/.git archive \
  --format=tar \
  --prefix=Pi3_src/ \
  -o /root/workspace/tianshanzhang/externals/Pi3_src_sparse.tar \
  HEAD pi3 pyproject.toml requirements.txt LICENSE README.md
tar -xf /root/workspace/tianshanzhang/externals/Pi3_src_sparse.tar -C /root/workspace/tianshanzhang/externals
/root/anaconda3/envs/sana/bin/python -m pip install --no-deps /root/workspace/tianshanzhang/externals/Pi3_src
/root/anaconda3/envs/sana/bin/python -c "import pi3; import pi3.models.pi3; import pi3.models.pi3x; import pi3.utils.geometry; print('PI3_IMPORT_OK')"
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
/root/.local/bin/micromamba run -n home-robot python /root/workspace/tianshanzhang/lmm_rollout_project/scripts/env_check/a6000_home_robot_import_preflight.py
```

A6000 HomeRobot/Habitat dependency repair recipe used on 2026-06-26:

```bash
cd /root/workspace/tianshanzhang/benchmark/home-robot
/root/.local/bin/micromamba install -y -n home-robot --override-channels -c aihabitat -c conda-forge habitat-sim=0.2.5 withbullet
apt-get install -y libopengl0
/root/.local/bin/micromamba install -y --no-deps -n home-robot --override-channels -c pytorch -c pytorch3d -c nvidia -c conda-forge torchvision=0.14.1=py39_cu117 pytorch3d=0.7.5=py39_cu117_pyt1131
/root/.local/bin/micromamba run -n home-robot python -m pip install -e src/third_party/habitat-lab/habitat-lab
/root/.local/bin/micromamba run -n home-robot python -m pip install -e src/third_party/habitat-lab/habitat-baselines
/root/.local/bin/micromamba run -n home-robot python -m pip install numpy==1.23.5 moviepy==1.0.3
/root/.local/bin/micromamba run -n home-robot python -m pip check
```

A6000 HomeRobot local package registration:

```bash
cd /root/workspace/tianshanzhang/benchmark/home-robot
/root/.local/bin/micromamba run -n home-robot python -m pip install --no-deps -e src/home_robot
/root/.local/bin/micromamba run -n home-robot python -m pip install --no-deps -e src/home_robot_sim
/root/.local/bin/micromamba run -n home-robot python -c "import home_robot, home_robot_sim; print('HOME_ROBOT_EDITABLE_IMPORT_OK')"
/root/.local/bin/micromamba run -n home-robot python /root/workspace/tianshanzhang/lmm_rollout_project/scripts/env_check/a6000_home_robot_import_preflight.py
```

A6000 OVMM data validation:

```bash
ssh -p 20400 root@219.223.207.18 \
  du -sh /root/workspace/tianshanzhang/benchmark/home-robot/data/hssd-hab \
         /root/workspace/tianshanzhang/benchmark/home-robot/data/objects \
         /root/workspace/tianshanzhang/benchmark/home-robot/data/datasets/ovmm \
         /root/workspace/tianshanzhang/benchmark/home-robot/data/robots/hab_stretch
```

A6000 bounded OVMM random-agent smoke:

```bash
ssh -p 20400 root@219.223.207.18 \
  'cd /root/workspace/tianshanzhang/benchmark/home-robot && PYTHONPATH=/root/workspace/tianshanzhang/lmm_rollout_project/shims:$PYTHONPATH timeout 240s /root/.local/bin/micromamba run -n home-robot python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_demo.yaml --agent_type random --num_episodes 1 habitat.environment.max_episode_steps=1 habitat.simulator.habitat_sim_v0.gpu_device_id=0'
```

A6000 experimental NVIDIA 570.86.10 runtime overlay, for diagnostics only:

```bash
LD_LIBRARY_PATH=/root/workspace/tianshanzhang/runtime_libs/nvidia57086_cuda:/root/workspace/tianshanzhang/runtime_libs/nvidia57086_gl/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
__EGL_VENDOR_LIBRARY_FILENAMES=/root/workspace/tianshanzhang/runtime_libs/nvidia57086_gl/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

Recommended data transfer notes:

- `OVMM_episodes`: `huggingface_hub.snapshot_download` works through `HF_ENDPOINT=https://hf-mirror.com`.
- `OVMM_objects`: mirror snapshot and Git LFS payload download are unreliable in this environment; use mirror Git metadata plus local payload rsync, or rsync a verified working tree directly.
- `hssd-hab`: use clean working-tree rsync excluding `.git`; remote runtime target should be `/root/workspace/tianshanzhang/benchmark/home-robot/data/hssd-hab`.
- For LFS-heavy repos where `.git/lfs/objects` is not copied, validate by runtime size and sample file hashes, not by `git status`.

## Dataset / Benchmark Notes

- OVMM is the immediate first benchmark for environment bring-up because it should be simpler than BEHAVIOR/Isaac.
- A6000 OVMM data assembly is complete enough for first smoke: episodes via HF mirror, objects via mirror metadata plus rsynced payload, HSSD via clean working-tree rsync, and Stretch robot assets via rsync.
- BEHAVIOR-1K is still the preferred later long-horizon mobile manipulation benchmark, but Isaac Sim rendering stability is a major risk.
- LIBERO/CALVIN/RoboTwin can be auxiliary checks but should not pull the project away from mobile manipulation.

## Model Notes

- BEHAVIOR champion policy is Pi0.5/OpenPI-derived with task embeddings, no text prompt processing at inference.
- Policy inputs include three RGB cameras resized to 224x224 and R1Pro proprioception.
- Policy outputs 23-dimensional actions after truncating from model action dimension 32.
- Existing inference wrapper includes stage voting, rolling inpainting, cubic action interpolation, and correction rules.
- Sol-RL reference uses preview rollout with NVFP4 compiled model and full rollout with compiled BF16 model for diffusion post-training.
- Local Sana/Sol-RL env is partially usable for imports and code reading. It is not complete for full Sol-RL training because `flash-attn` is missing.
- A6000 Sana/Sol-RL env is now the preferred CUDA setup target. It has editable Sana and Pi3 import-validated; `flash-attn` remains the main missing package before full Sol-RL path validation.
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
- `exp_20260626_005`: Sync project memory/skill/log to A6000. Result: first SSH timeout, retry succeeded.
- `exp_20260626_006`: Added A6000 HomeRobot import preflight script. Result: script checks imports and CUDA only.
- `exp_20260626_007`: A6000 HomeRobot import preflight before Habitat install. Result: core PyTorch/CUDA imports pass; `habitat_sim` and `habitat` fail.
- `exp_20260626_008`: Broad HomeRobot simulation env update. Result: solver stayed active over 13 minutes and was terminated.
- `exp_20260626_009`: Narrow Habitat-Sim install. Result: `habitat-sim 0.2.5` installed, but `torchvision` and `pytorch3d` were removed and `libOpenGL.so.0` was missing.
- `exp_20260626_010`: A6000 HomeRobot repair and core import validation. Result: installed `libopengl0`, restored `torchvision`/`pytorch3d`, installed editable Habitat-Lab, preflight passed.
- `exp_20260626_011`: Editable Habitat-Baselines install. Result: installed, then repaired numpy/moviepy pins; `pip check` passed.
- `exp_20260626_012`: Expanded A6000 HomeRobot import preflight. Result: passed, including `numpy`, `quaternion`, and `habitat_baselines`.
- `exp_20260626_013`: OVMM data availability inventory. Result: local data exists; A6000 data roots were missing or incomplete.
- `exp_20260626_014`: Local-to-A6000 OVMM scene/object rsync trial. Result: too slow for full payload; interrupted after partial progress.
- `exp_20260626_015`: Preserve partial rsync directories. Result: partial directories renamed and kept out of clean target paths.
- `exp_20260626_016`: A6000 OVMM episode download through HF mirror. Result: completed and pinned to `9ad25fbd86a3fd352c7a0fc1f99132fbb5802378`.
- `exp_20260626_017`: First A6000 OVMM object snapshot download. Result: failed due mirror endpoint/path behavior.
- `exp_20260626_018`: Retried A6000 OVMM object snapshot with endpoint inside micromamba and lower concurrency. Result: failed; mirror pagination redirected to `huggingface.co`.
- `exp_20260626_019`: A6000 OVMM object metadata clone through HF mirror Git endpoint. Result: metadata clone and checkout succeeded with LFS smudge disabled.
- `exp_20260626_020`: A6000 OVMM object Git LFS pull from mirror. Result: failed because `us.aws.cdn.hf-mirror.org` did not resolve.
- `exp_20260626_021`: Fill A6000 OVMM object working tree by rsyncing local payload. Result: completed; runtime payload hashes match samples.
- `exp_20260626_022`: Sync A6000 OVMM Stretch robot assets. Result: completed; sample hashes match.
- `exp_20260626_023`: Sync A6000 HSSD scene data by clean rsync excluding `.git`. Result: completed; 3.9G runtime size and sample hashes match.
- `exp_20260626_024`: A6000 OVMM data-root sanity check. Result: passed for scenes, objects, episodes, and Stretch robot assets.
- `exp_20260626_025`: First bounded A6000 OVMM random-agent smoke. Result: failed before simulator construction because `home_robot_sim` was not importable by the eval entrypoint.
- `exp_20260626_026`: A6000 editable HomeRobot package registration. Result: `home_robot` and `home_robot_sim` installed with `--no-deps`; direct import and expanded preflight passed.
- `exp_20260626_027`: A6000 bounded smoke after editable package registration. Result: progressed past `home_robot_sim`, failed on missing legacy `sophus` module.
- `exp_20260626_028`: Added `sophus` compatibility shim. Result: expanded import preflight passed.
- `exp_20260626_029`: A6000 bounded smoke with shim. Result: reached dataset/simulator init, failed with `no EGL devices found`.
- `exp_20260626_030`: A6000 EGL diagnostics. Result: NVIDIA device nodes and vendor json exist; `/dev/dri` and system `libegl1` were missing; system NVIDIA user-space libs mismatch suspected.
- `exp_20260626_031`: Installed `libegl1`. Result: generic `libEGL.so.1` is discoverable and loadable.
- `exp_20260626_032`: A6000 bounded smoke after `libegl1`. Result: failure changed to CUDA/EGL device mapping.
- `exp_20260626_033`: GPU ID mapping probe. Result: device IDs 0-3 all failed mapping to the exposed EGL device.
- `exp_20260626_034`: NVIDIA version diagnostics. Result: kernel module 570.86.10, system user-space NVIDIA libraries 570.133.07.
- `exp_20260626_035`: Built non-invasive 570.86.10 NVIDIA runtime overlay. Result: overlay libraries are present; manual multi-library `ctypes` validation is not clean due double-free at exit.
- `exp_20260626_036`: A6000 bounded smoke with overlay. Result: CUDA/EGL mapping changed to OpenGL version retrieval failure.
- `exp_20260626_037`: Magnum validation for overlay. Result: 4 EGL devices found and CUDA device 0 maps to EGL device 0; OpenGL context/version remains invalid.
- `exp_20260626_038`: Session resume and control-file sync. Result: confirmed Stage 1.1 and synced docs/tools.
- `exp_20260626_039`: Overlay smoke with `CUDA_VISIBLE_DEVICES=0`. Result: same OpenGL version retrieval failure.
- `exp_20260626_040`: Overlay smoke with `EGL_PLATFORM=surfaceless` and `PYOPENGL_PLATFORM=egl`. Result: same OpenGL version retrieval failure; current A6000 render setup is blocked by host/container NVIDIA GL/EGL stack.
- `exp_20260626_041`: A6000 Sana/Sol-RL env preflight. Result: no `sana` env existed yet; setup entrypoints and required packages identified.
- `exp_20260626_042`: A6000 `sana` base env creation. Result: conda env created at `/root/anaconda3/envs/sana` with Python 3.11.15.
- `exp_20260626_043`: A6000 CUDA toolkit install through old conda. Result: solver hung and was terminated; base Python env remained usable.
- `exp_20260626_044`: A6000 CUDA toolkit install through micromamba into existing conda prefix. Result: CUDA toolkit 12.8 installed, Python became 3.11.6, `nvcc 12.8` validated.
- `exp_20260626_045`: A6000 `sana` pip/setuptools preparation. Result: `SETUPTOOLS_USE_DISTUTILS=stdlib`, pip 26.1.2, setuptools 69.5.1, wheel 0.47.0 validated.
- `exp_20260626_046`: A6000 first PyTorch cu128 install attempt. Result: direct setup-script style pip command hung with no output and was terminated.
- `exp_20260626_047`: A6000 PyTorch cu128 index connectivity probe. Result: PyTorch cu128 root and torch sub-index returned HTTP 200.
- `exp_20260626_048`: A6000 PyTorch cu128 stack install. Result: `torch 2.9.1+cu128`, `torchvision 0.24.1+cu128`, `torchaudio 2.9.1+cu128`, and 4 CUDA-visible A6000 GPUs validated.
- `exp_20260626_049`: A6000 xformers install. Result: `xformers 0.0.33.post2` installed and import-validated.
- `exp_20260626_050`: A6000 mmcv install. Result: `mmcv 1.7.2` pure-Python install passed config imports; compiled `mmcv._ext` absent.
- `exp_20260626_051`: A6000 editable Sana install. Result: Sana 0.2.0 and core imports passed.
- `exp_20260626_058`: A6000 Pi3 source-only install. Result: `pi3 0.1` installed from a source-only Git archive and import-validated; slow PyTorch RoPE2D fallback warning observed.

## Key Results

- No scientific rollout results yet.
- Engineering result: local workstation cannot load BEHAVIOR champion checkpoint. Use remote large server for policy loading.
- Engineering result: local workstation has a mostly installed OVMM/HomeRobot stack and OVMM episodes, but cannot currently create the Habitat-Sim rendering context.
- Engineering result: A6000 now has a dedicated micromamba `home-robot` environment. Import-only validation passes; rendering is not validated yet.
- Engineering result: A6000 `home-robot` import-only preflight now passes with Habitat-Sim/Lab/Baselines and CUDA-visible PyTorch.
- Engineering result: A6000 now has all four OVMM data roots needed for first smoke; rendering/runtime validation is the next blocker.
- Engineering result: A6000 OVMM eval entrypoints should now resolve `home_robot` and `home_robot_sim` without manual `sys.path` injection.
- Engineering result: A6000 OVMM smoke reaches Habitat dataset and simulator initialization, so Python imports and OVMM runtime data are no longer the blocker.
- Engineering result: A6000 rendering is blocked by NVIDIA GL/EGL context validity. The strongest evidence is Magnum reporting 4 EGL devices and a correct CUDA/EGL mapping under the 570.86.10 overlay, followed by `GL::Context: cannot retrieve OpenGL version`.
- Engineering result: A6000 `sana` environment now passes core imports and Pi3 import validation; remaining known gap is `flash-attn`.

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
- A6000 broad `micromamba env update -f src/home_robot_sim/environment.yml` was too slow in the current channel setup; prefer narrow installs with `--override-channels`.
- A6000 narrow Habitat-Sim install with only `aihabitat` and `conda-forge` removed `torchvision` and `pytorch3d`; restore them with exact CUDA 11.7/PyTorch 1.13.1 builds and `--no-deps`.
- A6000 Habitat-Sim import needed system `libOpenGL.so.0`; fixed with `apt-get install -y libopengl0`.
- A6000 Habitat-Baselines install initially upgraded `numpy` to 2.0.2 and selected `moviepy` 2.2.1; pin `numpy==1.23.5` and `moviepy==1.0.3`.
- HF mirror can serve small OVMM episode snapshots, but large LFS-heavy repos can fail through pagination redirects or mirror CDN DNS. Use rsync from a verified working tree when mirror LFS is blocked.
- Remote data repos with runtime files but without `.git/lfs/objects` can show many modified LFS files in Git. Treat this as expected for runtime-only payload copies and validate by hashes instead.
- A6000 first OVMM smoke failure: `ModuleNotFoundError: No module named 'home_robot_sim'` from the eval entrypoint. Fixed by editable local package registration.
- A6000 full `pip install -e src/home_robot` attempted to build `sophuspy==0.0.8` and failed under the current CMake policy. Use `pip install --no-deps -e src/home_robot` and `pip install --no-deps -e src/home_robot_sim`, then install any later missing optional dependency narrowly.
- A6000 HomeRobot geometry imports require `sophus`; fixed with the local shim that re-exports installed `sophuspy`.
- A6000 Habitat-Sim EGL failures evolved as follows:
  - Before `libegl1`: `no EGL devices found`.
  - After `libegl1`: `unable to find CUDA device X among 1 EGL devices in total`.
  - With 570.86.10 overlay: CUDA/EGL mapping succeeds under Magnum validation, then `GL::Context: cannot retrieve OpenGL version`.
- A6000 NVIDIA mismatch: loaded kernel module 570.86.10 but system user-space libraries 570.133.07. Random EGL flags did not fix the final OpenGL context failure.
- A6000 Pi3 direct VCS install hung during GitHub clone; residual remote `pip`/`git` processes were terminated.
- A6000 Pi3 shallow clone fetched Git objects but checkout failed before completion, leaving an unusable working tree with `index.lock`.
- A6000 Pi3 full GitHub archive download exited with code 0 but failed `tar` validation with unexpected EOF. Use source-only archive from the shallow Git object database instead.
- A6000 Pi3 import warns that CUDA-compiled RoPE2D is missing and falls back to a slower PyTorch implementation. This does not block import preflight but may affect Pi3 runtime speed.

## Paper Story

Working claim:

Low-cost rollout scaling can improve online RL for long-horizon mobile manipulation only when its ranking reliability is explicitly measured and corrected, especially under long horizons and contact-rich stages.

Required evidence:

- Cheap-vs-high-precision ranking correlation by stage and horizon.
- Failure cases showing contact-rich ranking collapse.
- Comparison of no RL, high-precision-only rollout, naive cheap rollout, and cheap rollout plus verification/calibration.
- Wall-clock and GPU-hour efficiency, not only success rate.

## Next Milestones

1. Ask for or arrange host-level fix on the A6000 render server: align NVIDIA kernel module and user-space GL/EGL libraries, or reboot into the installed 570.133.07 stack, or use another NVIDIA render host.
2. Continue non-render setup while waiting: install `flash-attn` remotely in the A6000 `sana` env, not locally.
3. Decide whether to transfer 48GB BEHAVIOR champion checkpoints to A6000 or download/copy them from a faster shared source.
4. Build the standard rollout record schema on OVMM once a render host works.
5. Run BEHAVIOR environment check later on a working NVIDIA render machine: headless render, reset, step, observation extraction, and video/keyframe save.
