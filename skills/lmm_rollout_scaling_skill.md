# Skill: LMM Rollout Scaling

Last updated: 2026-06-27

## What this project is about

This project studies online RL with rollout scaling for long-horizon mobile manipulation. The key question is whether cheap rollout can act as a reliable proxy for high-precision rollout, and how to correct it when ranking collapses under long horizons or contact-rich manipulation.

## What this project is not about

- Not generic LMM as a language model project.
- Not pure VLA quantization.
- Not pure world modeling.
- Not only benchmark score chasing.
- Not planner/controller engineering unless it helps answer rollout scaling reliability and online RL utility.
- Not currently an OpenVLA baseline project.
- Not currently a J-EPA/V-JEPA/world-model reward project.
- Not required to use Transformer Engine, NVFP4, or FP4 just because Sol-RL used them for image generation.

## Key terminology

- LMM: Long-horizon mobile manipulation.
- VLA: Vision-language-action policy.
- BEHAVIOR-1K: Main candidate benchmark for household mobile manipulation in Isaac Sim / OmniGibson.
- OVMM: First smoke benchmark for easier mobile-manipulation rollout setup in Habitat/HomeRobot.
- BEHAVIOR champion solution: The Pi0.5/OpenPI-derived BEHAVIOR Challenge winning solution used as the engineering baseline.
- Sol-RL: Rollout-scaling reference from the Sana repo. Its FP4/NVFP4 explore and BF16 train setup is image-generation-specific; use the pattern, not necessarily the exact precision stack.
- cheap rollout: Lower-cost candidate rollout mode, such as quantized policy weights, smaller model, lower resolution, fewer flow/action integration steps, fewer action chunk candidates, cached features, or another fast surrogate.
- high-precision rollout: BF16/FP16/FP32 or otherwise trusted rollout mode used as the reference ranking.
- ranking correlation: Spearman, Kendall tau, top-k overlap, success agreement, false positive and false negative rates between cheap and high-precision candidate rankings.
- rollout scaling: Increasing the number of candidate rollouts to improve policy selection or training signal.
- online RL: Iteratively collecting rollouts from the current policy and updating the policy.
- contact-rich manipulation: Stages where contact, grasp, articulation, placement, and release dominate task success.
- stage-aware verification: Selectively rerunning cheap candidates with high precision based on stage risk, uncertainty, or expected utility.

## Standard workflow

At the beginning of each work session:

1. Read `lmm_rollout_project/memory/project_memory.md`.
2. Read the latest three files under `lmm_rollout_project/logs/daily/`.
3. Check git status for relevant subrepos: `Planning`, `basecode/behavior-1k-solution`, and `rltask/Sana`.
4. Check each relevant repo's latest commit.
5. Identify the current stage: Stage 0 docs, Stage 1 env/baseline, Stage 2 ranking, Stage 3 horizon/contact, Stage 4 method, or Stage 5 online RL.
6. Check `lmm_rollout_project/docs/current_plan.md` before using older notes.
7. State the plan before editing files or running experiments.

Treat `Planning/Doc/Todo.md` as historical exploration context. Do not follow its OpenVLA or J-EPA/V-JEPA route unless the user explicitly changes the project direction.

Current immediate route: OVMM first for the smallest environment loop, then BEHAVIOR champion for the stronger long-horizon platform.

Local resource policy:

- Do not run long `git lfs pull`, `flash-attn` compilation, Habitat/OVMM rendering smoke tests, BEHAVIOR checkpoint loading, or package installs that compile CUDA extensions on the local laptop unless the user explicitly approves.
- Even with explicit approval, stop local CUDA extension builds if swap fills or available memory drops sharply. A 2026-06-23 single-thread `flash-attn` install attempt filled swap and was interrupted.
- Prefer remote NVIDIA RTX server for simulation/rendering and remote larger machines for heavy dependency builds.
- Local work should stay limited to code edits, documentation, small file reads, light imports, and short metadata checks.

## Deployment workflow

Use the deployment scaffold when moving the workspace to remote machines:

```bash
cp lmm_rollout_project/configs/deploy/lmm_deploy.example.env lmm_rollout_project/configs/deploy/lmm_deploy.env
bash lmm_rollout_project/scripts/deploy/deploy_all.sh lmm_rollout_project/configs/deploy/lmm_deploy.env
```

Default deployment syncs code/docs/configs/scripts and excludes checkpoints, datasets, outputs, videos, and virtualenvs. Set `SYNC_CHECKPOINTS=1` only when transferring the 48GB BEHAVIOR checkpoints is intended.

Remote roles:

- NVIDIA render host: BEHAVIOR / OmniGibson / Isaac Sim evaluator and rendering.
- AMD train host: ROCm-compatible training/calibration/analysis after preflight. Access is tunnel-only unless SSH is explicitly enabled.

Do not assume the AMD host can run the champion policy or Sana CUDA setup unchanged.

Current A6000 NVIDIA render server:

```bash
ssh -p 20400 root@219.223.207.18
cd /root/workspace/tianshanzhang
```

Confirmed state on 2026-06-25:

- 4x NVIDIA RTX A6000, driver 570.86.10.
- Code/resources synced to `/root/workspace/tianshanzhang`.
- `git-lfs` installed.
- `/root/anaconda3` exists and `conda init bash` was run.
- NVIDIA render preflight passed.
- Docker is not installed.
- Micromamba exists at `/root/.local/bin/micromamba`, version 2.5.0.
- `home-robot` env was created at `/root/micromamba/envs/home-robot` from `benchmark/home-robot/src/home_robot/environment.yml`.
- `home-robot` import-only preflight passes after Habitat-Sim/Lab/Baselines install and dependency repairs.
- `sana` env exists at `/root/anaconda3/envs/sana`.
- A6000 `sana` core imports pass for PyTorch cu128, xformers, mmcv, editable Sana, Pi3, and flash-attn.
- A6000 `sana` still lacks `transformer-engine[pytorch]`; this blocks only Sol-RL's NVFP4/FP4 configs, not our core robotics route.
- OVMM data roots are now present under `/root/workspace/tianshanzhang/benchmark/home-robot/data`.
- OVMM Python imports and runtime data roots are configured, but headless rendering is blocked by the current NVIDIA GL/EGL driver stack.
- BEHAVIOR checkpoints were intentionally not synced.

After rsyncing to root-owned remote workspaces:

- If Git reports `dubious ownership`, fix ownership or safe.directory before using VS Code Git integration.
- If HomeRobot shows many LFS assets as modified, install `git-lfs` and recheck before assuming real code changes.

For AMD tunnel deployment:

```bash
bash lmm_rollout_project/scripts/deploy/package_for_tunnel.sh lmm_rollout_project/configs/deploy/lmm_deploy.env
```

Upload the tarball in VS Code Tunnel and run:

```bash
cd /scratch/$USER/longhorizon
bash lmm_rollout_project/scripts/deploy/remote_preflight_amd_train.sh
```

AMD full-node Slurm allocation:

```bash
salloc --reservation=gpu-4_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_gpu-29_reservation --exclusive --mem=0
```

At the end of each work session:

1. Update `lmm_rollout_project/logs/daily/YYYY-MM-DD.md`.
2. Update `lmm_rollout_project/memory/project_memory.md`.
3. Update this skill file when commands, workflow, or failure diagnostics change.
4. If experiments ran, record full metadata.
5. If code changed, state purpose, files, risk, verification, and proposed commit message.

## How to run OVMM environment check

Local OVMM smoke command, only with explicit approval because local rendering has caused instability:

```bash
bash lmm_rollout_project/scripts/env_check/ovmm_local_smoke.sh
```

Known local status:

- `home-robot` conda env exists.
- Host execution sees CUDA from PyTorch.
- OVMM episodes exist under `benchmark/home-robot/data/datasets/ovmm`.
- Local Habitat-Sim currently fails at OpenGL/EGL context creation with `GL::Context: cannot retrieve OpenGL version`.

Sol-RL local preflight, safe if it only imports already installed packages:

```bash
bash lmm_rollout_project/scripts/env_check/solrl_local_preflight.sh
```

Do not run Sana `environment_setup.sh` or `flash-attn` compilation locally. The local `sana` env has the main packages but is missing `flash_attn`; the A6000 `sana` env is the validated remote environment for flash-attn.

Current local Sana status:

- `torch 2.9.1+cu128`, `torchvision 0.24.1+cu128`, `torchaudio 2.9.1+cu128`.
- `sana`, `diffusers`, `transformers`, `xformers`, `mmcv`, `bitsandbytes` installed.
- `flash_attn` missing.
- `transformer-engine[pytorch]` missing and only required for NVFP4 / FP4 Sol-RL configs.

A6000 Sana/Sol-RL status:

- Env path: `/root/anaconda3/envs/sana`.
- Python: 3.11.6 after CUDA toolkit solve.
- CUDA toolkit: 12.8, `nvcc 12.8`.
- PyTorch stack: `torch 2.9.1+cu128`, `torchvision 0.24.1+cu128`, `torchaudio 2.9.1+cu128`.
- Installed and import-validated: `xformers 0.0.33.post2`, `mmcv 1.7.2`, editable `sana 0.2.0`, `diffusers`, `transformers`, `accelerate`, `bitsandbytes`, `clip`, `peft`, `timm`, `hpsv2`, `open_clip`, `wandb`, `gradio`, and `pi3 0.1`.
- Installed and import-validated: `flash-attn 2.8.3.post1` and compiled extension `flash_attn_2_cuda`.
- Not installed: `transformer-engine[pytorch]`. It is optional for NVFP4/FP4 Sol-RL paths and currently needs a different wheel/torch strategy.
- `mmcv._ext` is absent; pure-Python `mmcv` config imports pass. Only revisit if a runtime import needs compiled mmcv ops.
- Pi3 import warns that CUDA-compiled RoPE2D is missing and falls back to slow PyTorch RoPE2D. This does not block import preflight.

A6000 Sana quick validation:

```bash
/root/anaconda3/envs/sana/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
/root/anaconda3/envs/sana/bin/python -c "import sana, diffusers, transformers, xformers, bitsandbytes, pi3; print('SANA_CORE_IMPORT_OK')"
/root/anaconda3/envs/sana/bin/python -c "import pi3; import pi3.models.pi3; import pi3.models.pi3x; import pi3.utils.geometry; print('PI3_IMPORT_OK')"
/root/anaconda3/envs/sana/bin/python -c "import torch, xformers, mmcv, sana, pi3, flash_attn, flash_attn_2_cuda; print('SANA_PREFLIGHT_OK', torch.__version__, torch.version.cuda, flash_attn.__version__)"
```

A6000 Pi3 source-only install workflow:

```bash
cd /root/workspace/tianshanzhang
git --git-dir=/root/workspace/tianshanzhang/externals/Pi3_shallow/.git archive \
  --format=tar \
  --prefix=Pi3_src/ \
  -o /root/workspace/tianshanzhang/externals/Pi3_src_sparse.tar \
  HEAD pi3 pyproject.toml requirements.txt LICENSE README.md
tar -xf /root/workspace/tianshanzhang/externals/Pi3_src_sparse.tar -C /root/workspace/tianshanzhang/externals
/root/anaconda3/envs/sana/bin/python -m pip install --no-deps /root/workspace/tianshanzhang/externals/Pi3_src
```

Use this source-only workflow because direct `pip install git+https://github.com/yyfz/Pi3.git --no-deps` hung during clone, shallow Git checkout did not complete, and full GitHub archive downloads produced an invalid gzip file in this network environment.

A6000 flash-attn build workflow:

```bash
cd /root/workspace/tianshanzhang/rltask/Sana
timeout 7200s env \
  PATH=/root/anaconda3/envs/sana/bin:$PATH \
  CUDA_HOME=/root/anaconda3/envs/sana \
  CUDA_PATH=/root/anaconda3/envs/sana \
  CPATH=/root/anaconda3/envs/sana/targets/x86_64-linux/include:/root/anaconda3/envs/sana/include \
  LIBRARY_PATH=/root/anaconda3/envs/sana/targets/x86_64-linux/lib:/root/anaconda3/envs/sana/lib \
  LD_LIBRARY_PATH=/root/anaconda3/envs/sana/targets/x86_64-linux/lib:/root/anaconda3/envs/sana/lib:$LD_LIBRARY_PATH \
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  MAX_JOBS=8 \
  NVCC_THREADS=2 \
  /root/anaconda3/envs/sana/bin/python -m pip install -v --timeout 60 --retries 1 --progress-bar off --no-build-isolation "flash-attn>=2.7.0"
```

This recipe is needed because non-activated SSH commands do not put env `bin/` on `PATH`, and flash-attn needs CUDA headers from `/root/anaconda3/envs/sana/targets/x86_64-linux/include`.

A6000 transformer-engine status:

```bash
/root/anaconda3/envs/sana/bin/python -m pip show transformer-engine transformer-engine-torch transformer-engine-cu12
```

Current status: not installed. The attempted `transformer-engine[pytorch]` install selected `transformer-engine 2.16.1`, did not find a matching precompiled `transformer_engine_torch` wheel for `torch 2.9.1+cu128`, failed once on missing `cudnn.h`, and then became unobservable during an explicit-path retry. Do not repeat the same source-build attempt; use a compatible prebuilt wheel/torch pairing or defer NVFP4.

Project decision:

- Defer NVFP4 unless explicitly needed.
- Do not make Transformer Engine part of the critical path.
- For our VLM / flow-matching / action-chunk policies, define cheap rollout modes directly from the robotics stack.
- Candidate cheap modes: INT8/INT4 or weight-only quantization of VLM/action policy components, BF16/FP16 vs FP32 comparison, fewer flow/action integration steps, fewer generated action chunks, lower observation resolution, smaller draft policy, cached language/visual embeddings, or stage-specific cheap proxies.
- High-precision verification should be used for contact-rich or high-risk stages when cheap ranking is uncertain.

A6000 HomeRobot env commands:

```bash
ssh -p 20400 root@219.223.207.18
cd /root/workspace/tianshanzhang/benchmark/home-robot
/root/.local/bin/micromamba env list
/root/.local/bin/micromamba run -n home-robot python --version
/root/.local/bin/micromamba run -n home-robot python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
/root/.local/bin/micromamba run -n home-robot python /root/workspace/tianshanzhang/lmm_rollout_project/scripts/env_check/a6000_home_robot_import_preflight.py
```

A6000 OVMM data-root check:

```bash
ssh -p 20400 root@219.223.207.18 \
  du -sh /root/workspace/tianshanzhang/benchmark/home-robot/data/hssd-hab \
         /root/workspace/tianshanzhang/benchmark/home-robot/data/objects \
         /root/workspace/tianshanzhang/benchmark/home-robot/data/datasets/ovmm \
         /root/workspace/tianshanzhang/benchmark/home-robot/data/robots/hab_stretch
```

Do not run the full HomeRobot `install_deps.sh` blindly. It downloads datasets/checkpoints and installs multiple third-party stacks. Use import-only checks first, then install missing editable packages or specific missing dependencies one at a time.

A6000 HomeRobot/Habitat install recipe used successfully on 2026-06-26:

```bash
cd /root/workspace/tianshanzhang/benchmark/home-robot
/root/.local/bin/micromamba env create -y -n home-robot -f src/home_robot/environment.yml
/root/.local/bin/micromamba install -y -n home-robot --override-channels -c aihabitat -c conda-forge habitat-sim=0.2.5 withbullet
apt-get update
apt-get install -y libopengl0
/root/.local/bin/micromamba install -y --no-deps -n home-robot --override-channels -c pytorch -c pytorch3d -c nvidia -c conda-forge torchvision=0.14.1=py39_cu117 pytorch3d=0.7.5=py39_cu117_pyt1131
/root/.local/bin/micromamba run -n home-robot python -m pip install -e src/third_party/habitat-lab/habitat-lab
/root/.local/bin/micromamba run -n home-robot python -m pip install -e src/third_party/habitat-lab/habitat-baselines
/root/.local/bin/micromamba run -n home-robot python -m pip install numpy==1.23.5 moviepy==1.0.3
/root/.local/bin/micromamba run -n home-robot python -m pip check
/root/.local/bin/micromamba run -n home-robot python /root/workspace/tianshanzhang/lmm_rollout_project/scripts/env_check/a6000_home_robot_import_preflight.py
```

Register HomeRobot local packages before using eval entrypoints:

```bash
cd /root/workspace/tianshanzhang/benchmark/home-robot
/root/.local/bin/micromamba run -n home-robot python -m pip install --no-deps -e src/home_robot
/root/.local/bin/micromamba run -n home-robot python -m pip install --no-deps -e src/home_robot_sim
/root/.local/bin/micromamba run -n home-robot python -c "import home_robot, home_robot_sim; print('HOME_ROBOT_EDITABLE_IMPORT_OK')"
/root/.local/bin/micromamba run -n home-robot python /root/workspace/tianshanzhang/lmm_rollout_project/scripts/env_check/a6000_home_robot_import_preflight.py
```

Use `--no-deps` for these editable installs. A full `pip install -e src/home_robot` tries to build `sophuspy==0.0.8`, which failed on the A6000 under the current CMake policy. Install later missing runtime dependencies narrowly instead of reopening the full dependency solve.

Add the project shim path for OVMM eval commands:

```bash
export PYTHONPATH=/root/workspace/tianshanzhang/lmm_rollout_project/shims:$PYTHONPATH
```

The shim `lmm_rollout_project/shims/sophus.py` re-exports installed `sophuspy` under the legacy module name `sophus`, which HomeRobot imports from `home_robot.utils.geometry._base`.

Bounded A6000 OVMM random-agent smoke:

```bash
ssh -p 20400 root@219.223.207.18 \
  'cd /root/workspace/tianshanzhang/benchmark/home-robot && PYTHONPATH=/root/workspace/tianshanzhang/lmm_rollout_project/shims:$PYTHONPATH timeout 240s /root/.local/bin/micromamba run -n home-robot python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_demo.yaml --agent_type random --num_episodes 1 habitat.environment.max_episode_steps=1 habitat.simulator.habitat_sim_v0.gpu_device_id=0'
```

Current A6000 OVMM render status:

- Baseline with system user-space NVIDIA libs reaches dataset/simulator init but fails in EGL.
- `libegl1` is installed and `libEGL.so.1` is discoverable.
- Loaded NVIDIA kernel module is 570.86.10, but system `libnvidia-gl-570` / `libnvidia-compute-570` are 570.133.07.
- A diagnostic 570.86.10 runtime overlay exists at `/root/workspace/tianshanzhang/runtime_libs/`.
- With the overlay and `MAGNUM_GPU_VALIDATION=ON`, Magnum reports 4 EGL devices and maps CUDA device 0 to EGL device 0, then fails with `GL::Context: cannot retrieve OpenGL version`.
- `CUDA_VISIBLE_DEVICES=0`, `EGL_PLATFORM=surfaceless`, and `PYOPENGL_PLATFORM=egl` did not fix the final OpenGL context failure.
- Do not continue random EGL flag iteration on this server unless the host-level NVIDIA kernel/user-space library mismatch is fixed.

Diagnostic overlay env, for reproducing the current failure only:

```bash
export LD_LIBRARY_PATH=/root/workspace/tianshanzhang/runtime_libs/nvidia57086_cuda:/root/workspace/tianshanzhang/runtime_libs/nvidia57086_gl/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=/root/workspace/tianshanzhang/runtime_libs/nvidia57086_gl/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MAGNUM_GPU_VALIDATION=ON
```

Pitfalls from the A6000 setup:

- Broad `micromamba env update -f src/home_robot_sim/environment.yml` spent over 13 minutes solving and was terminated.
- `--override-channels` made Habitat-Sim solve quickly, but omitting PyTorch channels caused `torchvision` and `pytorch3d` to be removed.
- Habitat-Sim import needed the system package `libopengl0`.
- `habitat-baselines` initially pulled `numpy 2.0.2` and `moviepy 2.2.1`; repair with `numpy==1.23.5` and `moviepy==1.0.3`.
- `eval_baselines_agent.py` does not receive the manual `sys.path` injection used by the import preflight script. Install `src/home_robot` and `src/home_robot_sim` editable before eval.
- Full editable dependency install of `src/home_robot` can fail while compiling `sophuspy`; use editable `--no-deps` registration and keep dependency repairs narrow.
- Installed `sophuspy 1.2.0` exports module name `sophuspy`, while HomeRobot imports `sophus`. Use the project shim path.
- A6000 render failure progression:
  - Missing `libegl1`: `no EGL devices found`.
  - After `libegl1`: `unable to find CUDA device X among 1 EGL devices in total`.
  - With 570.86.10 overlay: EGL/CUDA mapping succeeds, then `GL::Context: cannot retrieve OpenGL version`.
- This progression points to a host/container NVIDIA GL/EGL stack issue, not a missing OVMM dataset or Python dependency.
- `OVMM_episodes` can be downloaded through `HF_ENDPOINT=https://hf-mirror.com` with `huggingface_hub.snapshot_download`.
- `OVMM_objects` is LFS-heavy. In this environment, mirror snapshot pagination redirected to `huggingface.co`, and mirror Git LFS payload URLs used the unresolved host `us.aws.cdn.hf-mirror.org`. Use mirror Git metadata plus rsynced local payload, or rsync a verified working tree.
- `hssd-hab` has about 56,669 runtime files. Use clean working-tree rsync excluding `.git`; validate by runtime size and sample hashes.
- If `.git/lfs/objects` is not copied, remote Git status can show many modified LFS files even when runtime files are valid. Do not use `git status` as the payload-integrity check for runtime-only data copies.

## How to run BEHAVIOR environment check

Do not run BEHAVIOR / Isaac Sim checks on weak local machines. Use a simulation machine with a supported NVIDIA RTX GPU.

Minimum environment check must verify:

- Isaac Sim headless/offscreen startup.
- GPU and driver visibility.
- BEHAVIOR/OmniGibson import.
- Environment reset.
- One or more simulation steps.
- RGB observation extraction.
- Video or keyframe save.
- No snow screen, black screen, or segmentation fault.

Record output under:

```text
lmm_rollout_project/experiments/env_check/
lmm_rollout_project/logs/daily/YYYY-MM-DD.md
lmm_rollout_project/docs/env_setup.md
```

## How to run baseline rollout

Baseline rollout should be the smallest complete episode possible:

- Use a trivial policy, rule-based policy, or champion websocket policy.
- Save observation trace, action trace, reward/progress trace, success/failure flag, and video/keyframes.
- Record seed, task, machine, GPU, driver, CUDA/ROCm, Isaac Sim version, Python environment, checkpoint, and log path.

Output should go under:

```text
lmm_rollout_project/experiments/baseline_rollout/
lmm_rollout_project/docs/baseline_rollout.md
lmm_rollout_project/logs/daily/YYYY-MM-DD.md
```

## How to run cheap rollout

Cheap rollout mode must be explicitly named and logged. Candidate modes include:

- Low precision inference.
- Quantized model or rollout surrogate. NVFP4 is optional; INT8/INT4/weight-only quantization, fewer flow/action steps, smaller draft policies, or lower-resolution observations can also define cheap rollout.
- Lower image resolution.
- Fewer denoising/action sampling steps.
- Smaller policy.
- Cached or approximate scoring policy.

Never assume cheap rollout is reliable. Always log speed, memory, success, reward/progress, stage metrics, and failure mode.

## How to run high-precision rollout

High-precision rollout is the reference mode for ranking. It should use the strongest feasible policy precision or the closest available trusted mode.

Record:

- dtype / precision mode.
- model checkpoint.
- inference latency.
- rollout wall-clock time.
- GPU memory.
- task, seed, candidate ID, and initial state.

## How to compute ranking correlation

For matching tasks, seeds, initial states, and candidate branches:

- Run cheap rollout for N candidates.
- Run high-precision rollout for the same candidate identities.
- Compute Spearman correlation.
- Compute Kendall tau.
- Compute top-k overlap.
- Compute success agreement.
- Compute false positive rate: cheap ranks good, high precision fails.
- Compute false negative rate: cheap ranks bad, high precision succeeds.
- Break down metrics by stage and horizon.

Store tables under `lmm_rollout_project/results/tables/` and plots under `lmm_rollout_project/results/figures/`.

## How to record experiments

Every experiment must have an ID:

```text
exp_YYYYMMDD_001
```

Record at least:

- Goal.
- Machine.
- GPU.
- Driver.
- CUDA or ROCm.
- Isaac Sim version.
- Python environment.
- Commit hash and dirty status.
- Config.
- Seed.
- Precision mode.
- Task.
- Number of episodes.
- Log path.
- Checkpoint path.
- Video path.
- Runtime.
- GPU memory.
- Failure mode.

## How to diagnose failures

Common early failures:

- Checkpoint load killed with exit code 137: likely system RAM or GPU memory pressure. Do not debug as Python logic unless a traceback exists.
- JAX sees only CPU: check whether command ran in sandbox or host environment.
- Websocket server not reachable: check host networking, `ss -ltnp`, firewall, SSH tunnel, and policy server bind host.
- BEHAVIOR mapping points to missing checkpoints: inspect `task_checkpoint_mapping*.json`.
- Isaac Sim black/snow screen: check GPU has RT cores, driver, headless/offscreen settings, container runtime, and Isaac Sim version.
- `ModuleNotFoundError: home_robot_sim` in OVMM eval: install `src/home_robot` and `src/home_robot_sim` editable with `--no-deps` in the active env, then rerun the import preflight.
- `ModuleNotFoundError: sophus` in OVMM eval: ensure `PYTHONPATH` includes `/root/workspace/tianshanzhang/lmm_rollout_project/shims`, then rerun the import preflight.
- `GL::Context: cannot retrieve OpenGL version` on A6000 after EGL/CUDA mapping succeeds: stop env-var iteration and fix the host NVIDIA GL/EGL stack or switch render machine.

## Common errors and fixes

- `~/models/checkpoint_*` missing: create or use a local mapping file pointing to actual checkpoint paths.
- Local 8GB GPU cannot load champion checkpoint: use a larger server; local machine is for code and docs only.
- `transformer_engine` missing in Sol-RL NVFP4 modes: install `transformer-engine[pytorch]` in the same Python interpreter as `torchrun`.
- `transformer-engine[pytorch]` on A6000 with `torch 2.9.1+cu128`: current direct pip install is not solved. Core Sana works without it; NVFP4/FP4 configs should be treated as unavailable until a compatible TE wheel/torch strategy is chosen.
- HPSv2 reward checkpoint missing: place required files under `reward_ckpts/`.
- `sophuspy==0.0.8` CMake policy failure during HomeRobot editable install: avoid the full dependency install path; use `pip install --no-deps -e src/home_robot` after the conda env already contains the required runtime dependencies.
- NVIDIA kernel/user-space mismatch on A6000: kernel module 570.86.10 but system user-space libraries 570.133.07. A runtime overlay can reproduce/diagnose, but a durable fix likely needs admin-level driver/library alignment or reboot.

## Paper-story alignment checklist

Before running an experiment, ask whether it answers at least one:

- Does it measure cheap-vs-high-precision agreement?
- Does it test horizon-driven ranking collapse?
- Does it isolate contact-rich failure modes?
- Does it compare naive cheap rollout scaling to verified/calibrated scaling?
- Does it support a main figure, table, ablation, or failure analysis?

If the answer is no, deprioritize the experiment.
